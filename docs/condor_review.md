# Condor Submission Infrastructure Review — BTVNanoCommissioning

Based on a structural/simplify/performance investigation of `condor/execute.sh`, `condor/submitter.py`,
`condor_lxplus/execute.sh`, `condor_lxplus/submitter.py`, and `scripts/fetch.py`'s redirector/site-selection
logic. These files were explicitly out of scope for the four earlier reviews
(`architecture_review.md`, `performance_review.md`, `io_review.md`, `simplify_review.md`), which stop at
`runner.py`/`correction.py`/workflow code. `docs/modernization_plan.md`'s "Post-plan additions" section
found two live bugs here (`condor/execute.sh`'s unbounded micromamba `wget`, `fetch.py` not preferring a
local EOS path) and asked for a dedicated pass over the rest of this area — this is that pass.

None of this touches physics-producing code; everything below is submission/setup/dataset-discovery
infrastructure.

---

## Data-locality note (read first)

Before the itemized findings: this repo has a stated goal that jobs read **nothing but ROOT files via
xrootd** during execution — the environment and code should be fully local to the worker node, not read
live off an eos/afs mount. The two condor variants currently disagree on this:

- **`condor/` (the `remoteRepo`/tarball variant) already satisfies it.** `condor/execute.sh:36-65`
  bootstraps micromamba entirely on local worker-node disk (the `HOME=$(pwd)` override at line 34 exists
  specifically to keep the installer writing to local scratch, not AFS). The code arrives via `git clone`
  or by extracting a tarball shipped through HTCondor's own `transfer_input_files` (lines 68-76) — never
  an eos/afs mount reference at runtime. The only network I/O during the job is the mamba/pip installs,
  the code fetch, and `runner.py`'s own xrootd reads of ROOT files.
- **`condor_lxplus/` does not, by design.** `condor_lxplus/execute.sh:21` prepends `$ENVPATH` to `PATH`,
  and the submitter (`condor_lxplus/submitter.py:225`) defaults `ENVPATH` to
  `/eos/home-m/milee/miniforge3/envs/btv_coffea/bin` — every Python interpreter start and import for the
  whole job runs off that eos mount over the network. `cd $COMMDIR` (`condor_lxplus/execute.sh:17`) also
  puts the job's working directory inside the shared eos/afs checkout itself, so `correction.py`, the SF
  payload files under `src/BTVNanoCommissioning/data/`, etc. are read live over the mount too. The proxy
  is the same story: `X509_USER_PROXY=$COMMDIR/proxy` (`condor_lxplus/execute.sh:18`) reads it from that
  mount rather than being condor-delegated (`use_x509userproxy = true`, as `condor/submitter.py:337`
  does) or transferred in. `condor_lxplus/README.md` documents this as the intentional tradeoff of the
  "lighter-weight" variant ("good for up to a few hundred files" relying on eos/afs mounts) — a real
  design choice, not a bug, but the direct opposite of the stated goal.

**None of the findings below change this axis either way** — they're orthogonal bug fixes. Making
`condor_lxplus` satisfy the locality goal would mean giving it `condor/`'s architecture (ship code +
bootstrap env locally, delegate/transfer the proxy instead of reading it off the mount), which mostly
collapses the two directories into one design and raises the question of why both would still exist
separately. Flagged here as a decision to make deliberately, not folded into a fix.

---

## Simplify findings

### S1. `condor_lxplus/submitter.py`: dead `make_tarfile()` and vestigial `--remoteRepo`

**Current complexity:** `make_tarfile()` (lines 9-17) has exactly one call site, and that call site
(lines 175-200) is entirely commented out — this variant never tars anything; it runs jobs directly off
the eos/afs mount per `condor_lxplus/README.md`'s design. `--remoteRepo` (parser arg at line 58, print at
line 176) is decorative: `condor_lxplus/execute.sh:27` reads it into `ARGS[remoteRepo]` but never branches
on it afterward. Both are leftovers from copy-pasting `condor/submitter.py`, where the same code is
load-bearing.

**Simplification:** delete the dead function, the commented block, and the unused `--remoteRepo`
plumbing — or restore it as real, tested behavior if it's intentionally staged rather than abandoned
(worth asking, don't assume).

**Risk:** None if truly unreachable (confirmed via grep — no other importers of this `make_tarfile`).

### S2. `condor_lxplus/submitter.py` still uses bare `input()` instead of the EOFError-safe `_prompt_yn()`

**Current complexity:** `condor/submitter.py:8-16` added `_prompt_yn()` specifically because scripts that
call the submitter non-interactively (e.g. a driver script looping over samples) have no stdin, so
`input()` raises `EOFError` and crashes instead of taking the sane bracketed-default. `condor_lxplus/submitter.py`
(lines 204, 235, 269) still uses bare `input()` at every prompt, so it's still exposed to exactly the
crash this fix in its sibling was written to prevent.

**Simplification:** port `_prompt_yn()` over verbatim; same file family, no design change.

**Risk:** Low — only changes the crash-on-EOF case, behavior-preserving for interactive use.

### S3. `scripts/fetch.py`: the global-redirector fallback dict is copy-pasted 3 times

**Current complexity:** `{"infn": "root://xrootd-cms.infn.it//", "fnal": "root://cmsxrootd.fnal.gov/",
"cern": "root://cms-xrd-global.cern.ch/"}` appears verbatim at lines ~1396, ~1493, and ~1522, all within
~130 lines of the site-selection logic.

**Simplification:** hoist to one module-level constant (e.g. `GLOBAL_REDIRECTORS`). Purely mechanical.

**Risk:** None.

### S4. `scripts/fetch.py`: duplicated CI-vs-local subprocess-wrapper generation

**Current complexity:** the XRootD-ping helper (~lines 660-815) and `run_xrdfs_command()` (~lines
817-928) each independently write out a temp bash wrapper (and, for the first, a temp Python script) with
near-identical proxy-detection, `XRD_*` env-var export, and CMS-environment-sourcing boilerplate (~40
duplicated lines in each).

**Simplification:** factor the shared "CI subprocess wrapper" (proxy detection + env export + cvmfs
source) into one helper both call into; keep only the actual ping command as the difference.

**Risk:** Low-medium — this path only runs in CI (`determine_execution_mode()`), so it's cheap to test in
isolation, but temp-file lifecycle and exit-code plumbing are fiddly enough to warrant an actual CI run
before merging, not just a local check.

---

## Performance findings

### P1. `check_site_responsiveness()` probes every file in the dataset, sequentially, per candidate site

**Location:** `scripts/fetch.py:1021-1094`, called from the main site-selection loop at line 1471 with
`dataset_files=flist` (the full file list).

**Problem:** When given `dataset_files`, the function tests **every file** in the dataset against a
candidate site — `open` then `stat`, each with a 30-40s timeout — before concluding the site is
responsive, and repeats this for the next candidate (in completion order) if the current one fails on any
single file ("fail fast" stops on the *first* inaccessible file, but only after however many succeeded
before it). For a dataset with hundreds/thousands of files this is O(N) serial XRootD round-trips just to
answer "is this site up," and it sits on the critical path of every `fetch.py` invocation for a
not-already-cached dataset.

**Expected speedup:** Large. `access_xrootd_file(..., check_all_variants=True)` already probes both open
and stat per file, so a small fixed sample (e.g. 5 files) is a statistically fine responsiveness probe —
testing all N is doing roughly N/5 the useful work for the same conclusion in the common case. Could turn
a multi-minute site-selection phase into single-digit seconds.

**Confidence:** High that this is expensive as written; medium on the exact wall-clock win, since it
depends on dataset size and how often a fail-fast abort happens early vs. late in the file list.

**Implementation effort:** Low — cap the list passed in at the call site (`flist[:5]` at line 1471
instead of the full `flist`); no change needed inside `check_site_responsiveness` itself.

**Risk:** Low-moderate, and it's a correctness-of-*selection* risk, not a physics risk: a smaller sample
could occasionally pick a site that's actually incomplete for files outside the sample.
`check_site_completion()` (percentage-of-dataset-present, already the primary quality gate) mitigates
this, and `--skipbadfiles`/retries remain the existing safety net per `modernization_plan.md`'s note on
the global-redirector fallback.

**Files:** `scripts/fetch.py`.

### P2. `condor/execute.sh`'s mamba bootstrap reinstalls the full environment from scratch every job — decided fix: ship a packed env from EOS instead

**Location:** `condor/execute.sh:36-65`.

**Problem:** When `remoteRepo` is set, every job installs micromamba, `python=3.10`, `xrootd`, and
`setuptools=70.1.1` from `conda-forge` from a clean worker-node scratch dir — this is the "redundant
per-job installs" category the modernization plan's remark anticipated beyond the wget-timeout fix
already applied. Beyond wall-clock cost, this also re-resolves the dependency graph against
conda-forge's *current* state on every job, so nothing pins the environment — a solve run in January and
one run in June could silently differ in transitive package versions with no one requesting a change.

**Decided direction — build once, ship via EOS, fetch over xrootd, not HTCondor's `transfer_input_files`:**

1. Build the `btv_coffea` environment once (as done today) and pack it with `conda-pack`/`micromamba
   pack` into a single relocatable tarball.
2. Stage that tarball on EOS (a fixed, versioned path — not the AFS-hosted checkout, which typically has
   much tighter quota).
3. Replace `condor/execute.sh:36-65` with an `xrdcp`/`root://` fetch of that tarball plus an unpack step,
   wrapped in the same bounded-retry pattern the script already uses for the current `wget` loop
   (lines 39-46) and timed with the existing `phase` markers.
4. Rebuild and re-stage the tarball deliberately when dependencies actually change; jobs always pull
   "current" (or a version-pinned path) at start time.

**Why EOS+xrootd over `transfer_input_files`:** the packed env (ROOT/xrootd + coffea + correctionlib +
numpy/awkward/numba + everything else in `test_env.yml`) is realistically 500MB-1.5GB — 15-40x the
current ~38MB code tarball. Routing that through HTCondor's own file transfer means every job in a
submission burst pulls a ~1GB blob from the single submission host (the AP/schedd) simultaneously: a
thundering-herd load on one machine's disk/network, and a poor fit for AFS-hosted checkouts with small
quotas. EOS is built for exactly the opposite profile — many concurrent clients reading a large blob
across a distributed storage backend — and fetching it via `xrdcp`/`root://` (not a mounted `/eos/...`
path) keeps the "nothing but xrootd, everything else local to the worker" principle from the data-locality
note above: the env becomes a local, unpacked directory on worker scratch after the fetch, exactly like
the code tarball is today, just delivered over xrootd instead of HTCondor's transfer mechanism. This also
reuses a path the script already trusts — `condor/execute.sh` already does the mirror-image operation
(`xrdcp` stage-out to `root://` output dirs, lines 132-145) — rather than introducing a new access
pattern. (Worth a one-time check, not a blocker: if the lxplus/cmsconnect HTCondor pool has an xrootd file-transfer
plugin enabled, `transfer_input_files` could point at a `root://` URL directly and let the worker fetch it
without relaying through the AP — potentially the best of both. Unverified for this pool; the explicit
`xrdcp`-in-`execute.sh` approach above doesn't depend on that being available.)

**Expected benefit:** Removes the conda-forge solve/download and the `micro.mamba.pm` dependency
per job entirely (not just bounding its timeout, which is the already-applied stopgap) — turns a
multi-minute (or, on a bad day, multi-hour) per-job cost into a single fixed-size fetch, and makes the
environment reproducible/pinned instead of re-resolved every run.

**Confidence:** Medium-high on the win existing; the exact wall-clock number should still be checked via
the existing `phase mamba` timer (`grep 'PHASE mamba'` across recent job `.out` files) before/after, to
confirm the magnitude.

**Implementation complexity:** Medium — needs a small "build+stage the packed env" script/step (new,
run manually or in CI when deps change) plus simplifying `condor/execute.sh:36-65` to a fetch+unpack.
No change needed to `condor/submitter.py` beyond possibly parameterizing the EOS path.

**Risk:** Low-medium. `conda-pack`/`micromamba pack` have known gotchas with hardcoded paths inside
activation scripts — must verify the unpacked env runs identically on a fresh worker node before trusting
it (a one-time correctness check on the packed artifact, not a per-job physics risk, since it's the same
resolved environment either way once verified).

**Files:** `condor/execute.sh` (replace lines 36-65), new packaging script (not yet written), no change
required to `condor/submitter.py` for the minimal version.

### P3. Not reviewed: `condor_lxplus/normalizeArrays.py`, `condor_lxplus/haddFinal.py`

Both have uncommitted/untracked changes in the working tree (`git status`) — same situation
`modernization_plan.md` already noted for `array_writer.py` ("wait until those settle"). Deferred rather
than reviewed mid-edit.

---

## Architecture findings

### A1. `condor/execute.sh` lacks `-e`; this is the confirmed root cause of the opaque-hold bug

**Problem:** `condor/execute.sh:1` (`#!/bin/bash -x`, no `-e`) vs. `condor_lxplus/execute.sh:1`
(`#!/bin/bash -xe`) — the two variants disagree on the most basic failure-propagation setting, and it's
backwards from what you'd want: `condor/execute.sh` is the heavier variant (micromamba bootstrap, git
clone/tar extraction, pip install — far more that can fail) yet it's the one relying on manually
remembering a `|| { echo ERROR; exit 1; }` after each guarded step, rather than a structural guarantee.
This is exactly the pattern that let the `.success`-file / opaque-hold bug from
`modernization_plan.md`'s post-plan section happen: an unguarded step fails, execution continues,
`.success` is never written, and HTCondor reports "transfer output files failure" instead of the real
cause.

**Why it matters:** Not hypothetical — it's the confirmed root cause of currently-held jobs per the plan
(`jobs_CustomSummer24_2D_e_Wc_sf_data`). `-e` would catch this class of bug structurally instead of
requiring every future line author to remember to add a guard.

**Proposed direction:** Add `-e` to `condor/execute.sh`'s shebang, matching `condor_lxplus/execute.sh`.
First audit for spots where a non-zero exit is *expected and handled* (e.g. the `wget` retry loop at
lines 39-46, which reads `$?` after a command that's allowed to fail) so `-e` doesn't break the
intentional retry logic there. Small and mechanical once that audit is done, not a rewrite.

**Risk:** Medium in the short term (an intentionally-tolerated failure elsewhere could now abort early
where it previously didn't); low once the retry-loop audit is done. Setup-script-only, no physics impact.

**Priority:** High — addresses a confirmed, currently-live operational bug, cheap relative to payoff.

### A2. `condor_lxplus/submitter.py` is missing the `--outputDir` locality check that `condor/submitter.py` added

**Problem:** `condor/submitter.py:205-211` validates that `--outputDir` is an absolute path or a
`root://` URL — added specifically because a relative path silently writes to worker-node scratch space
and is lost for good, since only `.success` is transferred back via HTCondor and the real output is
cp'd/xrdcp'd directly from the worker. `condor_lxplus/submitter.py` has no equivalent check on the same
`--outputDir` argument, despite `condor_lxplus/execute.sh` having the identical "output leaves the worker
directly, only `.success` comes back" shape that made the check necessary in the first place.

**Why it matters:** The exact silent-data-loss failure mode the guard in `condor/submitter.py` exists to
prevent is still reachable through the sibling script — a bug fix landed in one twin and never
propagated to the other (same shape as the `_prompt_yn` gap in S2).

**Proposed direction:** Port the same absolute-path-or-`root://` check into `condor_lxplus/submitter.py`,
right after job-dir creation, mirroring `condor/submitter.py:205-211`.

**Risk:** Low — purely additive validation; only rejects submissions that would already lose their
output.

**Priority:** Medium-high — cheap, and prevents a silent (not just noisy) failure, the worse failure
mode.

### A3. `check_site_responsiveness()` conflates redirector-shape parsing with the probe-and-decide loop

**Problem:** `scripts/fetch.py:1096-1140` — before any actual network probing starts, ~150 lines
(1021-1140 total) handle three different responsibilities in one function: (a) normalizing four distinct
redirector-prefix shapes (`list`, dict-of-regex-rules, plain `str`, dict without a recognizable pattern),
(b) picking a representative test path when no `dataset_files` were passed, and (c) driving the
open/stat probe loop itself.

**Why it matters:** This is the function most likely to need a fifth redirector shape added someday
(SITECONF formats aren't static), and that would mean editing deep inside a function that's also
responsible for the timeout/retry/fail-fast probing logic — two concerns that change for unrelated
reasons.

**Proposed direction:** Extract the redirector-shape normalization (`sites_xrootd_prefix[site]` → one
canonical test URL/server string) into its own small function, separate from the probe-and-decide loop.
Not a call for a bigger abstraction — one extracted function, not a class hierarchy.

**Risk:** Low if done as a pure extraction (same logic, moved), not a rewrite.

**Priority:** Low — maintainability observation, no bug attached; fix opportunistically when next
touching this function.

---

## Not re-litigated (already tracked elsewhere)

- `make_tarfile()` duplicated 4x with the safety guard only in one copy — already tracked in
  `modernization_plan.md`'s "below the line" section (recommends applying the guard to all four without
  consolidating, since the "independently runnable scripts" design is intentional per
  `condor_lxplus/README.md`).
- Three independently-maintained argparse definitions (`runner.py`/`suball.py` vs.
  `condor_lxplus/submitter.py`) — already tracked as `architecture_review.md` item 6 ("no urgent action,
  awareness only").

## Recommended order

1. **A1** (`-e` fix) — confirmed live bug, cheap, do first.
2. **A2** (`--outputDir` check) — cheap, closes a silent-data-loss gap.
3. **P1** (sample instead of full-dataset site probe) — one-line change, large win.
4. **S1-S4** — mechanical cleanup, any order, no urgency.
5. **P2** (packed env via EOS/xrootd) — decided direction, medium effort; build the packaging script and
   swap `condor/execute.sh`'s bootstrap once ready. Verify the packed env's correctness on a fresh worker
   before switching over.
6. The data-locality question for `condor_lxplus/` (see note above) still requires an explicit decision —
   whether to bring it in line with `condor/`'s architecture (largely absorbing item P2's approach) or
   keep it as the deliberately eos/afs-mounted lightweight variant. Not something to implement
   opportunistically.
