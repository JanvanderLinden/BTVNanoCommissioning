# Simplification Review — BTVNanoCommissioning

Scope: unnecessary complexity only — dead code, duplicate code, excessive wrappers, unnecessary
classes/config, complicated control flow. Not architecture (see `docs/architecture_review.md`)
or performance/I/O (see `docs/performance_review.md`, `docs/io_review.md`). All findings below
were confirmed by grepping usage repo-wide (including `SFbc-2D/`, `condor*/`) before being
reported as unused/redundant.

---

### 1. `campaign` parameter is accepted but never read in 4 selection functions

**Current complexity:** `softmu_mask(events, campaign, dxySigCut=0)` (`selection.py:227`), `mu_idiso(events, campaign)` (`selection.py:239`), `mu_promptmvaid(events, campaign)` (`selection.py:248`), and `btag_mu_idiso(events, campaign)` (`selection.py:261`) all take `campaign` as a required parameter, and every call site across all workflows dutifully passes `self._campaign` — but none of the four function bodies reference `campaign` at all.

**Simplification:** Drop the unused `campaign` parameter from these 4 function signatures and their call sites. Since these are plain functions (not part of a shared interface requiring uniform signatures — contrast with `weight_manager`/`common_shifts`), there's no polymorphism reason to keep the parameter for signature consistency.

**Risk:** Very low — purely removes a dead parameter from 4 functions and their ~15+ call sites; no behavior change. Mechanical, easy to verify via `grep`.

---

### 2. `ele_promptmvaid`'s internal campaign branch is unreachable and would break if it ever fired

**Current complexity:** `ele_promptmvaid(events, campaign)` (`selection.py:210-224`) contains its own campaign ternary at lines 220-222 that returns a bare Python scalar `0.3` (not a boolean array) for campaigns outside `["Summer24", "CustomSummer24", "Winter25", "Prompt25"]`, combined via `&` with a boolean array — this would be a runtime type error if it ever executed (`bool_array & 0.3` is not valid). It never executes today only because all 4 call sites (`_2D_Wc_valid_sf.py:168`, `_2D_ttsemilep_valid_sf.py:~138`, `ctag_DY_valid_sf.py:~176`, `ttdilep_valid_sf.py:~149`) independently re-implement the identical `if self._campaign in ["Summer24", "CustomSummer24", "Winter25", "Prompt25"]:` guard before ever calling this function — so the function's own branch is permanently dead, and the guard logic is duplicated 4x at the call sites instead of living once inside the function.

**Simplification:** Since every caller already guards for the NanoAODv15-only campaigns before calling, the internal ternary at lines 220-222 can be deleted and the function simplified to only implement the `promptMVA >= 0.9` branch (matching what actually ever executes) — collapsing dead/unreachable code and removing the false impression that this function is safe to call for older campaigns. Alternatively, if callers should be simplified to stop duplicating the guard, that's a larger call-site change; the minimal, safe fix is just deleting the unreachable branch in the function itself.

**Risk:** Low — deleting an unreachable branch cannot change any executed behavior since it's confirmed dead by every call site's guard. Verify no future call site is added without a guard before removing (a comment noting the precondition may be warranted, though per style guidance keep it minimal).

---

### 3. `ele_cuttightid`/`ele_mvatightid`/`ele_promptmvaid` share an identical etaSC ternary block

**Current complexity:** `selection.py:186-224` — all three functions independently repeat the same 5-line `ele_etaSC = ... if campaign not in [...] else ...` block and the same `(abs(ele_etaSC) < 1.4442) | (...)` eta-crack mask.

**Simplification:** Factor the shared `ele_etaSC` computation and eta-crack mask into one small helper (e.g. `_ele_etaSC(events, campaign)` and `_ele_eta_crack_mask(ele_etaSC)`), called by all three. This is a small, mechanical dedup — not a new abstraction layer, just removing 3x copy-paste of the same 5 lines.

**Risk:** Low — purely extracting identical existing logic into one place; straightforward to verify the extracted helper produces bit-identical masks before/after.

---

### 4. `make_tarfile()` duplicated 4x with a safety fix present in only one copy

**Current complexity:** `make_tarfile()` is defined nearly identically in `runner.py:155`, `condor/submitter.py:19`, `condor_lxplus/submitter.py:9`, and `scripts/maketar.py:5`. Only the `condor/submitter.py:19-38` copy has a guard (lines 27-37) preventing the output tarball from nesting inside itself or including a stale prior tarball; the other 3 copies lack this guard entirely.

**Simplification:** This is a duplication *and* a latent-bug-divergence finding: the 3 unguarded copies can silently produce a self-nesting or bloated tarball if run in a directory containing a stale `.tar.gz`. Consolidating to one shared implementation (e.g. in `helpers/` or a small shared submission-utils module) with the safety guard applied everywhere would both remove duplication and fix the 3 divergent copies at once.

**Risk:** Medium — touches 4 separate submission entry points (`runner.py`'s condor-standalone path, `condor/`, `condor_lxplus/`, and the standalone `scripts/maketar.py`). Consolidating requires care not to disturb each script's independent invocability (condor_lxplus/README.md documents these as separately runnable). Given this repo's "avoid large rewrites" rule, the safer minimal fix is applying the missing guard to the 3 unguarded copies without necessarily consolidating into a shared module — flagging the choice rather than prescribing one.

---

### 5. `get_condor_submitter_parser` defined in `runner.py` but never called there — dead weight

**Current complexity:** `runner.py:168-...` defines `get_condor_submitter_parser(parser)`, adding `--jobName`/`--condorFileSize`/etc. argparse arguments, but `runner.py` never calls this function anywhere else in the file (confirmed via grep — the only match for the call-pattern `get_condor_submitter_parser(` in `runner.py` is the `def` line itself). The same function is separately (and actually used) in `condor/submitter.py` and `condor_lxplus/submitter.py`.

**Simplification:** Delete the dead copy from `runner.py`; it does nothing there and appears to be a leftover from when condor-submitter argument parsing may have lived in `runner.py` before being split into the dedicated `condor/`/`condor_lxplus/` submitter scripts.

**Risk:** Very low — confirmed zero call sites within `runner.py`; deleting removes only unreachable definition code.

---

### 6. `jmar_sf()` — dead function, and the `AK4_parameters.py` config path that would feed it is also dead

**Current complexity:** `correction.py:3155` defines `jmar_sf(jet, correct_map, weights, syst=False)`, which reads `correct_map["JMAR_cfg"]`/`correct_map["JMAR"]` (populated by `load_SF()` at `correction.py:457-463` when a campaign config supplies a `"JMAR"` key). Grepping `AK4_parameters.py` for `"JMAR"` finds zero matches — no campaign config ever sets a `JMAR` key, so the `load_SF()` branch at line 457 never populates `correct_map["JMAR"]`, and `jmar_sf()` itself is never called from any workflow (confirmed via repo-wide grep — only the `def` site matches).

**Simplification:** Both `jmar_sf()` (`correction.py:3155-...`) and the corresponding `load_SF()` `"JMAR"` branch (`correction.py:457-463`) are currently unreachable/unused. Given this repo's rule to flag rather than silently delete pre-existing dead code the user didn't ask about, this is reported for awareness — deleting is straightforward if confirmed genuinely obsolete (e.g. not a placeholder for planned near-term JMAR SF support).

**Risk:** Low to delete if confirmed obsolete; note it before removing since JMAR (JetMET Analysis Recommendations) SFs may be an intentionally-staged, not-yet-wired feature rather than true dead code — worth a quick check with whoever added it before deleting.

---

### 7. `memory_usage_psutil()` and `num()` in `helpers/func.py` — zero references anywhere

**Current complexity:** `helpers/func.py:49` (`memory_usage_psutil()`) and `helpers/func.py:91` (`num()`) have zero call sites anywhere in the repo (confirmed via repo-wide grep, including `SFbc-2D/` and `condor*/`).

**Simplification:** Delete both — straightforward dead-code removal with no callers to update.

**Risk:** Very low.

---

## Not flagged as findings (checked, confirmed clean)

- `JPCalibHandler` (`correction.py:3920`) is a legitimate class — real shared state (loaded ROOT template, precomputed cumulative arrays) reused across its methods, not a namespace-only wrapper.
- All files in `workflows/` are registered in `workflows/__init__.py` — no orphaned/dead workflow processor files.
- `MuonScaRe.py` and `BTA_helper.py` functions (`cumsum`, `calc_ip_vector`, `is_from_GSP`, `to_bitwise_trigger`, `get_hadron_mass`) are all live, confirmed via grep.
- `weight_manager`'s hdamp-ML and fragmentation-decay hooks appear orphaned on a naive single-file grep but are called from within `correction.py` itself (`weight_manager` → `add_hdamp_ml_weight`/`add_fragmentation_decay_weights`) — not dead code.
- `scripts/dohadd.py` is referenced in `CLAUDE.md` but not invoked by any other script/README — plausibly an intentional standalone manual-run utility (same pattern as `scripts/dump_prescale.py`); not reported as a finding since confidence is too low to call it dead versus intentionally standalone.

---

## Summary table

| # | Issue | Location | Risk to fix |
|---|-------|----------|--------------|
| 1 | Unused `campaign` param, 4 functions | `selection.py:227,239,248,261` | Very low |
| 2 | Unreachable/broken branch in `ele_promptmvaid` | `selection.py:220-222` | Low |
| 3 | Duplicated etaSC block, 3 functions | `selection.py:186-224` | Low |
| 4 | `make_tarfile()` duplicated 4x, safety guard in only 1 | `runner.py:155`, `condor/submitter.py:19`, `condor_lxplus/submitter.py:9`, `scripts/maketar.py:5` | Medium |
| 5 | Dead `get_condor_submitter_parser` copy | `runner.py:168` | Very low |
| 6 | Dead `jmar_sf()` + dead JMAR config branch | `correction.py:3155`, `correction.py:457-463` | Low (verify not staged/planned) |
| 7 | Dead `memory_usage_psutil()`/`num()` | `helpers/func.py:49,91` | Very low |

Recommend doing #1, #5, #7 first (mechanical, zero-risk deletions), then #3 and #2 together (same file, related functions), and treating #4 and #6 as needing a quick check-in with the workflow owner before touching (submission-script safety behavior and possibly-staged JMAR support, respectively).
