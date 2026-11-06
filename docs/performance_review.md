# Computational Performance Review — BTVNanoCommissioning

Scope: per-chunk hot path (`process()`/`process_shift()` and everything called from it) across
`utils/correction.py`, `utils/selection.py`, `utils/histogramming/histogrammer.py`,
`utils/array_writer.py`, `helpers/func.py`, `helpers/update_branch.py`, and two representative
workflows (`example.py`, `ctag_Wctt_valid_sf.py`). Formatting/readability ignored per request.

Structural fact that amplifies several findings below: `process()` (example.py:66-74,
identical pattern in every workflow) builds a `shifts` list of object-level systematic
variations in `common_shifts()` (correction.py:4059) — with `isSyst=True` this is on the order
of **30-40 shift variants** (JES sources, JER, MET unclustered, muon/electron scale-smear, …).
`process_shift()` — and therefore `weight_manager()`, `histo_writter()`/the hand-rolled
equivalent, and `array_writer()` — runs **once per shift, not once per chunk**. Any per-shift
cost is therefore multiplied ~30-40x per chunk when running with systematics on.

---

### 1. `weight_manager()` recomputes all SFs on every systematic shift, not just nominal

**Location:** `example.py:71-73` (and identical call pattern in every workflow) → `weight_manager()` at `correction.py:4174`.

**Problem:** `weight_manager()` is invoked once per entry in the `shifts` generator, i.e. once per JES/JER/MET/lepton-scale variation (~30-40x per chunk with `isSyst=True`). Internally it recomputes pileup weight, PDF/scale/PS weights, muon ID SF, electron ID SF, and (for MC) the full b-tag SF chain — none of which depend on the JES/JER/MET-unclustered shift being processed (only jet-kinematics-dependent pieces like the b-tag shape SF genuinely need recomputation per jet-shift; PDF/pileup/lepton-ID SFs are shift-invariant for JES/JER/MET shifts, and only need recomputation for shifts that actually change lepton/jet kinematics they depend on).

**Expected speedup:** Large — if ~25 of ~35 shifts are pure jet-energy variations that don't touch leptons/pileup/PDF, then skipping shift-invariant weight recomputation for those removes a large fraction of `weight_manager()`'s total per-chunk cost (weight computation is typically a significant fraction of total processing time in these workflows). Rough order: could cut total weight-computation work by 2-5x depending on shift mix.

**Confidence:** Medium — the amplification factor (30-40x call count) is confirmed by reading the code; the fraction of `weight_manager()` cost that is genuinely shift-invariant vs. shift-dependent was not independently profiled.

**Implementation effort:** High — requires threading through which weight components actually depend on which shift type, likely via caching shift-invariant weight arrays on the processor instance/first shift and reusing them, or restructuring `common_shifts`/`process_shift` to separate "object-level shift" from "event-weight recomputation." Also directly affects priority #1 (physics results) — needs careful validation that skipped recomputation doesn't silently change any SF for jet-kinematics-dependent components (e.g. b-tag shape SF, which *does* depend on shifted jet pt/eta and must still be recomputed per shift).

---

### 2. `btagSFs()` — nested Python loop over jet slots × systematics, one `correctionlib.evaluate()` per cell

**Location:** `correction.py:2455-2461` (loop opens `for i, sys in enumerate(systlist)` around `for nj in range(ak.num(alljet.pt)[0])`), with `.evaluate()` calls inside (e.g. `correction.py:2474`).

**Problem:** For each b-tag systematic (`systlist` has 8-14 entries depending on `SFtype`) and each jet-slot index up to the max number of jets in the chunk, a separate `correctionlib.evaluate()` call is made on a `pt`-length-batched array (vectorized over events, but not over jets/systematics). This is O(n_systs × max_jets_per_event) correctionlib calls per shift, each with per-call Python/C++ boundary overhead, rather than one call per systematic on the full flattened jet array.

**Expected speedup:** Moderate — correctionlib call overhead is not free even when each call is internally vectorized over events; collapsing the jet-slot loop into one flattened-array `.evaluate()` call per systematic (using `ak.flatten`/`ak.unflatten` once) would cut the call count by the average jet multiplicity (typically 3-6x fewer calls).

**Confidence:** Medium — correctionlib's shape-SF evaluate supports flattened jagged input directly in recent versions, but this wasn't verified against the pinned `correctionlib==2.7.0` API in this environment.

**Implementation effort:** Medium — restructure to flatten once before the systematic loop, evaluate on the flat array, `ak.unflatten` after. Localized to `btagSFs()`.

---

### 3. `add_fragmentation_decay_weights()` — pure-Python double loop over every genparticle, every event

**Location:** `correction.py:3760-3809` (outer `for ievt in range(n_evt)`, inner `for i, m in enumerate(mother)` building `children[m].append(i)`, plus a further per-event `_find_top_hadron_pair` genealogy search and scalar `_x_had` calls).

**Problem:** This is a genuine per-event, per-genparticle Python loop — no vectorization at all. For a chunk of e.g. 50k events × ~200 GenPart entries/event, this is ~10M Python-level operations just for the children-list construction, before the hadron-pair search. Gated behind `ttbar_reweights == "full"`, so it only fires when that option is enabled, but when active it runs once (not per-shift, since it operates on gen-level truth information independent of detector-level shifts) — still, at O(n_events × n_genparts) pure Python, this is the most severe raw-Python-loop finding in the hot path.

**Expected speedup:** Very large if vectorized (10-50x plausible for this specific function), but genealogy-graph traversal (finding mother→children chains and the specific top→hadron decay chain) is inherently harder to vectorize in awkward than flat kinematic ops — a full rewrite would need either `ak`-native graph traversal patterns or a compiled path (numba, which coffea/awkward already integrates with via `ak.numba` or plain `@numba.njit` on the extracted numpy arrays, since `pt_all`/`mother_all` etc. are already per-event numpy sublists at that point).

**Confidence:** High that this is a real hot Python loop when the code path is active; effort to fix depends on how easily the genealogy search maps to numba (each event is independent, so numba JIT + Python-level `for ievt` loop replaced by a numba-compiled function would likely be the pragmatic fix rather than full awkward vectorization).

**Implementation effort:** Medium-high — wrapping the existing per-event logic in a `@numba.njit` function (numba plays well with the existing scalar/list-based logic since it's already flat per-event lists) is more tractable than a full awkward-native rewrite, and would preserve exact numerical behavior (important given priority #1).

---

### 4. 2D working-point bin lookup — Python list comprehension over every jet in the chunk

**Location:** `helpers/update_branch.py:416-428`, specifically the list comprehension at lines 417-424 iterating `range(len(ihfvlf))` (i.e., over every jet across every event in the chunk, flattened).

**Problem:** For each jet, does a Python-level dict/2D-array lookup (`btag_wp_dict[...]["2D"]["mapping"][ihfvlf[i]][ibvc[i]]`) inside a list comprehension, then `ak.unflatten`s the result. This runs once per chunk (called from `missing_branch()`, itself called once per `process()` before the shift loop — not multiplied by shifts), but at O(total jets in chunk), which for large chunks (chunksize often 100k-500k events, several jets each) is a non-trivial pure-Python cost.

**Expected speedup:** Moderate — `btag_wp_dict[...]["2D"]["mapping"]` is a small fixed-size 2D table; if it's converted to a numpy 2D array, the whole lookup becomes `mapping_array[ihfvlf, ibvc]` (single vectorized numpy fancy-index), replacing the Python loop entirely. Likely 5-20x faster for this specific operation, though it's a small fraction of total per-chunk time since it only runs once per chunk (not per-shift).

**Confidence:** High that the vectorized numpy replacement is correct and straightforward, given `mapping` is described as a fixed lookup table.

**Implementation effort:** Low — convert `mapping` to a numpy 2D array once (at load time, not per-chunk) and replace the comprehension with fancy indexing plus a `where(hfvlf == -1, -1, ...)` for the sentinel case already present in the comprehension (line 420).

---

### 5. `fixPhiRange` applied via Python list comprehension, once per JES/JER/MET shift

**Location:** `correction.py:912`, `phi_miss_final = ak.Array([fixPhiRange(_phi) for _phi in phi_miss_final])`, inside `calc_T1_MET()` which is called from `JME_shifts()` once per systematic variant (~15-17x per chunk based on the JES/JER/UES source count).

**Problem:** `fixPhiRange()` (correction.py:853-858) is a scalar function doing two conditional shifts to wrap an angle into `[-π, π]`. Called via Python list comprehension over every event's MET phi, once per shift — O(n_events) pure Python per shift × ~15-17 shifts per chunk.

**Expected speedup:** High for this specific line — replacing with the standard vectorized wrap `np.mod(phi + np.pi, 2*np.pi) - np.pi` eliminates the Python loop entirely; for large chunks this specific line could go from O(n_events) Python overhead to a single numpy vectorized call, easily 10-50x faster for this operation, though it's one line among many in `calc_T1_MET`.

**Confidence:** High — the replacement is a standard, exact-equivalent vectorization of the two-branch wraparound logic (same numerical result for all finite inputs).

**Implementation effort:** Low — single-line change, easy to unit-check against the old scalar function on a few boundary values (`phi` near `±π`) before replacing.

---

### 6. `weights.partial_weight(exclude=...)` recomputed repeatedly inside the per-histogram loop

**Location:** `ctag_Wctt_valid_sf.py:498, 514, 574, 598` — `weights.partial_weight(exclude=exclude_btv)` called at 4+ separate call sites within the hand-rolled histogram-filling dispatch (part of the same function flagged in the architecture review for duplicating `histo_writter`).

**Problem:** `coffea.analysis_tools.Weights.partial_weight()` recomputes the product of all included per-event weight columns from scratch on every call (it does not cache the partial product for a given `exclude` set). Since `exclude_btv` is the same exclusion set across these call sites within one `process_shift()` invocation, the same O(n_events) weight-multiplication work is repeated 4+ times per shift.

**Expected speedup:** Moderate — caching the result of `weights.partial_weight(exclude=exclude_btv)` in a local variable once per shift and reusing it across the 4 call sites would cut this specific redundant work to 1/4 of its current cost; overall chunk-level impact depends on how large a fraction of total time this represents (likely small-to-moderate, since weight multiplication itself is cheap per call — the win is mostly from avoiding repeated full-array elementwise multiplies across ~10+ weight columns).

**Confidence:** High that the call sites are redundant (same `exclude` argument, same shift context); moderate on the magnitude of the win since `partial_weight` itself is not extremely expensive per call.

**Implementation effort:** Low — hoist one `partial_weight(exclude=exclude_btv)` call to a local variable before the histogram-filling block and reuse it at all 4 sites.

---

### 7. Redundant `ak.flatten`/`ak.num` recomputation in histogram dispatch (both `histo_writter` and the `ctag_Wctt_valid_sf.py` duplicate)

**Location:** `histogrammer.py:146-414` (the substring-dispatch `if/elif` chain) and the structurally parallel dispatch in `ctag_Wctt_valid_sf.py:486-599`.

**Problem:** Both dispatch chains process histograms one at a time via string-matching on histogram name, and several branches independently call `ak.flatten(...)`/`ak.num(...)` on what is often the same underlying jet/lepton collection (e.g. flattening `sjets` or `SelJet` fields repeatedly across multiple histogram branches within the same shift). Because the dispatch is structured as many independent `if`/`elif` blocks rather than a single vectorized pass, the same flatten/broadcast operation can recur several times per shift for different histograms drawing from the same collection.

**Expected speedup:** Low-moderate per call (each individual flatten is cheap relative to the SF/weight computation), but multiplied by ~30-40 shifts and by however many histograms redundantly re-flatten the same collection, this could add up to a non-trivial fraction of total histogram-filling time.

**Confidence:** Low-medium — this is a structural observation from reading the dispatch code; the actual redundancy count (how many histogram branches share the same underlying flatten) wasn't exhaustively tabulated.

**Implementation effort:** Medium — would require restructuring the dispatch to precompute shared flattened/broadcast arrays once per shift and pass them into the per-histogram fill calls, rather than recomputing inside each branch. Cleanest to do only if/when finding #1 in the architecture review (unifying the two duplicated dispatch implementations) is tackled, since fixing this in only one of the two copies leaves the duplication problem unresolved.

---

## Not flagged / lower priority

- `array_writer.py`'s `xs_dict`-style lookup rebuilding (mentioned by initial scan) fires once per chunk (nominal-shift-only path), not per-shift — low severity, not included as a ranked finding.
- `eleSFs()`/`muSFs()`/`jmar_sf()` were reported to have similar jet/lepton-slot Python loop patterns to `btagSFs()` (finding #2) but were not independently line-verified in this pass; likely the same class of fix applies (flatten-once-per-systematic instead of loop-over-slots), worth checking alongside #2 rather than as a separate effort.

---

## Summary table

| # | Location | Fires per chunk | Expected speedup | Confidence | Effort |
|---|----------|-----------------|-------------------|------------|--------|
| 1 | `weight_manager()` shift-invariant recompute | ~30-40x | Large | Medium | High |
| 2 | `btagSFs()` jet-slot × syst loop | ~8-14 systs × max_jets, per shift | Moderate | Medium | Medium |
| 3 | `add_fragmentation_decay_weights()` gen-loop | 1x (gated by option) | Very large (if active) | High | Medium-high |
| 4 | 2D WP bin lookup comprehension | 1x per chunk | Moderate | High | Low |
| 5 | `fixPhiRange` list comprehension | ~15-17x per chunk | High (for this line) | High | Low |
| 6 | `partial_weight()` repeated calls | 4x+ per shift | Moderate | High | Low |
| 7 | Redundant flatten in histogram dispatch | ~30-40x per chunk | Low-moderate | Low-medium | Medium |

Recommend starting with #4, #5, #6 (low effort, high confidence, no physics-behavior ambiguity), then #2, before attempting #1 or #3 which need careful validation against reference outputs given priority #1 (never change physics results silently).
