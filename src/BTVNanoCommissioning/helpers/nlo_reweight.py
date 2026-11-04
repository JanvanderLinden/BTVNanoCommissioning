"""NLO negative-weight reweighting for use *inside* an existing Coffea processor.

Call ``calculate_nlo_reweighting(events, ...)`` once per Coffea event chunk.

Dependencies: numpy, awkward, onnxruntime, and the project's compiled
``negativeReweight`` Python extension. The ONNX model and optional JSON sidecar
must be available on the Coffea worker.

Only the first nine (non-NPME) model heads are used: nominal + eight LHE scale
heads. The ONNX model may still internally evaluate its unused NPME heads.

All reweight factors below multiply the *original signed generator weight*:
    w_reweighted = w_gen * result["nloReweight"]
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import awkward as ak
import numpy as np
import onnxruntime as ort

# Canonical LHE scale slots (the central index 4 is excluded).
SCALE_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
SCALE_NAMES = tuple(f"scale_{i}" for i in SCALE_INDICES)
NOMINAL_HEAD_NAMES = ("nominal",) + SCALE_NAMES

# Matches the column order used by the original nlo_reweight_vjets() helper.
SYSTEMATIC_NAMES = (
    tuple(f"nlo_vjet_{i}" for i in range(10))
    + ("nlo_vjet_muF", "nlo_vjet_muR", "nlo_vjet_muFmuR", "nlo_vjet_nominalWeight")
)

PROCESS_IDS = {"znunu": 0, "wlnu": 1, "dyll": 2}
ERA_IDS = {"ul": 0, "run3_22_23": 1, "run3_24": 2}
BEAM_ENERGIES = {"ul": 6500.0, "run3_22_23": 6800.0, "run3_24": 6800.0}


@lru_cache(maxsize=4)
def _get_runtime(model_path: str):
    """Load the ONNX session once per worker process, rather than per chunk."""
    metadata_path = Path(model_path + ".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    features_input = next(i for i in session.get_inputs() if i.name == "pf_features")
    fixed_batch_size = features_input.shape[0]
    batch_size = (
        fixed_batch_size if isinstance(fixed_batch_size, int)
        else int(metadata.get("batch_size", 64))
    )
    particle_length = metadata.get("particle_length", features_input.shape[-1])
    if not isinstance(particle_length, int) or particle_length <= 0:
        raise ValueError("ONNX particle length must be fixed or given in the .onnx.json sidecar")
    if batch_size <= 0:
        raise ValueError("Invalid ONNX batch size")

    output_shape = session.get_outputs()[0].shape
    if isinstance(output_shape[-1], int) and output_shape[-1] != 18:
        raise ValueError("Expected an 18-head ONNX model: 9 nominal + 9 NPME")
    output_names = tuple(metadata.get("output_names", ()))
    if output_names and output_names[:9] != NOMINAL_HEAD_NAMES:
        raise ValueError(f"Unexpected non-NPME head ordering: {output_names[:9]}")

    calibration_source = metadata.get("calibration_jet_bin_source", "ps_jet_count")
    if calibration_source not in ("ps_jet_count", "np_me"):
        raise ValueError(f"Unsupported calibration jet bin source: {calibration_source}")
    return session, int(batch_size), int(particle_length), calibration_source


def _lhe_scale_ratios(events):
    """(N, 8) absolute LHE scale ratios, in canonical scale-index order.

    Handles both eight-entry weights (missing the central slot 4) and
    nine-entry weights (containing slot 4). Missing LHE weights use unity.
    """
    n_events = len(events)
    canonical = np.ones((n_events, 9), dtype=np.float32)
    if n_events == 0 or "LHEScaleWeight" not in events.fields:
        return canonical[:, SCALE_INDICES]

    lhe = ak.fill_none(events.LHEScaleWeight, 1.0, axis=-1)
    counts = ak.to_numpy(ak.fill_none(ak.num(lhe, axis=1), 0))
    unsupported = np.unique(counts[~np.isin(counts, (0, 8, 9))])
    if unsupported.size:
        raise ValueError(f"LHEScaleWeight must have 0, 8, or 9 values; got {unsupported.tolist()}")

    eight = counts == 8
    if eight.any():
        values = ak.to_numpy(lhe[eight]).astype(np.float32)
        canonical[eight, :4] = values[:, :4]
        canonical[eight, 5:] = values[:, 4:]
    nine = counts == 9
    if nine.any():
        canonical[nine] = ak.to_numpy(lhe[nine]).astype(np.float32)
    return np.abs(canonical[:, SCALE_INDICES])


def _extract_features(events, particle_length, process, era, network_libdir):
    """Call the same GenPart -> model-input extractor as the source module."""
    if network_libdir is not None:
        libdir = str(Path(network_libdir).expanduser().resolve())
        if libdir not in sys.path:
            sys.path.insert(0, libdir)
    import negativeReweight

    genpart, generator = events.GenPart, events.Generator

    def flatten(name, dtype):
        return ak.to_numpy(ak.flatten(getattr(genpart, name), axis=1)).astype(dtype, copy=False)

    def gen(name, dtype):
        return ak.to_numpy(getattr(generator, name)).astype(dtype, copy=False)

    extracted = negativeReweight.extract_particles(
        flatten("pt", np.float32),
        flatten("eta", np.float32),
        flatten("phi", np.float32),
        flatten("mass", np.float32),
        flatten("status", np.int32),
        flatten("statusFlags", np.int32),
        flatten("pdgId", np.int32),
        flatten("genPartIdxMother", np.int32),
        ak.to_numpy(ak.num(genpart.pt, axis=1)).astype(np.int32),
        gen("x1", np.float32), gen("x2", np.float32),
        gen("id1", np.int32), gen("id2", np.int32),
        float(BEAM_ENERGIES[era]), int(particle_length),
        int(PROCESS_IDS[process]), int(ERA_IDS[era]),
    )

    names = (
        "pf_features", "pf_vectors", "pf_mask", "event_features",
        "ps_encoding_features", "calibration_features", "ps_dr",
        "ps_jet_count", "np_me",
    )
    missing = set(names).difference(extracted)
    if missing:
        raise KeyError(f"negativeReweight.extract_particles missing: {sorted(missing)}")
    features = {key: np.asarray(extracted[key], dtype=np.float32) for key in names}

    # The 3rd particle token contains the post-shower vector boson in this
    # extractor. Calculate V pT *before* dropping padded particles.
    vectors = features["pf_vectors"]  # (N, 4, L) = px, py, pz, E
    ps_vpt = np.hypot(vectors[:, 0, 2], vectors[:, 1, 2]).astype(np.float32)

    # Normalize (N,L), (N,1,L), or (N,L,1) to (N,1,L), as in the
    # original implementation. This shape broadcasts across feature channels.
    mask = features["pf_mask"].astype(bool)
    if mask.ndim == 3:
        singleton_axes = tuple(
            axis for axis in range(1, mask.ndim) if mask.shape[axis] == 1
        )
        if singleton_axes:
            mask = np.squeeze(mask, axis=singleton_axes)
    if mask.ndim != 2 or mask.shape != (len(events), vectors.shape[2]):
        raise ValueError(
            f"Cannot align pf_mask {features['pf_mask'].shape} "
            f"with pf_vectors {vectors.shape}"
        )
    mask = mask[:, None, :]
    for name in ("pf_features", "pf_vectors"):
        if features[name].ndim != 3 or features[name].shape[2] != mask.shape[2]:
            raise ValueError(f"Cannot apply pf_mask {mask.shape} to {name} {features[name].shape}")
        features[name] = np.where(mask, features[name], 0.0).astype(np.float32)

    return features, ps_vpt


def _pad(a, batch_size):
    """Pad a short final batch for fixed-batch ONNX models."""
    if len(a) == batch_size:
        return np.asarray(a, dtype=np.float32)
    if len(a) > batch_size:
        raise ValueError("Batch larger than configured ONNX batch size")
    return np.pad(a, ((0, batch_size - len(a)),) + ((0, 0),) * (a.ndim - 1)).astype(
        np.float32, copy=False
    )


def _build_results(probabilities, gen_sign, abs_scale_ratios, ps_vpt, ps_jet_count):
    """Build exactly the supplied nominal/bootstrap/composite-scale convention."""
    if probabilities.ndim != 3 or probabilities.shape[1:] != (20, 9):
        raise ValueError(f"Expected ONNX predictions (N, 20, 9); got {probabilities.shape}")

    predictions = (2.0 * probabilities - 1.0).astype(np.float32)
    bootstrap = predictions[:, :, 0]                          # (N,20)
    nominal = gen_sign * bootstrap.mean(axis=1)               # (N,)
    uncertainty = bootstrap.std(axis=1)                       # (N,)

    # These are multipliers relative to the original signed generator weight.
    # Each individual scale head uses |LHEScaleWeight|, never NPME weights.
    scale = (
        gen_sign[:, None] * abs_scale_ratios * predictions[:, :, 1:9].mean(axis=1)
    ).astype(np.float32)                                      # (N,8)
    pos = {scale_index: col for col, scale_index in enumerate(SCALE_INDICES)}
    scale_up = scale[:, [pos[5], pos[7], pos[8]]]    # muF, muR, both up
    scale_down = scale[:, [pos[3], pos[1], pos[0]]]  # muF, muR, both down

    # Columns 0:10 = independent bootstrap heads paired 0:9 with 10:19;
    # 10:13 = 3 composite scales; 13 = unity to restore the original weight.
    ones = np.ones((len(gen_sign), 1), dtype=np.float32)
    up = np.concatenate([gen_sign[:, None] * bootstrap[:, :10], scale_up, ones], axis=1)
    down = np.concatenate([gen_sign[:, None] * bootstrap[:, 10:20], scale_down, ones], axis=1)
    return {
        "nloScoreMean": nominal,              # alias from the original interface
        "nloScoreUncert": uncertainty,
        "nloReweight": nominal,
        "nloReweightUp": up,
        "nloReweightDown": down,
        "allPreds": predictions,              # (N,20,9), no NPME heads
        "bootstrapScores": bootstrap,         # (N,20), unsigned (2p-1)
        "scaleWeights": scale,                # (N,8), ordered by SCALE_NAMES
        "ps_vpt": ps_vpt,
        "ps_jet_count": ps_jet_count,
    }


def calculate_nlo_reweighting(
    events,
    *,
    model_path,
    process="dyll",
    era="run3_22_23",
    network_libdir=None,
):
    """Compute NLO reweighting on one Coffea/NanoEvents chunk.

    Args:
        events: NanoEvents chunk containing GenPart, Generator, LHEScaleWeight.
        model_path: Path to the ensemble ONNX model (18 output heads).
        process: 'znunu', 'wlnu', or 'dyll'.
        era: 'ul', 'run3_22_23', or 'run3_24'.
        network_libdir: Optional directory containing compiled ``negativeReweight``.
            Omit if importable through PYTHONPATH.

    Returns:
        A dictionary of per-event NumPy arrays. The important fields are:
          nloReweight      (N,)      nominal multiplier
          nloReweightUp    (N, 14)  10 bootstrap + 3 scale + unity
          nloReweightDown  (N, 14)  same ordering
          scaleWeights     (N, 8)   individual non-NPME scale multipliers
          allPreds         (N,20,9) signed 2*p-1 predictions, non-NPME only
        It also returns nloScoreMean, nloScoreUncert, bootstrapScores, ps_vpt,
        and ps_jet_count. See SYSTEMATIC_NAMES and SCALE_NAMES for column names.

    Returned factors already include sign(genWeight), They should multiply the 
    original signed ``events.Generator.weight`` exactly once.
    """
    if process not in PROCESS_IDS:
        raise ValueError(f"process must be one of {tuple(PROCESS_IDS)}")
    if era not in ERA_IDS:
        raise ValueError(f"era must be one of {tuple(ERA_IDS)}")

    n = len(events)
    if n == 0:
        # Avoid requiring model initialization for empty chunks.
        return _build_results(
            np.empty((0, 20, 9), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty((0, 8), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    model_path = str(Path(model_path).expanduser().resolve())
    session, batch_size, particle_length, calibration_source = _get_runtime(model_path)
    features, ps_vpt = _extract_features(events, particle_length, process, era, network_libdir)
    abs_scale_ratios = _lhe_scale_ratios(events)

    gen_weight = ak.to_numpy(events.Generator.weight).astype(np.float32)
    gen_sign = np.where(gen_weight != 0, np.sign(gen_weight), 1.0).astype(np.float32)

    # Preserve the 18-column absolute-weight *input* expected by the existing
    # ONNX graph. The second nine inputs duplicate the first nine, with no 1.52
    # factor. The corresponding NPME output columns are entirely discarded.
    first_nine = np.column_stack([np.ones(n, dtype=np.float32), abs_scale_ratios])
    abs_weights = np.concatenate([first_nine, first_nine], axis=1).astype(np.float32)

    model_inputs = {
        name: features[name] for name in (
            "pf_features", "pf_vectors", "pf_mask", "event_features",
            "ps_encoding_features", "calibration_features",
        )
    }
    model_inputs.update({
        "jet_bin": features[calibration_source].reshape(-1, 1),
        "ps_dr": features["ps_dr"].reshape(-1, 1),
        "process_id": np.full((n, 1), PROCESS_IDS[process], dtype=np.float32),
        "pythia_era_id": np.full((n, 1), ERA_IDS[era], dtype=np.float32),
        "abs_weights": abs_weights,
    })

    predicted_batches = []
    output_name = session.get_outputs()[0].name
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch_inputs = {name: _pad(value[start:stop], batch_size)
                        for name, value in model_inputs.items()}
        output = np.asarray(session.run([output_name], batch_inputs)[0], dtype=np.float32)
        if output.ndim == 2:
            output = output[:, None, :]
        if output.ndim != 3 or output.shape[1] != 20 or output.shape[2] != 18:
            raise ValueError(f"Expected ONNX probabilities (batch,20,18), got {output.shape}")
        predicted_batches.append(output[:stop - start, :, :9])  # only non-NPME heads

    probs = np.concatenate(predicted_batches, axis=0)
    return _build_results(
        probs, gen_sign, abs_scale_ratios, ps_vpt,
        features["ps_jet_count"].astype(np.int32),
    )
