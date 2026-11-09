# External inputs for the `stag_*_tt1L_s` workflows (Summer24, CustomSummer24UParTPaper, Prompt25)

| file | `STAG_CONFIG` key | env override | content |
| --- | --- | --- | --- |
| `stag-wp.json` | `tagging_wps` | `STAG_TAGGING_WPS` | correctionlib json, correction `tagging_bin` (inputs `pt, bvall, cvnotb, svudg`; output: s-tagging bin 0-7 = BT, BL, CT, CL, L, SL, SM, ST) |
| - | `tagging_scores` | `STAG_TAGGING_SCORES` | definition of the `tagging_bin` score inputs (sfs-3d `wps_scores.json`); not set: the UParT discriminants `btagUParTAK4B`, `btagUParTAK4CvNotB`, `btagUParTAK4SvUDG` |
| `spanet_model_fold{0,1}.onnx` | `spanet_model` | `STAG_SPANET_MODEL` (comma-separated) | SPANet jet-parton assignment models for single-lepton ttbar, one per fold: model k evaluates the events with `event % 2 == k` and was trained on the other fold (sfs-3d `spanet/`; inputs `Momenta`, `Lepton`, `Met`; event-particles `lt`, `htb`, `W`) |
| `spanet_model.onnx` | - | `STAG_SPANET_MODEL` | single SPANet model of the columnflow setup (all events) |

Relative paths are resolved from the directory `runner.py` is executed in (repository root).
For condor/dask scale-out the files are needed on the worker nodes.
