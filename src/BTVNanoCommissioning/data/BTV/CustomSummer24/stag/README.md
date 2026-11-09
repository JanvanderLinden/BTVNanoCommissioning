# External inputs for the `stag_*_tt1L_s` workflows (CustomSummer24, CustomSummer25)

Private NanoAODv15 tuples, s-tagging WPs of NoPIDParTAK4 (class probabilities
`btagNoPIDParTAK4prob{b,bb,c,cc,g,s,ud}`).

| file | `STAG_CONFIG` key | env override | content |
| --- | --- | --- | --- |
| `stag-wp-NoPIDParTAK4.json` | `tagging_wps` | `STAG_TAGGING_WPS` | `wps.json` of sfs-3d `B_derive_wps.sh` with `wp/configs/stag_2024_UParTNoPID.yml`: correction `tagging_bin` (inputs `pt, bvall, cvnotb, svudg`; output: s-tagging bin 0-7 = BT, BL, CT, CL, L, SL, SM, ST) |
| `stag-wp-NoPIDParTAK4_scores.json` | `tagging_scores` | `STAG_TAGGING_SCORES` | `wps_scores.json` of the same derivation: the scores `bvall`, `cvnotb`, `svudg` from the class probabilities, incl. score reweightings |
| `../../Summer24/stag/spanet_model_fold{0,1}.onnx` | `spanet_model` | `STAG_SPANET_MODEL` | SPANet fold models (tagger independent) |

The SPANet training inputs (`stag_{mu,e}_tt1L_s_spanet_data`) need none of these files.
