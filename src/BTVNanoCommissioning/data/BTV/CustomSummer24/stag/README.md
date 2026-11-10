# External inputs for the s-tagging workflows (CustomSummer24, CustomSummer25)

Workflows `stag_{mu,e}_tt1L_s`, `stag_{mu,e}_Wc_s`, `stag_{mu,e}_DY_s`, `stag_emu_ttdilep_s`.
Private NanoAODv15 tuples, s-tagging WPs of NoPIDParTAK4 (class probabilities
`btagNoPIDParTAK4prob{b,bb,c,cc,g,s,ud}`).

| file | `STAG_CONFIG` key | env override | content |
| --- | --- | --- | --- |
| `wp_v1_ptincl_rew_svudg.json` | `tagging_wps` | `STAG_TAGGING_WPS` | `wps.json` of sfs-3d `B_derive_wps.sh` with `wp/configs/stag_2024_UParTNoPID.yml` (v1: pt-inclusive WPs, svudg reweighted ud/g with x = 1): corrections `tagging_bin`(pt, raw tagger outputs) -> s-tagging bin 0-7 = BT, BL, CT, CL, L, SL, SM, ST, and `score_{bvall,cvnotb,svudg}`(raw tagger outputs); scores and score reweightings are computed inside the corrections |
| `../../Summer24/stag/spanet_model_fold{0,1}.onnx` | `spanet_model` | `STAG_SPANET_MODEL` | SPANet fold models (tt1L-s only, tagger independent) |

The SPANet training inputs (`stag_{mu,e}_tt1L_s_spanet_data`) need none of these files.
