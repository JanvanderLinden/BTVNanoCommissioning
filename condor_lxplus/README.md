# Light Condor submission script for lxplus

Submit from the checkout (AFS) with the conda environment active in which `pip install -e .`
was run. The jobs do not use AFS:

- The environment and the code are shipped as tarballs from EOS (`--sandboxDir`, default
  `$BTV_CONDOR_SANDBOX`, else `/eos/user/<u>/<user>/btv_condor_sandbox`), with the EOS xrootd
  transfer plugin:
  - `btv_env_<hash>.tar.gz`: the active conda environment packed with conda-pack (one-time
    `pip install conda-pack` in the environment); built once per environment version (the
    hash is over `conda list --explicit` and `pip freeze`), a few minutes and 1-2 GB;
  - `btv_code_<hash>.tar.gz`: `runner.py` and `src/` (code, correction and model files);
    rebuilt when any of these files changes.
  The jobs unpack both in their scratch directory, take the code from there (the editable
  install of the environment is dropped) and use the scratch directory as `$HOME`.
- The split sample json goes to `<outputDir>/.condor_sandbox/split_samples.json`; only
  `arguments.json` (and the executable and the proxy) come from the job directory.
- `max_materialize = --maxRunning` (default 200) limits the jobs of a submission in the
  queue at a time; the others are added as these finish. `request_disk = --disk` (10GB),
  `request_memory = --memory` (MB; not set by default: the lxplus default, 2000 MB per CPU).
- `--noSandbox` runs the jobs from the checkout and environment directly (the old way):
  only for a few jobs. Hundreds of jobs reading a conda environment on AFS overload the AFS
  server (CERN batch storage guidelines, KB0003076).
- Ships a copy of the proxy with each job (`x509userproxy`).

## Example
```
voms-proxy-init --voms cms --valid 192:00
conda activate btv_coffea   # the environment with which 'pip install -e .' was run
python condor_lxplus/submitter.py --workflow ctag_DY_sf --json fetched_list.json --campaign Summer22 --year 2022 --isArray --isSyst all --skipbadfiles --jobName condor_1 --outputDir /preferably/eos/output/directory --submit
```

## Check outputs
Check if all jobs succeeded with:
```
python condor_lxplus/checkoutputs.py <job_condor_dir>
```
Then manually resubmit failed jobs with the newly created config.

## Merge the outputs of a submission
```
python condor_lxplus/mergeoutputs.py <outputDir> -j 8                   # merge and check
python condor_lxplus/mergeoutputs.py <outputDir> -j 8 --delete-inputs   # + remove the per-job files
```
merges the per-job outputs of one submission:
- `hists_<n>/hists_<n>.coffea` -> `<outputDir>/hists_<name>.coffea` (`<name>`: base name of
  `<outputDir>`, `--name`), the input of `scripts/plotdataMC.py`; with `--split-hists` one
  file per histogram instead, `<outputDir>/hists/<histogram>.coffea` (`{dataset: {histogram,
  sumw, sumw_<variation>}}`), `_other.coffea` (all other entries) and `index.json`
  (histograms with axes and datasets): merged in chunks of `--hist-files-per-chunk` files
  (default 20), each written per histogram to the scratch directory, then each histogram on
  its own, so that the memory is that of one chunk or of the largest merged histogram;
- `arrays_*/<syst>/<dataset>/*.root` -> `<outputDir>/hadd/<syst>/<dataset>.root` (hadd with
  `-f102`, see `haddFinal.py`), the input of `checkHadd.py` and `normalizeArrays.py`.

The files are listed with `eos find` and read as `root://` URLs; the merged files are written
to a local scratch directory (`--tmpdir`, default `$TMPDIR`) and copied to EOS once. Checks:
the merged histograms contain every histogram of every input file, and every merged ROOT file
has one `TotalEventCount` entry per input file. A merge that already exists is skipped if it
is complete and reported otherwise (jobs that finished later: rerun with `-f`).
`--delete-inputs` removes `hists_<n>/` and `arrays_*/` (`eos rm -r`, after a confirmation,
`-y` to skip it) only when all checks passed; `merge_manifest_<time>.json` lists the inputs of
each merged file. hadd comes from the environment, else from `--root-setup` (an LCG view
`setup.sh`, sourced only for hadd). `--only hists|arrays` merges one of the two, `--groups
<pattern>` only the matching array groups `<syst>/<dataset>`.

Speed: the coffea files are added in place and read by `--hist-workers` processes (memory:
one partial sum, `--split-hists`: one chunk or histogram, per process); `-j` hadd processes run in parallel, each with
`--hadd-threads` (hadd `-j`).

Cascade merge on condor (DAGMan), for one or several submissions:
```
python condor_lxplus/submit_merge.py --jobDir <jobdir>/merge_<tag> --tasks tasks.txt --submit
```
with one line `<outputDir> <name> <hists|arrays|all>` per submission (or `--task` on the
command line). Stage-1 jobs merge `--hist-files-per-job` coffea files (default 20) or up to
`--root-files-per-job` ROOT files (default 50, at most `--root-items-per-job` outputs per job)
into `<outputDir>/.merge_tmp/` (or directly into the final file if all inputs fit into one
job); stage-2 jobs, started when the stage-1 jobs of that output directory are done, merge
these into `hists_<name>.coffea` and `hadd/<syst>/<dataset>.root`; a final job per output
directory checks the merged files (TotalEventCount entries = number of per-job files),
writes the merge manifest and removes `.merge_tmp/`. `--split-hists`: the stage-1 jobs write
their chunk as one file per histogram, `--key-groups` stage-2 jobs per output directory
(default 16) merge the histograms of their group one by one into `hists/`, the final job
checks them against the histogram lists of the stage-1 jobs and writes `hists/index.json`;
no job holds more than one chunk or one merged histogram (`--memory`: for the largest).
`--resplit` (task `<outputDir> <name> hists`): no merge, the existing `hists_<name>.coffea` is
split into `hists/` by one job (memory: the whole file), for merges whose per-job files are
deleted. The final job of each output directory requests 1 core and 2000 MB. Each node runs `merge_job.sh`
(mergeoutputs.py `--items` / `--finalize`) with the conda-pack tarball of the active
environment from the sandbox directory (as `submitter.py`), reads from EOS with root:// and
writes with xrdcp; failed nodes are retried twice, completed outputs are skipped on a retry.
Options: `--cpus` (default 4), `--memory` (MB, default 2000 x cpus), `--disk` (default 20GB),
`--jobFlavour`, `--hadd-threads`, `--maxJobs` (DAGMan, default 500), `-f` (redo existing
merges). Progress: `condor_q -batch <jobName>`, `<jobDir>/merge.dag.dagman.out`; after a
failure `condor_submit_dag merge.dag` in the job directory resubmits the failed nodes (rescue
DAG). The jobs do not delete per-job files: afterwards run `mergeoutputs.py` interactively
(complete merges are skipped) and with `--delete-inputs`.

## Plots on condor
Per-histogram merges (`--split-hists`): the plottable histograms from `index.json`, the jobs
read `<variable>.coffea` of every `--indir` that has it:
```
python scripts/split_coffea.py --from-index <outputDir1>/hists,<outputDir2>/hists --list vars.txt
python condor_lxplus/submit_plots.py --phase <workflow> --indir <outputDir1>/hists,<outputDir2>/hists \
    --vars vars.txt --outdir <eosdir>/plots --jobDir <jobdir>/plots --submit -- <plotdataMC.py options>
```
Single merged files: split them once into one file per histogram first:
```
python scripts/split_coffea.py "<outputDir>/*/hists_*.coffea" -o <eosdir>/plot_split --list vars.txt
python condor_lxplus/submit_plots.py --phase <workflow> --indir <eosdir>/plot_split --vars vars.txt \
    --outdir <eos plot dir> --jobDir <jobdir>/plots_<name> --submit -- --lumi <pb> --syst-band
```
`split_coffea.py` reads the coffea files once and writes one file per plottable histogram
(with `sumw`, `sumw_<variation>`), `submit_plots.py` submits one `plotdataMC.py` job per
histogram (`plot_job.sh`: sandbox tarballs as `submitter.py`, input with xrdcp, plots copied
to `--outdir` on EOS); plotdataMC.py options after `--`.

`haddoutputs.py <job_output_dir>` writes the hadd commands for the arrays to a script instead.

After normalizing (`condor_lxplus/normalizeArrays.py`, adds a final per-event `weight` branch),
merge all datasets into one MC.root and one data.root per systematic using
```
python condor_lxplus/haddFinal.py <output_dir> --mc <hadd_normalized_dir> [<hadd_normalized_dir> ...] --data <hadd_normalized_dir> [...]
```


## Notes
- Proxy handling works even if condor's native `user_proxy` method fails.
- The condor user log stays on AFS (status updates only); stdout and stderr come back once per
  job at the end. Keep the job output small.
- Needs EOS for the sandbox and the outputs (not usable on CMSConnect).

## Outputs, proxy and logs

- Each job runs `runner.py` with `--outputdir` in its local working directory and copies the
  outputs once to `--outputDir` at the end (`execute.sh`): `xrdcp` for `/eos/...` paths and
  `root://` URLs (3 attempts per file), `cp` otherwise. `submitter.py` makes a relative
  `--outputDir` absolute.
- The proxy is copied to `<job dir>/proxy` and shipped with each job (`x509userproxy`). It must
  stay valid until the last job has finished.
- `--jobDir` (or `$BTV_CONDOR_JOBDIR`) sets where `jobs_<jobName>` is created. Outside `/eos`
  (e.g. on AFS) the jobs are submitted without `-spool`. The condor user log goes to
  `jobs_<jobName>/log/`, stdout and stderr to `jobs_<jobName>/out/` and `jobs_<jobName>/err/`.
