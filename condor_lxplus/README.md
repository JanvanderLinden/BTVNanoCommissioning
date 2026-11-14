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
  queue at a time; the others are added as these finish. `request_disk = --disk` (10GB).
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
  `<outputDir>`, `--name`), the input of `scripts/plotdataMC.py`;
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
`setup.sh`, sourced only for hadd). `--only hists|arrays` merges one of the two.

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
