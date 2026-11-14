# Light Condor submission script for lxplus

The jobs submitted by this script rely on eos/afs being mounted on the condor nodes.
**This is good for processing a few hundred files only.** Larger submissions are likely to make afs unstable.

- It does not create a new installation of conda/mamba and instead uses a preinstalled conda env (you can replace this with your own conda/mamba path in the PATH variable, if needed).
- It does not create a tarball of the BTVNanoComm code either, simply `cd`s to the working directory in eos/afs.
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


## Pros
- This is likely faster and will run out of the box.
- Proxy handling works even if condor's native `user_proxy` method fails.

## Cons
- Relies on eos/afs mount, hence jobs will fail to run on condor nodes where the mount is unstable or when afs is overloaded. **You may need to keep releasing jobs in that case.**
- Will not work where eos is not mounted, e.g. on CMSConnect nodes.

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
