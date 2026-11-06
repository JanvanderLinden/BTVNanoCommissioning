// Reads back every entry/branch of a merged ROOT file to catch corruption that `hadd`
// can silently introduce (its exit code stays 0 even when it writes unreadable baskets --
// see e.g. ROOT's TTreeCloner fast-clone mishandling jagged branches from uproot-written
// files). Exits nonzero the moment any basket/branch read error is reported, before it can
// cascade into a crash reading a later branch.
void abortOnError(Int_t level, Bool_t /*abort_bool*/, const char *location, const char *msg) {
  if (level >= kError) {
    fprintf(stderr, "FATAL (validateMerge): %s: %s\n", location, msg);
    gSystem->Exit(1);
  }
}

void validateMerge(const char *path, const char *treename = "Events") {
  SetErrorHandler(abortOnError);

  TFile *f = TFile::Open(path);
  if (!f || f->IsZombie()) {
    fprintf(stderr, "FATAL (validateMerge): cannot open %s\n", path);
    gSystem->Exit(1);
  }

  TTree *t = (TTree *)f->Get(treename);
  if (!t) {
    fprintf(stderr, "FATAL (validateMerge): no tree '%s' in %s\n", treename, path);
    gSystem->Exit(1);
  }

  Long64_t n = t->GetEntries();
  for (Long64_t i = 0; i < n; i++) {
    Int_t nb = t->GetEntry(i);
    if (nb <= 0) {
      fprintf(stderr, "FATAL (validateMerge): GetEntry(%lld) returned %d in %s\n", i, nb, path);
      gSystem->Exit(1);
    }
  }

  printf("OK (validateMerge): %s tree=%s entries=%lld\n", path, treename, n);
  gSystem->Exit(0);
}
