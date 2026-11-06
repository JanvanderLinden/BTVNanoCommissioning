// Native-ROOT replacement for the write step of normalizeArrays.py. uproot 4.3.7's TTree
// writer has been observed to silently write malformed baskets for some jagged/vector branches
// in these files (e.g. SelJet_eta/SelJet_phi come out unreadable or fast-clone-fragile after a
// uproot.open()+uproot.recreate() round trip, even though the same branches read back cleanly
// from the pre-normalization hadd'd file). CloneTree()+Fill() here only ever copies branch
// values ROOT itself already wrote and read correctly, and adds `weight`/`lumi` the same way
// hadd's own native TTree code works -- so it doesn't hit that bug.
void normalizeArray(const char *in_file, const char *out_file, double lumi_pb, bool dry_run = false) {
  const std::vector<std::string> WEIGHT_BRANCHES = {
      "genWeight", "ttbar_weight", "puwei", "muWeight", "eleWeight", "ftagWeight"};

  TFile *f = TFile::Open(in_file);
  if (!f || f->IsZombie()) {
    fprintf(stderr, "WARNING: could not open %s, skipping\n", in_file);
    return;
  }

  TTree *tec = (TTree *)f->Get("TotalEventCount");
  if (!tec) {
    printf("WARNING: no TotalEventCount tree in %s (malformed hadd?), skipping\n", in_file);
    return;
  }
  Long64_t tec_val = 0;
  tec->SetBranchAddress("TotalEventCount", &tec_val);
  double n_total = 0;
  for (Long64_t i = 0; i < tec->GetEntries(); i++) {
    tec->GetEntry(i);
    n_total += tec_val;
  }

  TTree *tew = (TTree *)f->Get("TotalEventWeight");
  bool is_mc = (tew != nullptr);
  double sumw = 0;
  if (is_mc) {
    Double_t tew_val = 0;
    tew->SetBranchAddress("TotalEventWeight", &tew_val);
    for (Long64_t i = 0; i < tew->GetEntries(); i++) {
      tew->GetEntry(i);
      sumw += tew_val;
    }
  }

  TTree *events = (TTree *)f->Get("Events");
  if (!events || events->GetEntries() == 0) {
    printf("WARNING: no selected events in %s, skipping\n", in_file);
    return;
  }

  if (is_mc && sumw == 0) {
    fprintf(stderr, "FATAL: sumw is 0 for %s, cannot normalize\n", in_file);
    gSystem->Exit(1);
  }

  if (is_mc) {
    printf("%s: MC, n_processed=%.0f, sumw=%.6g, lumi=%.6g/pb\n", in_file, n_total, sumw, lumi_pb);
  } else {
    printf("%s: data, n_processed=%.0f\n", in_file, n_total);
  }
  if (dry_run) {
    f->Close();
    return;
  }

  TFile *fout = TFile::Open(out_file, "RECREATE");
  TTree *outtree = events->CloneTree(0);

  Double_t weight_val = 0, lumi_val = lumi_pb;
  outtree->Branch("weight", &weight_val, "weight/D");
  outtree->Branch("lumi", &lumi_val, "lumi/D");

  Double_t xSecWeight = 0;
  std::vector<Double_t> wbranch_vals(WEIGHT_BRANCHES.size(), 0);
  if (is_mc) {
    events->SetBranchAddress("xSecWeight", &xSecWeight);
    for (size_t k = 0; k < WEIGHT_BRANCHES.size(); k++) {
      events->SetBranchAddress(WEIGHT_BRANCHES[k].c_str(), &wbranch_vals[k]);
    }
  }

  Long64_t n = events->GetEntries();
  for (Long64_t i = 0; i < n; i++) {
    events->GetEntry(i);
    if (is_mc) {
      double w = xSecWeight * (lumi_pb / sumw);
      for (size_t k = 0; k < WEIGHT_BRANCHES.size(); k++) w *= wbranch_vals[k];
      weight_val = w;
    } else {
      weight_val = 1.0;
    }
    outtree->Fill();
  }
  outtree->Write();

  TTree tec_out("TotalEventCount", "TotalEventCount");
  Long64_t tec_out_val = (Long64_t)n_total;
  tec_out.Branch("TotalEventCount", &tec_out_val, "TotalEventCount/L");
  tec_out.Fill();
  tec_out.Write();

  if (is_mc) {
    TTree tew_out("TotalEventWeight", "TotalEventWeight");
    Double_t tew_out_val = sumw;
    tew_out.Branch("TotalEventWeight", &tew_out_val, "TotalEventWeight/D");
    tew_out.Fill();
    tew_out.Write();
  }

  fout->Close();
  f->Close();
}
