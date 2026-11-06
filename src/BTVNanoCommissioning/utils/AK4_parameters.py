correction_config = {
    "2016preVFP-UL": {
        "default": {
            "DC": "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
            "LUM": None,
            "JME": {
                "MC": "Summer20UL16APVNanoV15_V1 Summer20UL16APV_JRV5",
                "Run2016B": "Summer20UL16APVNanoV15_V1",
                "Run2016C": "Summer20UL16APVNanoV15_V1",
                "Run2016D": "Summer20UL16APVNanoV15_V1",
                "Run2016E": "Summer20UL16APVNanoV15_V1",
                "Run2016F": "Summer20UL16APVNanoV15_V1",
            },
            "jetveto": {"Summer19UL16_V1": "jetvetomap"},
            "JME_path": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016preVFP-UL-NanoAODv15/latest/jet_jerc.json.gz",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
            },
            "EGM": {
                "ele_Reco 2016 Electron-ID-SF": "",
                "ele_ID 2016 Electron-ID-SF": "wp80iso",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
            "roccor": "",
            "cvmfs_override": {
                "MUO": "Run2-2016preVFP-UL-NanoAODv9",  # Nanov15 unavailable
                "muonSS": "Run2-2016preVFP-UL-NanoAODv9",  # Nanov15 unavailable
                "EGM": "Run2-2016preVFP-UL-NanoAODv15",
                "electronSS": "Run2-2016preVFP-UL-NanoAODv15",
                "jetveto": "Run2-2016preVFP-UL-NanoAODv15",
                "JME": "Run2-2016preVFP-UL-NanoAODv15",
            },
        },
        "tt_semilep_mu": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
                "mu_Trig": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # in the future this should be something like NUM_IsoMu24_DEN_CutBasedIdTight_and_promptMVA_WP64ID
            },
        },
        "tt_semilep_el": {
            "EGM": {
                "ele_Reco 2016preVFP UL-Electron-ID-SF": "",
                "ele_ID 2016preVFP UL-Electron-ID-SF": "wp80iso",
            },
        },
    },
    "2016postVFP-UL": {
        "default": {
            "DC": "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
            "LUM": None,
            "JME": {
                "MC": "Summer20UL16NanoV15_V1 Summer20UL16_JRV5",
                "Run2016F": "Summer20UL16NanoV15_V1",
                "Run2016G": "Summer20UL16NanoV15_V1",
                "Run2016H": "Summer20UL16NanoV15_V1",
            },
            "jetveto": {"Summer19UL16_V1": "jetvetomap"},
            "JME_path": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016postVFP-UL-NanoAODv15/latest/jet_jerc.json.gz",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
            },
            "EGM": {
                "ele_Reco 2016 Electron-ID-SF": "",
                "ele_ID 2016 Electron-ID-SF": "wp80iso",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
            "roccor": "",
            "cvmfs_override": {
                "MUO": "Run2-2016postVFP-UL-NanoAODv9",  # Nanov15 unavailable
                "muonSS": "Run2-2016postVFP-UL-NanoAODv9",  # Nanov15 unavailable
                "EGM": "Run2-2016postVFP-UL-NanoAODv15",
                "electronSS": "Run2-2016postVFP-UL-NanoAODv15",
                "jetveto": "Run2-2016postVFP-UL-NanoAODv15",
                "JME": "Run2-2016postVFP-UL-NanoAODv15",
            },
        },
        "tt_semilep_mu": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
                "mu_Trig": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # in the future this should be something like NUM_IsoMu24_DEN_CutBasedIdTight_and_promptMVA_WP64ID
            },
        },
        "tt_semilep_el": {
            "EGM": {
                "ele_Reco 2016postVFP UL-Electron-ID-SF": "",
                "ele_ID 2016postVFP UL-Electron-ID-SF": "wp80iso",
            },
        },
    },
    "2017-UL": {
        "default": {
            "DC": "Cert_294927-306462_13TeV_UL2017_Collisions17_MuonJSON.txt",
            "LUM": None,
            "JME": {
                "MC": "Summer20UL17NanoV15_V1 Summer19UL17_JRV4",
                "Run2017B": "Summer20UL17NanoV15_V1",
                "Run2017C": "Summer20UL17NanoV15_V1",
                "Run2017D": "Summer20UL17NanoV15_V1",
                "Run2017E": "Summer20UL17NanoV15_V1",
                "Run2017F": "Summer20UL17NanoV15_V1",
            },
            "jetveto": {"Summer19UL17_V1": "jetvetomap"},
            "JME_path": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2017-UL-NanoAODv15/latest/jet_jerc.json.gz",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
            },
            "EGM": {
                "ele_Reco 2017 Electron-ID-SF": "",
                "ele_ID 2017 Electron-ID-SF": "wp80iso",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
            "roccor": "",
            "cvmfs_override": {
                "MUO": "Run2-2017-UL-NanoAODv9",  # Nanov15 unavailable
                "muonSS": "Run2-2017-UL-NanoAODv9",  # Nanov15 unavailable
                "EGM": "Run2-2017-UL-NanoAODv15",
                "electronSS": "Run2-2017-UL-NanoAODv15",
                "jetveto": "Run2-2017-UL-NanoAODv15",
            },
        },  # obj_eff only: pileup weights and the IsoMu27 trigger SF (other workflows keep "default")
        "ObjEffM": {
            "LUM": "puWeights.json.gz",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
                "mu_Trig": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            },
        },
        "tt_semilep_mu": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
                "mu_Trig": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",  # in the future this should be something like NUM_IsoMu24_DEN_CutBasedIdTight_and_promptMVA_WP64ID
            },
        },
        "tt_semilep_el": {
            "EGM": {
                "ele_Reco 2017 UL-Electron-ID-SF": "",
                "ele_ID 2017 UL-Electron-ID-SF": "wp80iso",
            },
        },
        "ObjEffE": {"LUM": "puWeights.json.gz"},
    },
    "2018-UL": {
        "default": {
            "DC": "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
            "LUM": "puweight_2018_UL.HLTPFJet40.histo",
            "JME": {
                "MC": "Summer20UL18NanoV15_V1 Summer19UL18_JRV3",
                "Run2018A": "Summer20UL18NanoV15_V1",
                "Run2018B": "Summer20UL18NanoV15_V1",
                "Run2018C": "Summer20UL18NanoV15_V1",
                "Run2018D": "Summer20UL18NanoV15_V1",
            },
            "jetveto": {"Summer19UL18_V1": "jetvetomap"},
            "JME_path": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2018-UL-NanoAODv15/latest/jet_jerc.json.gz",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
            },
            "EGM": {
                "ele_Reco 2018 Electron-ID-SF": "",
                "ele_ID 2018 Electron-ID-SF": "wp80iso",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
            "roccor": "",
            "cvmfs_override": {
                "MUO": "Run2-2018-UL-NanoAODv9",  # Nanov15 unavailable
                "muonSS": "Run2-2018-UL-NanoAODv9",  # Nanov15 unavailable
                "EGM": "Run2-2018-UL-NanoAODv15",
                "electronSS": "Run2-2018-UL-NanoAODv15",
                "jetveto": "Run2-2018-UL-NanoAODv15",
            },
        },
        "tt_semilep_mu": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightRelIso_DEN_TightIDandIPCut",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # in the future this should be something like NUM_IsoMu24_DEN_CutBasedIdTight_and_promptMVA_WP64ID
            },
        },
        "tt_semilep_el": {
            "EGM": {
                "ele_Reco 2018 UL-Electron-ID-SF": "",
                "ele_ID 2018 UL-Electron-ID-SF": "wp80iso",
            },
        },
    },
    "Summer22": {
        "default": {
            "DC": "Cert_Collisions2022_355100_362760_Golden.json",
            "LUM": "puwei_2022_preEE.histo.root",  # new PU files, based on preEE
            "JME": {
                "MC": "Summer22_22Sep2023_V4 Summer22_22Sep2023_JRV2",
                "Run2022C": "Summer22_22Sep2023_V4",
                "Run2022D": "Summer22_22Sep2023_V4",
            },  # update to latest JEC
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
            },
            "EGM": {
                "ele_ID 2022Re-recoBCD Electron-ID-SF": "Tight",
                "ele_Reco 2022Re-recoBCD Electron-ID-SF": "",
            },
            "JPCalib": {
                "Run2022C": "calibeHistoWrite_Data2022C_NANO130X_v1.root",
                "Run2022D": "calibeHistoWrite_Data2022D_NANO130X_v1.root",
                "MC": "calibeHistoWrite_MC2022_NANO130X_v2.root",
            },
            "jetveto": {"Summer22_23Sep2023_RunCD_V1": "jetvetomap"},
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "DYM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs
                # "mu_Trig": "",
            },
        },
        "DYE_2D": {
            "EGM": {
                "ele_ID 2022Re-recoBCD Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoBCD Electron-ID-SF": "",
                # TODO: need to derive double lepton trigger SFs
                # "ele_Trig 2022Re-recoBCD Electron-HLT-SF": "",
            },
        },
        "WcM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
        },
        "WcE_2D": {
            "EGM": {
                "ele_ID 2022Re-recoBCD Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoBCD Electron-ID-SF": "",
                "ele_Trig 2022Re-recoBCD Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "ttdilep_sf_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            "EGM": {
                "ele_ID 2022Re-recoBCD Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoBCD Electron-ID-SF": "",
                "ele_Trig 2022Re-recoBCD Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "semittM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "semittE_2D": {
            "EGM": {
                "ele_ID 2022Re-recoBCD Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoBCD Electron-ID-SF": "",
                "ele_Trig 2022Re-recoBCD Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
    },
    "Summer22EE": {
        "default": {
            "DC": "Cert_Collisions2022_355100_362760_Golden.json",
            "LUM": "puwei_2022_postEE.histo.root",  # new PU file, post EE
            "JME": {
                "MC": "Summer22EE_22Sep2023_V4 Summer22EE_22Sep2023_JRV2",
                "Run2022E": "Summer22EE_22Sep2023_V4",
                "Run2022F": "Summer22EE_22Sep2023_V4",
                "Run2022G": "Summer22EE_22Sep2023_V4",
            },
            "JES_MC_year": "2022EE",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
            },
            "EGM": {
                "ele_ID 2022Re-recoE+PromptFG Electron-ID-SF": "Tight",
                "ele_Reco 2022Re-recoE+PromptFG Electron-ID-SF": "",
            },
            "jetveto": {"Summer22EE_23Sep2023_RunEFG_V1": "jetvetomap"},
            # use for BTA production, jet probablity
            "JPCalib": {
                "Run2022E": "calibeHistoWrite_Data2022F_NANO130X_v1.root",
                "Run2022F": "calibeHistoWrite_Data2022F_NANO130X_v1.root",
                "Run2022G": "calibeHistoWrite_Data2022G_NANO130X_v1.root",
                "MC": "calibeHistoWrite_MC2022EE_NANO130X_v1.root",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "DYM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs
                # "mu_Trig": "",
            },
        },
        "DYE_2D": {
            "EGM": {
                "ele_ID 2022Re-recoE+PromptFG Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoE+PromptFG Electron-ID-SF": "",
                # TODO: need to derive double lepton trigger SFs
                # "ele_Trig 2022Re-recoE+PromptFG Electron-HLT-SF": "",
            },
        },
        "WcM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
        },
        "WcE_2D": {
            "EGM": {
                "ele_ID 2022Re-recoE+PromptFG Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoE+PromptFG Electron-ID-SF": "",
                "ele_Trig 2022Re-recoE+PromptFG Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "ttdilep_sf_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            "EGM": {
                "ele_ID 2022Re-recoE+PromptFG Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoE+PromptFG Electron-ID-SF": "",
                "ele_Trig 2022Re-recoE+PromptFG Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "semittM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "semittE_2D": {
            "EGM": {
                "ele_ID 2022Re-recoE+PromptFG Electron-ID-SF": "wp80iso",
                "ele_Reco 2022Re-recoE+PromptFG Electron-ID-SF": "",
                "ele_Trig 2022Re-recoE+PromptFG Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
    },
    "Summer23": {
        "default": {
            "DC": "Cert_Collisions2023_366442_370790_Golden.json",
            "LUM": "puwei_Summer23.histo.root",
            "JME": {
                "MC": "Summer23Prompt23_V4 Summer23Prompt23_RunCv1234_JRV3",
                "Run2023C": "Summer23Prompt23_V4",
            },
            "jetveto": {"Summer23Prompt23_RunC_V1": "jetvetomap"},
            "JPCalib": {
                "Run2023C-22Sep2023_v1": "calibeHistoWrite_Data2023C-22Sep2023_v1.root",
                "Run2023C-22Sep2023_v2": "calibeHistoWrite_Data2023C-22Sep2023_v2.root",
                "Run2023C-22Sep2023_v3": "calibeHistoWrite_Data2023C-22Sep2023_v3.root",
                "Run2023C-22Sep2023_v4": "calibeHistoWrite_Data2023C-22Sep2023_v4.root",
                "MC": "calibeHistoWrite_MC2023_Summer23.root",
            },
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
            },
            "EGM": {
                "ele_ID 2023PromptC Electron-ID-SF": "Tight",
                "ele_Reco 2023PromptC Electron-ID-SF": "",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "DYM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs
                # "mu_Trig": "",
            },
        },
        "DYE_2D": {
            "EGM": {
                "ele_ID 2023PromptC Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptC Electron-ID-SF": "",
                # TODO: need to derive double lepton trigger SFs
                # "ele_Trig 2023PromptC Electron-HLT-SF": "",
            },
        },
        "WcM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
        },
        "WcE_2D": {
            "EGM": {
                "ele_ID 2023PromptC Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptC Electron-ID-SF": "",
                "ele_Trig 2023PromptC Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "ttdilep_sf_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            "EGM": {
                "ele_ID 2023PromptC Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptC Electron-ID-SF": "",
                "ele_Trig 2023PromptC Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "semittM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "semittE_2D": {
            "EGM": {
                "ele_ID 2023PromptC Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptC Electron-ID-SF": "",
                "ele_Trig 2023PromptC Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
    },
    "Summer23BPix": {
        "default": {
            "DC": "Cert_Collisions2023_366442_370790_Golden.json",
            "LUM": "puwei_Summer23BPix.histo.root",
            "JME": {
                "MC": "Summer23BPixPrompt23_V4 Summer23BPixPrompt23_RunD_JRV3",
                "Run2023D": "Summer23BPixPrompt23_V4",
            },
            "JES_MC_year": "2023BPix",
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
            },
            "EGM": {
                "ele_ID 2023PromptD Electron-ID-SF": "Tight",
                "ele_Reco 2023PromptD Electron-ID-SF": "",
            },
            # This is from Mikko https://indico.cern.ch/event/1315421/contributions/5532963/attachments/2697975/4683826/2023_08_16_jetvetomaps_v3.pdf
            "jetveto": {"Summer23BPixPrompt23_RunD_V1": "jetvetomap"},
            "JPCalib": {
                "Run2023D-22Sep2023_v1": "calibeHistoWrite_Data2023D-22Sep2023_v1.root",
                "Run2023D-22Sep2023_v2": "calibeHistoWrite_Data2023D-22Sep2023_v2.root",
                "MC": "calibeHistoWrite_MC2023_Summer23BPix.root",
            },
            "muonSS": "",
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "DYM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs
                # "mu_Trig": "",
            },
        },
        "DYE_2D": {
            "EGM": {
                "ele_ID 2023PromptD Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptD Electron-ID-SF": "",
                # TODO: need to derive double lepton trigger SFs
                # "ele_Trig 2023PromptD Electron-HLT-SF": "",
            },
        },
        "WcM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
        },
        "WcE_2D": {
            "EGM": {
                "ele_ID 2023PromptD Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptD Electron-ID-SF": "",
                "ele_Trig 2023PromptD Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "ttdilep_sf_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            "EGM": {
                "ele_ID 2023PromptD Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptD Electron-ID-SF": "",
                "ele_Trig 2023PromptD Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
        },
        "semittM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_TightPFIso_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "semittE_2D": {
            "EGM": {
                "ele_ID 2023PromptD Electron-ID-SF": "wp80iso",
                "ele_Reco 2023PromptD Electron-ID-SF": "",
                "ele_Trig 2023PromptD Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",
            },
            # # "BTV": {"ctag": "ctagging.json.gz"},
        },
    },
    "Summer24": {
        "default": {
            "DC": "Cert_Collisions2024_378981_386951_Golden.json",
            # 2024 analyses use eras C-I only (no 2024B)
            "LUM": "puWeights_CDEFGHI.json.gz",
            "JME": {
                "MC": "Summer24Prompt24_V5 Summer24Prompt24_JRV2",
                "Run2024C": "Summer24Prompt24_V5",
                "Run2024D": "Summer24Prompt24_V5",
                "Run2024E": "Summer24Prompt24_V5",
                "Run2024F": "Summer24Prompt24_V5",
                "Run2024G": "Summer24Prompt24_V5",
                "Run2024H": "Summer24Prompt24_V5",
                "Run2024I": "Summer24Prompt24_V5",
            },
            "jetveto": {"Summer24Prompt24_RunBCDEFGHI_V1": "jetvetomap"},
            # "MUO": {
            #     "mu_ID": "NUM_TightID_DEN_TrackerMuons",
            #     "mu_Iso": "NUM_TightPFIso_DEN_TightID",
            # },
            # "EGM": {
            #     "ele_Reco 2024Prompt Electron-ID-SF": "",
            #     "ele_ID 2024Prompt Electron-ID-SF": "wp80iso",
            # },
            # "muonSS": "",
            # "electronSS": [
            #     "Scale",
            #     "SmearAndSyst",
            # ],
        },
        "tt_dilep": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
            },
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
            },
            # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "tt_semilep_mu": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # in the future this should be something like NUM_IsoMu24_DEN_CutBasedIdTight_and_promptMVA_WP64ID
            },
        },
        "tt_semilep_el": {
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
                "ele_Trig 2024Prompt Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",  # in the future this should be something like HLT_SF_Ele30_PromptMVAID
            },
        },
        "DYM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs
                # "mu_Trig": "",
            },
            "muonSS": "",
        },
        "DYE_2D": {
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
                # TODO: need to derive double lepton trigger SFs
                # "ele_Trig 2024Prompt Electron-HLT-SF": "",
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "WcM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # for now
            },
            "muonSS": "",
        },
        "WcE_2D": {
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                "ele_Trig 2024Prompt Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",  # for now
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "ttdilep_sf_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs with prompt MVA
                # "mu_Trig": "",
            },
            "muonSS": "",
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
                # TODO: need to derive double lepton trigger SFs with prompt MVA
                # "ele_Trig 2024Prompt Electron-HLT-SF": "",
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "semittM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # for now
            },
            "muonSS": "",
            # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "semittE_2D": {
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                "ele_Trig 2024Prompt Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",  # for now
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
            # "BTV": {"ctag": "ctagging.json.gz"},
        },
    },
    "Prompt25": {
        "default": {
            "DC": "Cert_Collisions2025_391658_398860_Golden.json",
            "LUM": "puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz",
            # Year-dependent JES regrouped uncertainties use the MC JEC campaign year,
            # not the data year.  Summer24 JECs have "_2024_" in the correction names.
            "JME": {
                "MC": "Summer24Prompt25_V3 Summer24Prompt25_JRV2",
                "Run2025C": "Summer24Prompt25_V3",
                "Run2025D": "Summer24Prompt25_V3",
                "Run2025E": "Summer24Prompt25_V3",
                "Run2025F": "Summer24Prompt25_V3",
                "Run2025G": "Summer24Prompt25_V3",
            },
            "jetveto": {"Winter25Prompt25_RunCDEFG_V1": "jetvetomap"},
            # "MUO": {
            #     "mu_ID": "NUM_TightID_DEN_TrackerMuons",
            #     "mu_Iso": "NUM_TightPFIso_DEN_TightID",
            # },
            # "EGM": {
            #     "ele_Reco 2025Prompt Electron-ID-SF": "",
            #     "ele_ID 2025Prompt Electron-ID-SF": "wp80iso",
            # },
            # "muonSS": "",
            # "electronSS": [
            #     "Scale",
            #     "SmearAndSyst",
            # ],
            # Per-POG CVMFS path overrides.
            # JME: data L2L3Residual from Winter25 JME era.
            # JME_MC: MC truth JECs must come from the Summer24 era (L2Relative differs).
            # jetveto: jetvetomap from Winter25 JME era.
            "cvmfs_override": {
                "JME": "Run3-25Prompt-Summer24-NanoAODv15",
                "JME_MC": "Run3-25Prompt-Summer24-NanoAODv15",
                "jetveto": "Run3-25Prompt-Summer24-NanoAODv15",
            },
        },
        "tt_semilep_mu": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # in the future this should be something like NUM_IsoMu24_DEN_CutBasedIdTight_and_promptMVA_WP64ID
            },
        },
        "tt_semilep_el": {
            "EGM": {
                "ele_Reco 2024Prompt Electron-ID-SF": "",
                "ele_ID 2024Prompt Electron-ID-SF": "PromptMVA-Tight",
                "ele_Trig 2024Prompt Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",  # in the future this should be something like HLT_SF_Ele30_PromptMVAID
            },
        },
        "DYM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs
                # "mu_Trig": "",
            },
            "muonSS": "",
        },
        "DYE_2D": {
            "EGM": {
                "ele_Reco 2025Prompt Electron-ID-SF": "",
                # TODO: no 2025 electron PromptMVA ID SFs available yet
                "ele_ID 2025Prompt Electron-ID-SF": "wp80iso",  # for now
                # TODO: need to derive double lepton trigger SFs
                # "ele_Trig 2025Prompt Electron-HLT-SF": "",
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "WcM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # for now
            },
            "muonSS": "",
        },
        "WcE_2D": {
            "EGM": {
                "ele_Reco 2025Prompt Electron-ID-SF": "",
                # TODO: no 2025 electron PromptMVA ID SFs available yet
                "ele_ID 2025Prompt Electron-ID-SF": "wp80iso",  # for now
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                # TODO: no 2025 electron trigger SFs available yet
                # "ele_Trig 2025Prompt Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",  # for now
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "ttdilep_sf_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive double lepton trigger SFs with prompt MVA
                # "mu_Trig": "",
            },
            "muonSS": "",
            "EGM": {
                "ele_Reco 2025Prompt Electron-ID-SF": "",
                # TODO: no 2025 electron PromptMVA ID SFs available yet
                "ele_ID 2025Prompt Electron-ID-SF": "wp80iso",  # for now
                # TODO: need to derive double lepton trigger SFs with prompt MVA
                # "ele_Trig 2025Prompt Electron-HLT-SF": "",
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
        },
        "semittM_2D": {
            "MUO": {
                "mu_ID": "NUM_TightID_DEN_TrackerMuons",
                "mu_Iso": "NUM_promptMVA_WP64ID_DEN_TightID",
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                "mu_Trig": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",  # for now
            },
            "muonSS": "",
            # "BTV": {"ctag": "ctagging.json.gz"},
        },
        "semittE_2D": {
            "EGM": {
                "ele_Reco 2025Prompt Electron-ID-SF": "",
                # TODO: no 2025 electron PromptMVA ID SFs available yet
                "ele_ID 2025Prompt Electron-ID-SF": "wp80iso",  # for now
                # TODO: need to derive single lepton trigger SFs with prompt MVA
                # TODO: no 2025 electron trigger SFs available yet
                # "ele_Trig 2025Prompt Electron-HLT-SF": "HLT_SF_Ele30_MVAiso80ID",  # for now
            },
            "electronSS": [
                "Scale",
                "SmearAndSyst",
            ],
            # "BTV": {"ctag": "ctagging.json.gz"},
        },
    },
    "prompt_dataMC": {
        "default": {"DC": "$PROMPT_DATAMC"},
    },
}

# CustomSummer24: private NanoAODv15 production (extra taggers, phys03/USER samples)
# on top of the same Summer24 MiniAOD conditions -- reuse Summer24 corrections as-is.
correction_config["CustomSummer24"] = correction_config["Summer24"]
correction_config["CustomSummer24UParTPaper"] = correction_config["Summer24"]

# CustomSummer25: CustomSummer24 private MC + private 2025 data NanoTuples. Uses the
# official "2025 data on Summer24 MC" JME era (Run3-25Prompt-Summer24-NanoAODv15) for
# MC and data alike; lepton SFs as for Prompt25 (2025Prompt EGM, 2025 MUO).
correction_config["CustomSummer25"] = {
    **correction_config["Prompt25"],
    "default": {
        "DC": "Cert_Collisions2025_391658_398903_Golden.json",
        "LUM": "puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz",
        "JME": {
            "MC": "Summer24Prompt25_V3 Summer24Prompt25_JRV2",
            "Run2025B": "Summer24Prompt25_V3",
            "Run2025C": "Summer24Prompt25_V3",
            "Run2025D": "Summer24Prompt25_V3",
            "Run2025E": "Summer24Prompt25_V3",
            "Run2025F": "Summer24Prompt25_V3",
            "Run2025G": "Summer24Prompt25_V3",
        },
        "jetveto": {"Summer24Prompt25_RunCDEFG_V1": "jetvetomap"},
        # The 2025Prompt EGM era has no electron HLT SFs and no PromptMVA ID SFs (the
        # selection uses PromptMVA-Tight): take electron reco/ID/HLT SFs from the 2024 era as a
        # stand-in. Electron scale/smearing keeps its own era (separate "electronSS" key).
        "cvmfs_override": {
            "EGM": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"
        },
    },
}
for _wf, _cfg in correction_config["CustomSummer25"].items():
    if (
        _wf != "default"
        and "EGM" in _cfg
        and "EGM" in correction_config["Summer24"].get(_wf, {})
    ):
        correction_config["CustomSummer25"][_wf] = {
            **_cfg,
            "EGM": correction_config["Summer24"][_wf]["EGM"],
        }
