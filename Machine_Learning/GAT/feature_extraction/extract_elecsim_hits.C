#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TMath.h"
#include "TROOT.h"

#include "TRestHitsEvent.h"
#include "TRestG4Event.h"

// -----------------------------------------------------------------------------
// Extraction macro for a hierarchical hexagonal readout geometry.
//
// The geometry implemented here follows the XML layout (rest_workspace/NvDEX_Meta_test.rml):
//   - one module contains 16 x 16 hexagonal pads;
//   - local pad x pitch is 8 mm;
//   - local pad y pitch is 4*sqrt(3) mm;
//   - even local rows have x = col*8 + 8;
//   - odd  local rows have x = col*8 + 4;
//   - modules are arranged in six columns with 4,6,6,6,6,4 modules.
//
// Important:
// The module pitch in y is 188*sqrt(3)/3 mm. This is not an integer multiple of
// the local pad pitch 4*sqrt(3) mm. Therefore the full detector should not be
// treated as one single continuous fine hex lattice. This macro stores both the
// local pad hex coordinates and the module-level coordinates.
// -----------------------------------------------------------------------------

struct ModuleInfo {
    int id;
    int module_col;
    int module_row;
    double origin_x;
    double origin_y;
};

struct PadMatch {
    bool valid;
    int module_id;
    int module_col;
    int module_row;
    int local_col;
    int local_row;
    int local_q;
    int local_r;
    int module_q;
    int module_r;
    int q;
    int r;
    double center_x;
    double center_y;
    double residual_x;
    double residual_y;
    double distance;

    PadMatch()
        : valid(false), module_id(-1), module_col(-1), module_row(-999),
          local_col(-1), local_row(-1), local_q(0), local_r(0),
          module_q(0), module_r(0), q(0), r(0), center_x(0.0), center_y(0.0),
          residual_x(0.0), residual_y(0.0), distance(0.0) {}
};

static int positive_mod2(int value) {
    int m = value % 2;
    return (m < 0) ? (m + 2) : m;
}

static int round_to_int(double x) {
    return (x >= 0.0) ? static_cast<int>(std::floor(x + 0.5))
                      : static_cast<int>(std::ceil(x - 0.5));
}

static void build_modules(std::vector<ModuleInfo>& modules) {
    modules.clear();

    const double module_pitch_x = 124.0;
    const double module_pitch_y = 188.0 * TMath::Sqrt(3.0) / 3.0;

    int id = 0;

    // XML block: modX 0..3, x = -372, y index -2..1
    for (int i = 0; i < 4; ++i) {
        ModuleInfo m;
        m.id = id++;
        m.module_col = 0;
        m.module_row = i - 2;
        m.origin_x = -3.0 * module_pitch_x;
        m.origin_y = (i - 2) * module_pitch_y;
        modules.push_back(m);
    }

    // XML blocks: four central columns, each with y index -3..2
    for (int c = 1; c <= 4; ++c) {
        for (int i = 0; i < 6; ++i) {
            ModuleInfo m;
            m.id = id++;
            m.module_col = c;
            m.module_row = i - 3;
            m.origin_x = (-3.0 + c) * module_pitch_x;
            m.origin_y = (i - 3) * module_pitch_y;
            modules.push_back(m);
        }
    }

    // XML block: modX 28..31, x = 248, y index -2..1
    for (int i = 0; i < 4; ++i) {
        ModuleInfo m;
        m.id = id++;
        m.module_col = 5;
        m.module_row = i - 2;
        m.origin_x = 2.0 * module_pitch_x;
        m.origin_y = (i - 2) * module_pitch_y;
        modules.push_back(m);
    }
}

static void offset_odd_row_to_axial(int col, int row, int& q, int& r) {
    const int odd = positive_mod2(row);
    q = col - (row - odd) / 2;
    r = row;
}

static PadMatch find_nearest_pad(double x, double y, const std::vector<ModuleInfo>& modules) {
    const double pad_pitch_x = 8.0;
    const double pad_pitch_y = 4.0 * TMath::Sqrt(3.0);

    double best_d2 = std::numeric_limits<double>::max();
    PadMatch best;

    for (size_t i = 0; i < modules.size(); ++i) {
        const ModuleInfo& mod = modules[i];

        const double local_x = x - mod.origin_x;
        const double local_y = y - mod.origin_y;

        const int row = round_to_int(local_y / pad_pitch_y);
        if (row < 0 || row > 15) continue;

        const double row_x0 = (positive_mod2(row) == 0) ? 8.0 : 4.0;
        const int col = round_to_int((local_x - row_x0) / pad_pitch_x);
        if (col < 0 || col > 15) continue;

        const double center_x = mod.origin_x + row_x0 + col * pad_pitch_x;
        const double center_y = mod.origin_y + row * pad_pitch_y;

        const double dx = x - center_x;
        const double dy = y - center_y;
        const double d2 = dx * dx + dy * dy;

        if (d2 < best_d2) {
            int local_q = 0;
            int local_r = 0;
            offset_odd_row_to_axial(col, row, local_q, local_r);

            // Module-level coordinates. These are coarse coordinates for the
            // module layout, not fine pad-lattice coordinates.
            const int module_q = mod.module_col - 3;
            const int module_r = mod.module_row;

            best_d2 = d2;
            best.valid = true;
            best.module_id = mod.id;
            best.module_col = mod.module_col;
            best.module_row = mod.module_row;
            best.local_col = col;
            best.local_row = row;
            best.local_q = local_q;
            best.local_r = local_r;
            best.module_q = module_q;
            best.module_r = module_r;

            // Hierarchical integer labels. These are unique-ish coordinates for
            // storage and grouping. Do not use their subtraction as a physical
            // hex distance across module boundaries.
            best.q = module_q * 100 + local_q;
            best.r = module_r * 100 + local_r;

            best.center_x = center_x;
            best.center_y = center_y;
            best.residual_x = dx;
            best.residual_y = dy;
            best.distance = std::sqrt(d2);
        }
    }

    return best;
}

int extract_elecsim_hits_for_ml_hier_hex(const char* inputFilename,
                                 const char* outputFilename = "hits_for_ml.root",
                                 const char* inputTreeName = "EventTree",
                                 const char* inputBranchName = "TRestHitsEventBranch",
                                 Long64_t maxEvents = -1,
                                 const char* g4BranchName = "TRestG4EventBranch") {
    std::vector<ModuleInfo> modules;
    build_modules(modules);

    TFile* inputFile = TFile::Open(inputFilename, "READ");
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "ERROR: cannot open input file: " << inputFilename << std::endl;
        return 1;
    }

    TTree* inputTree = dynamic_cast<TTree*>(inputFile->Get(inputTreeName));
    if (!inputTree) {
        std::cerr << "ERROR: input tree not found: " << inputTreeName << std::endl;
        inputFile->Close();
        delete inputFile;
        return 2;
    }

    TBranch* hitsBranch = inputTree->GetBranch(inputBranchName);
    if (!hitsBranch) {
        std::cerr << "ERROR: hits branch not found: " << inputBranchName << std::endl;
        inputFile->Close();
        delete inputFile;
        return 4;
    }

    TRestHitsEvent* hitsEvent = 0;
    inputTree->SetBranchAddress(inputBranchName, &hitsEvent);

    TBranch* g4Branch = inputTree->GetBranch(g4BranchName);
    TRestG4Event* g4Event = 0;
    if (g4Branch) {
        inputTree->SetBranchAddress(g4BranchName, &g4Event);
    } else {
        std::cerr << "WARNING: G4 truth branch not found: " << g4BranchName << std::endl;
        std::cerr << "WARNING: truth branches will be filled with default values." << std::endl;
    }

    TFile* outputFile = TFile::Open(outputFilename, "RECREATE");
    if (!outputFile || outputFile->IsZombie()) {
        std::cerr << "ERROR: cannot create output file: " << outputFilename << std::endl;
        inputFile->Close();
        delete inputFile;
        return 3;
    }

    TTree* outputTree = new TTree("HitTree", "Sparse detector hits with hierarchical hex coordinates");

    Int_t event_id = 0;
    Int_t n_hits = 0;
    Float_t total_energy = 0.0;

    Int_t n_primaries = 0;
    Int_t n_tracks = 0;
    Float_t primary_origin_x = 0.0;
    Float_t primary_origin_y = 0.0;
    Float_t primary_origin_z = 0.0;
    Float_t g4_total_deposited_energy = 0.0;
    Float_t g4_sensitive_volume_energy = 0.0;

    std::vector<float>* x = new std::vector<float>();
    std::vector<float>* y = new std::vector<float>();
    std::vector<float>* z = new std::vector<float>();
    std::vector<float>* energy = new std::vector<float>();
    std::vector<float>* log_energy = new std::vector<float>();

    std::vector<std::string>* primary_particle_name = new std::vector<std::string>();
    std::vector<float>* primary_energy = new std::vector<float>();
    std::vector<float>* primary_dir_x = new std::vector<float>();
    std::vector<float>* primary_dir_y = new std::vector<float>();
    std::vector<float>* primary_dir_z = new std::vector<float>();

    std::vector<int>* module_id = new std::vector<int>();
    std::vector<int>* module_col = new std::vector<int>();
    std::vector<int>* module_row = new std::vector<int>();
    std::vector<int>* module_q = new std::vector<int>();
    std::vector<int>* module_r = new std::vector<int>();

    std::vector<int>* local_col = new std::vector<int>();
    std::vector<int>* local_row = new std::vector<int>();
    std::vector<int>* local_q = new std::vector<int>();
    std::vector<int>* local_r = new std::vector<int>();

    std::vector<int>* q = new std::vector<int>();
    std::vector<int>* r = new std::vector<int>();
    std::vector<int>* valid_geometry = new std::vector<int>();

    std::vector<float>* pad_center_x = new std::vector<float>();
    std::vector<float>* pad_center_y = new std::vector<float>();
    std::vector<float>* residual_x = new std::vector<float>();
    std::vector<float>* residual_y = new std::vector<float>();
    std::vector<float>* nearest_pad_distance = new std::vector<float>();

    outputTree->Branch("event_id", &event_id, "event_id/I");
    outputTree->Branch("n_hits", &n_hits, "n_hits/I");
    outputTree->Branch("total_energy", &total_energy, "total_energy/F");

    outputTree->Branch("n_primaries", &n_primaries, "n_primaries/I");
    outputTree->Branch("n_tracks", &n_tracks, "n_tracks/I");
    outputTree->Branch("primary_origin_x", &primary_origin_x, "primary_origin_x/F");
    outputTree->Branch("primary_origin_y", &primary_origin_y, "primary_origin_y/F");
    outputTree->Branch("primary_origin_z", &primary_origin_z, "primary_origin_z/F");
    outputTree->Branch("g4_total_deposited_energy", &g4_total_deposited_energy, "g4_total_deposited_energy/F");
    outputTree->Branch("g4_sensitive_volume_energy", &g4_sensitive_volume_energy, "g4_sensitive_volume_energy/F");

    outputTree->Branch("primary_particle_name", &primary_particle_name);
    outputTree->Branch("primary_energy", &primary_energy);
    outputTree->Branch("primary_dir_x", &primary_dir_x);
    outputTree->Branch("primary_dir_y", &primary_dir_y);
    outputTree->Branch("primary_dir_z", &primary_dir_z);

    outputTree->Branch("x", &x);
    outputTree->Branch("y", &y);
    outputTree->Branch("z", &z);
    outputTree->Branch("energy", &energy);
    outputTree->Branch("log_energy", &log_energy);

    outputTree->Branch("module_id", &module_id);
    outputTree->Branch("module_col", &module_col);
    outputTree->Branch("module_row", &module_row);
    outputTree->Branch("module_q", &module_q);
    outputTree->Branch("module_r", &module_r);

    outputTree->Branch("local_col", &local_col);
    outputTree->Branch("local_row", &local_row);
    outputTree->Branch("local_q", &local_q);
    outputTree->Branch("local_r", &local_r);

    outputTree->Branch("q", &q);
    outputTree->Branch("r", &r);
    outputTree->Branch("valid_geometry", &valid_geometry);

    outputTree->Branch("pad_center_x", &pad_center_x);
    outputTree->Branch("pad_center_y", &pad_center_y);
    outputTree->Branch("residual_x", &residual_x);
    outputTree->Branch("residual_y", &residual_y);
    outputTree->Branch("nearest_pad_distance", &nearest_pad_distance);

    const Long64_t nInputEvents = inputTree->GetEntries();
    const Long64_t nHitsBranchEntries = hitsBranch ? hitsBranch->GetEntries() : 0;
    const Long64_t nG4BranchEntries = g4Branch ? g4Branch->GetEntries() : 0;
    Long64_t nEventsToProcess = nInputEvents;
    if (maxEvents >= 0 && maxEvents < nInputEvents) nEventsToProcess = maxEvents;

    std::cout << "Input file: " << inputFilename << std::endl;
    std::cout << "Output file: " << outputFilename << std::endl;
    std::cout << "Input tree entries: " << nInputEvents << std::endl;
    std::cout << "Hits branch entries (" << inputBranchName << "): " << nHitsBranchEntries << std::endl;
    if (g4Branch) {
        std::cout << "G4 branch entries (" << g4BranchName << "): " << nG4BranchEntries << std::endl;
        if (nHitsBranchEntries != nG4BranchEntries) {
            std::cerr << "WARNING: hits and G4 branch entry counts are different." << std::endl;
            std::cerr << "WARNING: check event synchronization before using truth labels for training." << std::endl;
        }
    }
    std::cout << "Events to process: " << nEventsToProcess << std::endl;
    std::cout << "Loaded modules from XML geometry: " << modules.size() << std::endl;

    for (Long64_t iEvent = 0; iEvent < nEventsToProcess; ++iEvent) {
        inputTree->GetEntry(iEvent);
        if (!hitsEvent) {
            std::cerr << "WARNING: null TRestHitsEvent at event " << iEvent << std::endl;
            continue;
        }

        event_id = static_cast<Int_t>(iEvent);
        n_hits = hitsEvent->GetNumberOfHits();
        total_energy = 0.0;

        n_primaries = 0;
        n_tracks = 0;
        primary_origin_x = 0.0;
        primary_origin_y = 0.0;
        primary_origin_z = 0.0;
        g4_total_deposited_energy = 0.0;
        g4_sensitive_volume_energy = 0.0;

        primary_particle_name->clear();
        primary_energy->clear();
        primary_dir_x->clear();
        primary_dir_y->clear();
        primary_dir_z->clear();

        if (g4Branch && g4Event) {
            n_primaries = g4Event->GetNumberOfPrimaries();
            n_tracks = g4Event->GetNumberOfTracks();

            const TVector3 origin = g4Event->GetPrimaryEventOrigin();
            primary_origin_x = static_cast<Float_t>(origin.X());
            primary_origin_y = static_cast<Float_t>(origin.Y());
            primary_origin_z = static_cast<Float_t>(origin.Z());

            g4_total_deposited_energy = static_cast<Float_t>(g4Event->GetTotalDepositedEnergy());
            g4_sensitive_volume_energy = static_cast<Float_t>(g4Event->GetSensitiveVolumeEnergy());

            primary_particle_name->reserve(n_primaries);
            primary_energy->reserve(n_primaries);
            primary_dir_x->reserve(n_primaries);
            primary_dir_y->reserve(n_primaries);
            primary_dir_z->reserve(n_primaries);

            for (Int_t iPrimary = 0; iPrimary < n_primaries; ++iPrimary) {
                const TString particleName = g4Event->GetPrimaryEventParticleName(iPrimary);
                const TVector3 direction = g4Event->GetPrimaryEventDirection(iPrimary);
                const Double_t primaryEnergy = g4Event->GetPrimaryEventEnergy(iPrimary);

                primary_particle_name->push_back(std::string(particleName.Data()));
                primary_energy->push_back(static_cast<Float_t>(primaryEnergy));
                primary_dir_x->push_back(static_cast<Float_t>(direction.X()));
                primary_dir_y->push_back(static_cast<Float_t>(direction.Y()));
                primary_dir_z->push_back(static_cast<Float_t>(direction.Z()));
            }
        }

        x->clear(); y->clear(); z->clear(); energy->clear(); log_energy->clear();
        module_id->clear(); module_col->clear(); module_row->clear(); module_q->clear(); module_r->clear();
        local_col->clear(); local_row->clear(); local_q->clear(); local_r->clear();
        q->clear(); r->clear(); valid_geometry->clear();
        pad_center_x->clear(); pad_center_y->clear(); residual_x->clear(); residual_y->clear(); nearest_pad_distance->clear();

        x->reserve(n_hits); y->reserve(n_hits); z->reserve(n_hits); energy->reserve(n_hits); log_energy->reserve(n_hits);
        module_id->reserve(n_hits); module_col->reserve(n_hits); module_row->reserve(n_hits); module_q->reserve(n_hits); module_r->reserve(n_hits);
        local_col->reserve(n_hits); local_row->reserve(n_hits); local_q->reserve(n_hits); local_r->reserve(n_hits);
        q->reserve(n_hits); r->reserve(n_hits); valid_geometry->reserve(n_hits);
        pad_center_x->reserve(n_hits); pad_center_y->reserve(n_hits); residual_x->reserve(n_hits); residual_y->reserve(n_hits); nearest_pad_distance->reserve(n_hits);

        for (Int_t iHit = 0; iHit < n_hits; ++iHit) {
            const float hit_x = hitsEvent->GetX(iHit);
            const float hit_y = hitsEvent->GetY(iHit);
            const float hit_z = hitsEvent->GetZ(iHit);
            const float hit_energy = hitsEvent->GetEnergy(iHit);

            const PadMatch match = find_nearest_pad(hit_x, hit_y, modules);

            x->push_back(hit_x);
            y->push_back(hit_y);
            z->push_back(hit_z);
            energy->push_back(hit_energy);
            log_energy->push_back(static_cast<float>(std::log(1.0 + std::max(0.0f, hit_energy))));

            module_id->push_back(match.module_id);
            module_col->push_back(match.module_col);
            module_row->push_back(match.module_row);
            module_q->push_back(match.module_q);
            module_r->push_back(match.module_r);

            local_col->push_back(match.local_col);
            local_row->push_back(match.local_row);
            local_q->push_back(match.local_q);
            local_r->push_back(match.local_r);

            q->push_back(match.q);
            r->push_back(match.r);
            valid_geometry->push_back(match.valid ? 1 : 0);

            pad_center_x->push_back(static_cast<float>(match.center_x));
            pad_center_y->push_back(static_cast<float>(match.center_y));
            residual_x->push_back(static_cast<float>(match.residual_x));
            residual_y->push_back(static_cast<float>(match.residual_y));
            nearest_pad_distance->push_back(static_cast<float>(match.distance));

            total_energy += hit_energy;
        }

        outputTree->Fill();

        if ((iEvent + 1) % 1000 == 0 || iEvent + 1 == nEventsToProcess) {
            std::cout << "Processed " << (iEvent + 1) << " / " << nEventsToProcess << " events" << std::endl;
        }
    }

    outputFile->cd();
    outputTree->Write();
    outputFile->Close();
    inputFile->Close();

    delete x; delete y; delete z; delete energy; delete log_energy;
    delete primary_particle_name; delete primary_energy; delete primary_dir_x; delete primary_dir_y; delete primary_dir_z;
    delete module_id; delete module_col; delete module_row; delete module_q; delete module_r;
    delete local_col; delete local_row; delete local_q; delete local_r;
    delete q; delete r; delete valid_geometry;
    delete pad_center_x; delete pad_center_y; delete residual_x; delete residual_y; delete nearest_pad_distance;
    delete outputFile;
    delete inputFile;

    std::cout << "Done. Wrote HitTree to " << outputFilename << std::endl;
    return 0;
}

// Default interactive entry point (no arguments).
//
// For batch use with explicit input/output, invoke via the quoted ROOT call:
//   restROOT -b -q 'extract_elecsim_hits.C("input.root","output.root")'
//
// For environment-variable driven batch use:
//   INPUT_FILE=... OUTPUT_FILE=... restROOT -b -q extract_elecsim_hits.C
int extract_elecsim_hits() {
    const char* envInput  = gSystem->Getenv("INPUT_FILE");
    const char* envOutput = gSystem->Getenv("OUTPUT_FILE");

    const char* inputFile  = (envInput  && envInput[0]  != '\0') ? envInput
        : "/public/home/liuz1/work/26.03.18_rest/test_example/rebuildEvents/Xe_sim_1.root";
    const char* outputFile = (envOutput && envOutput[0] != '\0') ? envOutput
        : "/public/home/liuz1/work/26.03.18_rest/test_example/rebuildEvents/normal_root/processed_Xe_sim_1.root";

    return extract_elecsim_hits(inputFile, outputFile, "EventTree", "TRestHitsEventBranch", -1, "TRestG4EventBranch");
}
