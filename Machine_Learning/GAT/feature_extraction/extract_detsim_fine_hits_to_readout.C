#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <algorithm>
#include <cstdlib>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TMath.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVector3.h"

#include "TRestHitsEvent.h"
#include "TRestG4Event.h"

// -----------------------------------------------------------------------------
// Generalized extraction macro: fine Geant4/REST hits -> hex readout cells.
//
// This is the user-configurable variant of:
//   extract_fine_hits_to_2mm_readout.C
//   extract_fine_hits_to_3mm_readout.C
//   extract_fine_hits_to_12mm_readout.C
//
// The readout pitch (in mm) is supplied as a runtime argument. The module
// layout is identical to the nominal 12 mm geometry; only the local cell
// pitch and the local grid size are rescaled so that the physical module
// footprint stays the same.
//
//   reference (nominal):   pitch = 12 mm,  16 x 16 cells per module
//   for pitch = P mm:      scale = P / 12, cell pitch and offsets * scale,
//                          local grid = ceil(16 * 12 / P) cells per side
//
// For each occupied readout cell:
//   - energy is the sum of fine-hit energies in that cell;
//   - x/y are the readout cell center coordinates;
//   - weighted_x/y/z store the energy-weighted mean position of the fine
//     hits assigned to that cell;
//   - fine_hit_count stores how many fine hits contributed to this cell.
//
// Entry points:
//   restRoot -b -q 'extract_detsim_fine_hits_to_readout.C("in.root","out.root",3.0)'
//   READOUT_PITCH_MM=8 INPUT_FILE=... OUTPUT_FILE=... \
//     restRoot -b -q extract_detsim_fine_hits_to_readout.C
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

struct AggregatedCell {
    PadMatch match;
    int fine_hit_count = 0;
    double sum_energy = 0.0;
    double sum_xe = 0.0;
    double sum_ye = 0.0;
    double sum_ze = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
};

// A unique key for one readout cell within the global module array.
// std::map is used intentionally for deterministic output order.
typedef std::tuple<int, int, int> CellKey;  // module_id, local_col, local_row

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

// Geometry parameters derived from the requested readout pitch.
// Computed once per run and passed in by reference to keep the hot loop tight.
struct ReadoutGeometry {
    double readout_pitch_mm;
    double cell_pitch_x;
    double cell_pitch_y;
    double even_row_x0;
    double odd_row_x0;
    int n_local_cols;
    int n_local_rows;
    int qr_spacing_factor;  // encodes (module, local) into global (q, r)
};

static ReadoutGeometry build_readout_geometry(double readout_pitch_mm) {
    ReadoutGeometry geo;
    geo.readout_pitch_mm = readout_pitch_mm;

    // Nominal local geometry (12 mm-like, used as reference):
    //   local pad x pitch       = 8 mm
    //   local pad y pitch       = 4 * sqrt(3) mm
    //   even-row x0             = 8 mm
    //   odd-row  x0             = 4 mm
    //   16 x 16 pads per module
    //
    // For an arbitrary pitch P (mm), scale the local pitch and offsets by
    // P/12 while keeping the module physical positions unchanged. The local
    // grid count is rescaled so the local footprint covers the same area.
    const double readout_scale = readout_pitch_mm / 12.0;
    geo.cell_pitch_x = 8.0 * readout_scale;
    geo.cell_pitch_y = 4.0 * TMath::Sqrt(3.0) * readout_scale;
    geo.even_row_x0  = 8.0 * readout_scale;
    geo.odd_row_x0   = 4.0 * readout_scale;

    // Use ceil so non-integer ratios (e.g. 8 mm -> 24 cells) are not truncated.
    // Hits outside the module footprint are naturally rejected by the row/col
    // bounds check anyway, so an over-estimate is safe.
    const double n_cells = std::ceil(16.0 * 12.0 / readout_pitch_mm);
    geo.n_local_cols = static_cast<int>(n_cells);
    geo.n_local_rows = static_cast<int>(n_cells);

    // Choose qr_spacing_factor as the smallest power of ten that exceeds
    // n_local_cols, so q = module_q * factor + local_q stays invertible.
    int factor = 100;
    while (factor <= geo.n_local_cols) factor *= 10;
    geo.qr_spacing_factor = factor;

    return geo;
}

static PadMatch find_nearest_readout_cell(double x, double y,
                                          const std::vector<ModuleInfo>& modules,
                                          const ReadoutGeometry& geo) {
    double best_d2 = std::numeric_limits<double>::max();
    PadMatch best;

    for (size_t i = 0; i < modules.size(); ++i) {
        const ModuleInfo& mod = modules[i];

        const double local_x = x - mod.origin_x;
        const double local_y = y - mod.origin_y;

        const int row = round_to_int(local_y / geo.cell_pitch_y);
        if (row < 0 || row >= geo.n_local_rows) continue;

        const double row_x0 = (positive_mod2(row) == 0) ? geo.even_row_x0 : geo.odd_row_x0;
        const int col = round_to_int((local_x - row_x0) / geo.cell_pitch_x);
        if (col < 0 || col >= geo.n_local_cols) continue;

        const double center_x = mod.origin_x + row_x0 + col * geo.cell_pitch_x;
        const double center_y = mod.origin_y + row * geo.cell_pitch_y;

        const double dx = x - center_x;
        const double dy = y - center_y;
        const double d2 = dx * dx + dy * dy;

        if (d2 < best_d2) {
            int local_q = 0;
            int local_r = 0;
            offset_odd_row_to_axial(col, row, local_q, local_r);

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

            best.q = module_q * geo.qr_spacing_factor + local_q;
            best.r = module_r * geo.qr_spacing_factor + local_r;

            best.center_x = center_x;
            best.center_y = center_y;
            best.residual_x = dx;
            best.residual_y = dy;
            best.distance = std::sqrt(d2);
        }
    }

    return best;
}

int extract_detsim_fine_hits_to_readout(const char* inputFilename,
                                 const char* outputFilename,
                                 double readout_pitch_mm,
                                 const char* inputTreeName = "EventTree",
                                 const char* inputBranchName = "TRestHitsEventBranch",
                                 Long64_t maxEvents = -1,
                                 const char* g4BranchName = "TRestG4EventBranch") {
    if (!(readout_pitch_mm > 0.0)) {
        std::cerr << "ERROR: readout_pitch_mm must be positive, got " << readout_pitch_mm << std::endl;
        return 10;
    }

    std::vector<ModuleInfo> modules;
    build_modules(modules);

    const ReadoutGeometry geo = build_readout_geometry(readout_pitch_mm);

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

    TString treeTitle = TString::Format("Fine hits aggregated onto %.3g mm hex readout cells",
                                        readout_pitch_mm);
    TTree* outputTree = new TTree("HitTree", treeTitle.Data());

    // Persist the configured pitch so downstream tools can recover the
    // readout granularity without parsing filenames.
    TTree* metaTree = new TTree("ReadoutMeta", "Readout geometry metadata");
    Float_t meta_pitch_mm = static_cast<Float_t>(readout_pitch_mm);
    Int_t meta_n_local_cols = geo.n_local_cols;
    Int_t meta_n_local_rows = geo.n_local_rows;
    Int_t meta_qr_factor = geo.qr_spacing_factor;
    metaTree->Branch("readout_pitch_mm", &meta_pitch_mm, "readout_pitch_mm/F");
    metaTree->Branch("n_local_cols", &meta_n_local_cols, "n_local_cols/I");
    metaTree->Branch("n_local_rows", &meta_n_local_rows, "n_local_rows/I");
    metaTree->Branch("qr_spacing_factor", &meta_qr_factor, "qr_spacing_factor/I");
    metaTree->Fill();

    Int_t event_id = 0;
    Int_t n_hits = 0;                  // occupied readout cells at this pitch
    Int_t n_fine_hits = 0;             // original input fine hits
    Int_t n_valid_fine_hits = 0;
    Int_t n_invalid_fine_hits = 0;
    Float_t total_energy = 0.0;        // sum of valid aggregated readout energy
    Float_t fine_total_energy = 0.0;   // sum of all input fine-hit energy
    Float_t invalid_energy = 0.0;
    Float_t readout_pitch_mm_branch = static_cast<Float_t>(readout_pitch_mm);

    Int_t n_primaries = 0;
    Int_t n_tracks = 0;
    Float_t primary_origin_x = 0.0;
    Float_t primary_origin_y = 0.0;
    Float_t primary_origin_z = 0.0;
    Float_t g4_total_deposited_energy = 0.0;
    Float_t g4_sensitive_volume_energy = 0.0;

    std::vector<float>* x = new std::vector<float>();  // readout cell center x
    std::vector<float>* y = new std::vector<float>();  // readout cell center y
    std::vector<float>* z = new std::vector<float>();  // energy-weighted fine-hit z
    std::vector<float>* weighted_x = new std::vector<float>();
    std::vector<float>* weighted_y = new std::vector<float>();
    std::vector<float>* weighted_z = new std::vector<float>();
    std::vector<float>* energy = new std::vector<float>();
    std::vector<float>* log_energy = new std::vector<float>();
    std::vector<int>* fine_hit_count = new std::vector<int>();

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

    std::vector<float>* readout_center_x = new std::vector<float>();
    std::vector<float>* readout_center_y = new std::vector<float>();
    std::vector<float>* residual_x = new std::vector<float>();  // weighted_x - center_x
    std::vector<float>* residual_y = new std::vector<float>();  // weighted_y - center_y
    std::vector<float>* nearest_cell_distance = new std::vector<float>();

    outputTree->Branch("event_id", &event_id, "event_id/I");
    outputTree->Branch("n_hits", &n_hits, "n_hits/I");
    outputTree->Branch("n_fine_hits", &n_fine_hits, "n_fine_hits/I");
    outputTree->Branch("n_valid_fine_hits", &n_valid_fine_hits, "n_valid_fine_hits/I");
    outputTree->Branch("n_invalid_fine_hits", &n_invalid_fine_hits, "n_invalid_fine_hits/I");
    outputTree->Branch("total_energy", &total_energy, "total_energy/F");
    outputTree->Branch("fine_total_energy", &fine_total_energy, "fine_total_energy/F");
    outputTree->Branch("invalid_energy", &invalid_energy, "invalid_energy/F");
    outputTree->Branch("readout_pitch_mm", &readout_pitch_mm_branch, "readout_pitch_mm/F");

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
    outputTree->Branch("weighted_x", &weighted_x);
    outputTree->Branch("weighted_y", &weighted_y);
    outputTree->Branch("weighted_z", &weighted_z);
    outputTree->Branch("energy", &energy);
    outputTree->Branch("log_energy", &log_energy);
    outputTree->Branch("fine_hit_count", &fine_hit_count);

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

    outputTree->Branch("readout_center_x", &readout_center_x);
    outputTree->Branch("readout_center_y", &readout_center_y);
    outputTree->Branch("residual_x", &residual_x);
    outputTree->Branch("residual_y", &residual_y);
    outputTree->Branch("nearest_cell_distance", &nearest_cell_distance);

    const Long64_t nInputEvents = inputTree->GetEntries();
    const Long64_t nHitsBranchEntries = hitsBranch ? hitsBranch->GetEntries() : 0;
    const Long64_t nG4BranchEntries = g4Branch ? g4Branch->GetEntries() : 0;
    Long64_t nEventsToProcess = nInputEvents;
    if (maxEvents >= 0 && maxEvents < nInputEvents) nEventsToProcess = maxEvents;

    std::cout << "Input file: " << inputFilename << std::endl;
    std::cout << "Output file: " << outputFilename << std::endl;
    std::cout << "Requested readout pitch: " << readout_pitch_mm << " mm" << std::endl;
    std::cout << "Local grid per module: " << geo.n_local_cols << " x " << geo.n_local_rows
              << " cells, qr_spacing_factor = " << geo.qr_spacing_factor << std::endl;
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
        n_fine_hits = hitsEvent->GetNumberOfHits();
        n_valid_fine_hits = 0;
        n_invalid_fine_hits = 0;
        n_hits = 0;
        total_energy = 0.0;
        fine_total_energy = 0.0;
        invalid_energy = 0.0;

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

        std::map<CellKey, AggregatedCell> cells;

        for (Int_t iHit = 0; iHit < n_fine_hits; ++iHit) {
            const double hit_x = hitsEvent->GetX(iHit);
            const double hit_y = hitsEvent->GetY(iHit);
            const double hit_z = hitsEvent->GetZ(iHit);
            const double hit_energy = hitsEvent->GetEnergy(iHit);

            fine_total_energy += static_cast<Float_t>(hit_energy);

            const PadMatch match = find_nearest_readout_cell(hit_x, hit_y, modules, geo);
            if (!match.valid) {
                ++n_invalid_fine_hits;
                invalid_energy += static_cast<Float_t>(hit_energy);
                continue;
            }

            ++n_valid_fine_hits;
            CellKey key(match.module_id, match.local_col, match.local_row);
            AggregatedCell& cell = cells[key];
            if (cell.fine_hit_count == 0) {
                cell.match = match;
            }

            cell.fine_hit_count += 1;
            cell.sum_energy += hit_energy;
            cell.sum_xe += hit_x * hit_energy;
            cell.sum_ye += hit_y * hit_energy;
            cell.sum_ze += hit_z * hit_energy;
            cell.sum_x += hit_x;
            cell.sum_y += hit_y;
            cell.sum_z += hit_z;
        }

        x->clear(); y->clear(); z->clear();
        weighted_x->clear(); weighted_y->clear(); weighted_z->clear();
        energy->clear(); log_energy->clear(); fine_hit_count->clear();
        module_id->clear(); module_col->clear(); module_row->clear(); module_q->clear(); module_r->clear();
        local_col->clear(); local_row->clear(); local_q->clear(); local_r->clear();
        q->clear(); r->clear(); valid_geometry->clear();
        readout_center_x->clear(); readout_center_y->clear(); residual_x->clear(); residual_y->clear(); nearest_cell_distance->clear();

        n_hits = static_cast<Int_t>(cells.size());

        x->reserve(n_hits); y->reserve(n_hits); z->reserve(n_hits);
        weighted_x->reserve(n_hits); weighted_y->reserve(n_hits); weighted_z->reserve(n_hits);
        energy->reserve(n_hits); log_energy->reserve(n_hits); fine_hit_count->reserve(n_hits);
        module_id->reserve(n_hits); module_col->reserve(n_hits); module_row->reserve(n_hits); module_q->reserve(n_hits); module_r->reserve(n_hits);
        local_col->reserve(n_hits); local_row->reserve(n_hits); local_q->reserve(n_hits); local_r->reserve(n_hits);
        q->reserve(n_hits); r->reserve(n_hits); valid_geometry->reserve(n_hits);
        readout_center_x->reserve(n_hits); readout_center_y->reserve(n_hits); residual_x->reserve(n_hits); residual_y->reserve(n_hits); nearest_cell_distance->reserve(n_hits);

        for (std::map<CellKey, AggregatedCell>::const_iterator it = cells.begin(); it != cells.end(); ++it) {
            const AggregatedCell& cell = it->second;
            const PadMatch& match = cell.match;
            const double e = cell.sum_energy;

            double wx = 0.0;
            double wy = 0.0;
            double wz = 0.0;
            if (e > 0.0) {
                wx = cell.sum_xe / e;
                wy = cell.sum_ye / e;
                wz = cell.sum_ze / e;
            } else if (cell.fine_hit_count > 0) {
                // Fallback for zero or negative deposited energy.
                wx = cell.sum_x / cell.fine_hit_count;
                wy = cell.sum_y / cell.fine_hit_count;
                wz = cell.sum_z / cell.fine_hit_count;
            }

            x->push_back(static_cast<float>(match.center_x));
            y->push_back(static_cast<float>(match.center_y));
            z->push_back(static_cast<float>(wz));
            weighted_x->push_back(static_cast<float>(wx));
            weighted_y->push_back(static_cast<float>(wy));
            weighted_z->push_back(static_cast<float>(wz));
            energy->push_back(static_cast<float>(e));
            log_energy->push_back(static_cast<float>(std::log(1.0 + std::max(0.0, e))));
            fine_hit_count->push_back(cell.fine_hit_count);

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
            valid_geometry->push_back(1);

            readout_center_x->push_back(static_cast<float>(match.center_x));
            readout_center_y->push_back(static_cast<float>(match.center_y));
            residual_x->push_back(static_cast<float>(wx - match.center_x));
            residual_y->push_back(static_cast<float>(wy - match.center_y));
            nearest_cell_distance->push_back(static_cast<float>(std::sqrt((wx - match.center_x) * (wx - match.center_x) +
                                                                         (wy - match.center_y) * (wy - match.center_y))));

            total_energy += static_cast<Float_t>(e);
        }

        outputTree->Fill();

        if ((iEvent + 1) % 1000 == 0 || iEvent + 1 == nEventsToProcess) {
            std::cout << "Processed " << (iEvent + 1) << " / " << nEventsToProcess
                      << " events, last event fine hits = " << n_fine_hits
                      << ", occupied cells = " << n_hits
                      << ", invalid fine hits = " << n_invalid_fine_hits << std::endl;
        }
    }

    outputFile->cd();
    outputTree->Write();
    metaTree->Write();
    outputFile->Close();
    inputFile->Close();

    delete x; delete y; delete z; delete weighted_x; delete weighted_y; delete weighted_z;
    delete energy; delete log_energy; delete fine_hit_count;
    delete primary_particle_name; delete primary_energy; delete primary_dir_x; delete primary_dir_y; delete primary_dir_z;
    delete module_id; delete module_col; delete module_row; delete module_q; delete module_r;
    delete local_col; delete local_row; delete local_q; delete local_r;
    delete q; delete r; delete valid_geometry;
    delete readout_center_x; delete readout_center_y; delete residual_x; delete residual_y; delete nearest_cell_distance;
    delete outputFile;
    delete inputFile;

    std::cout << "Done. Wrote HitTree (and ReadoutMeta) to " << outputFilename << std::endl;
    return 0;
}

// Default interactive / environment-variable driven entry point.
//
// Batch examples:
//   restRoot -b -q 'extract_detsim_fine_hits_to_readout.C("in.root","out.root",3.0)'
//
// Environment-variable driven batch use:
//   READOUT_PITCH_MM=8 INPUT_FILE=... OUTPUT_FILE=... \
//     restRoot -b -q extract_detsim_fine_hits_to_readout.C
int extract_detsim_fine_hits_to_readout() {
    const char* envInput  = gSystem->Getenv("INPUT_FILE");
    const char* envOutput = gSystem->Getenv("OUTPUT_FILE");
    const char* envPitch  = gSystem->Getenv("READOUT_PITCH_MM");

    const char* inputFile  = (envInput  && envInput[0]  != '\0') ? envInput
        : "/public/home/liuz1/work/26.03.18_rest/test_example/rebuildEvents/detsim/Xe_sim_10.root";

    double pitch_mm = 3.0;
    if (envPitch && envPitch[0] != '\0') {
        char* end = 0;
        const double parsed = std::strtod(envPitch, &end);
        if (end && end != envPitch && parsed > 0.0) {
            pitch_mm = parsed;
        } else {
            std::cerr << "WARNING: invalid READOUT_PITCH_MM='" << envPitch
                      << "', falling back to " << pitch_mm << " mm" << std::endl;
        }
    }

    TString defaultOutput = TString::Format(
        "/public/home/liuz1/work/26.03.18_rest/test_example/rebuildEvents/normal_root_detsim/processed_Xe_sim_10_%.3gmm.root",
        pitch_mm);
    const char* outputFile = (envOutput && envOutput[0] != '\0') ? envOutput : defaultOutput.Data();

    return extract_detsim_fine_hits_to_readout(inputFile, outputFile, pitch_mm,
                                        "EventTree", "TRestHitsEventBranch", -1, "TRestG4EventBranch");
}
