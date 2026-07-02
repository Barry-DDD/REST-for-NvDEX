#include <iostream>
#include <vector>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TStopwatch.h"

struct InputConfig {
    std::string label;
    std::string file_path;
    Long64_t n_select;
};

struct HitEvent {
    int event_id = 0;
    int n_hits = 0;
    float total_energy = 0.0f;
    int n_primaries = 0;
    int n_tracks = 0;
    float primary_origin_x = 0.0f;
    float primary_origin_y = 0.0f;
    float primary_origin_z = 0.0f;
    float g4_total_deposited_energy = 0.0f;
    float g4_sensitive_volume_energy = 0.0f;

    std::vector<std::string>* primary_particle_name = nullptr;
    std::vector<float>* primary_energy = nullptr;
    std::vector<float>* primary_dir_x = nullptr;
    std::vector<float>* primary_dir_y = nullptr;
    std::vector<float>* primary_dir_z = nullptr;

    std::vector<float>* x = nullptr;
    std::vector<float>* y = nullptr;
    std::vector<float>* z = nullptr;
    std::vector<float>* energy = nullptr;
    std::vector<float>* log_energy = nullptr;

    std::vector<int>* module_id = nullptr;
    std::vector<int>* module_col = nullptr;
    std::vector<int>* module_row = nullptr;
    std::vector<int>* module_q = nullptr;
    std::vector<int>* module_r = nullptr;

    std::vector<int>* local_col = nullptr;
    std::vector<int>* local_row = nullptr;
    std::vector<int>* local_q = nullptr;
    std::vector<int>* local_r = nullptr;

    std::vector<int>* q = nullptr;
    std::vector<int>* r = nullptr;
    std::vector<int>* valid_geometry = nullptr;

    //std::vector<float>* pad_center_x = nullptr;
    //std::vector<float>* pad_center_y = nullptr;
    std::vector<float>* residual_x = nullptr;
    std::vector<float>* residual_y = nullptr;
    //std::vector<float>* nearest_pad_distance = nullptr;

    int id = 0;

    void ResetVectorPointers()
    {
        primary_particle_name = nullptr;
        primary_energy = nullptr;
        primary_dir_x = nullptr;
        primary_dir_y = nullptr;
        primary_dir_z = nullptr;

        x = nullptr;
        y = nullptr;
        z = nullptr;
        energy = nullptr;
        log_energy = nullptr;

        module_id = nullptr;
        module_col = nullptr;
        module_row = nullptr;
        module_q = nullptr;
        module_r = nullptr;

        local_col = nullptr;
        local_row = nullptr;
        local_q = nullptr;
        local_r = nullptr;

        q = nullptr;
        r = nullptr;
        valid_geometry = nullptr;

        //pad_center_x = nullptr;
        //pad_center_y = nullptr;
        residual_x = nullptr;
        residual_y = nullptr;
        //nearest_pad_distance = nullptr;
    }
};

bool CheckBranch(TTree* tree, const char* name)
{
    if (!tree->GetBranch(name)) {
        std::cerr << "Error: missing branch '" << name
                  << "' in tree '" << tree->GetName() << "'" << std::endl;
        return false;
    }
    return true;
}

bool SetInputBranches(TTree* tree, HitEvent& ev)
{
    const char* required_branches[] = {
        "event_id",
        "n_hits",
        "total_energy",
        "n_primaries",
        "n_tracks",
        "primary_origin_x",
        "primary_origin_y",
        "primary_origin_z",
        "g4_total_deposited_energy",
        "g4_sensitive_volume_energy",
        "primary_particle_name",
        "primary_energy",
        "primary_dir_x",
        "primary_dir_y",
        "primary_dir_z",
        "x",
        "y",
        "z",
        "energy",
        "log_energy",
        "module_id",
        "module_col",
        "module_row",
        "module_q",
        "module_r",
        "local_col",
        "local_row",
        "local_q",
        "local_r",
        "q",
        "r",
        "valid_geometry",
        //"pad_center_x",
        //"pad_center_y",
        "residual_x",
        "residual_y",
        //"nearest_pad_distance",
        "id"
    };

    for (const char* br : required_branches) {
        if (!CheckBranch(tree, br)) return false;
    }

    tree->SetBranchStatus("*", 1);
    tree->ResetBranchAddresses();

    ev.ResetVectorPointers();

    tree->SetBranchAddress("event_id", &ev.event_id);
    tree->SetBranchAddress("n_hits", &ev.n_hits);
    tree->SetBranchAddress("total_energy", &ev.total_energy);
    tree->SetBranchAddress("n_primaries", &ev.n_primaries);
    tree->SetBranchAddress("n_tracks", &ev.n_tracks);
    tree->SetBranchAddress("primary_origin_x", &ev.primary_origin_x);
    tree->SetBranchAddress("primary_origin_y", &ev.primary_origin_y);
    tree->SetBranchAddress("primary_origin_z", &ev.primary_origin_z);
    tree->SetBranchAddress("g4_total_deposited_energy", &ev.g4_total_deposited_energy);
    tree->SetBranchAddress("g4_sensitive_volume_energy", &ev.g4_sensitive_volume_energy);

    tree->SetBranchAddress("primary_particle_name", &ev.primary_particle_name);
    tree->SetBranchAddress("primary_energy", &ev.primary_energy);
    tree->SetBranchAddress("primary_dir_x", &ev.primary_dir_x);
    tree->SetBranchAddress("primary_dir_y", &ev.primary_dir_y);
    tree->SetBranchAddress("primary_dir_z", &ev.primary_dir_z);

    tree->SetBranchAddress("x", &ev.x);
    tree->SetBranchAddress("y", &ev.y);
    tree->SetBranchAddress("z", &ev.z);
    tree->SetBranchAddress("energy", &ev.energy);
    tree->SetBranchAddress("log_energy", &ev.log_energy);

    tree->SetBranchAddress("module_id", &ev.module_id);
    tree->SetBranchAddress("module_col", &ev.module_col);
    tree->SetBranchAddress("module_row", &ev.module_row);
    tree->SetBranchAddress("module_q", &ev.module_q);
    tree->SetBranchAddress("module_r", &ev.module_r);

    tree->SetBranchAddress("local_col", &ev.local_col);
    tree->SetBranchAddress("local_row", &ev.local_row);
    tree->SetBranchAddress("local_q", &ev.local_q);
    tree->SetBranchAddress("local_r", &ev.local_r);

    tree->SetBranchAddress("q", &ev.q);
    tree->SetBranchAddress("r", &ev.r);
    tree->SetBranchAddress("valid_geometry", &ev.valid_geometry);

    //tree->SetBranchAddress("pad_center_x", &ev.pad_center_x);
    //tree->SetBranchAddress("pad_center_y", &ev.pad_center_y);
    tree->SetBranchAddress("residual_x", &ev.residual_x);
    tree->SetBranchAddress("residual_y", &ev.residual_y);
    //tree->SetBranchAddress("nearest_pad_distance", &ev.nearest_pad_distance);

    tree->SetBranchAddress("id", &ev.id);

    return true;
}

void CreateOutputBranches(TTree* tree, HitEvent& ev)
{
    tree->Branch("event_id", &ev.event_id, "event_id/I");
    tree->Branch("n_hits", &ev.n_hits, "n_hits/I");
    tree->Branch("total_energy", &ev.total_energy, "total_energy/F");
    tree->Branch("n_primaries", &ev.n_primaries, "n_primaries/I");
    tree->Branch("n_tracks", &ev.n_tracks, "n_tracks/I");
    tree->Branch("primary_origin_x", &ev.primary_origin_x, "primary_origin_x/F");
    tree->Branch("primary_origin_y", &ev.primary_origin_y, "primary_origin_y/F");
    tree->Branch("primary_origin_z", &ev.primary_origin_z, "primary_origin_z/F");
    tree->Branch("g4_total_deposited_energy", &ev.g4_total_deposited_energy, "g4_total_deposited_energy/F");
    tree->Branch("g4_sensitive_volume_energy", &ev.g4_sensitive_volume_energy, "g4_sensitive_volume_energy/F");

    tree->Branch("primary_particle_name", &ev.primary_particle_name);
    tree->Branch("primary_energy", &ev.primary_energy);
    tree->Branch("primary_dir_x", &ev.primary_dir_x);
    tree->Branch("primary_dir_y", &ev.primary_dir_y);
    tree->Branch("primary_dir_z", &ev.primary_dir_z);

    tree->Branch("x", &ev.x);
    tree->Branch("y", &ev.y);
    tree->Branch("z", &ev.z);
    tree->Branch("energy", &ev.energy);
    tree->Branch("log_energy", &ev.log_energy);

    tree->Branch("module_id", &ev.module_id);
    tree->Branch("module_col", &ev.module_col);
    tree->Branch("module_row", &ev.module_row);
    tree->Branch("module_q", &ev.module_q);
    tree->Branch("module_r", &ev.module_r);

    tree->Branch("local_col", &ev.local_col);
    tree->Branch("local_row", &ev.local_row);
    tree->Branch("local_q", &ev.local_q);
    tree->Branch("local_r", &ev.local_r);

    tree->Branch("q", &ev.q);
    tree->Branch("r", &ev.r);
    tree->Branch("valid_geometry", &ev.valid_geometry);

    //tree->Branch("pad_center_x", &ev.pad_center_x);
    //tree->Branch("pad_center_y", &ev.pad_center_y);
    tree->Branch("residual_x", &ev.residual_x);
    tree->Branch("residual_y", &ev.residual_y);
    //tree->Branch("nearest_pad_distance", &ev.nearest_pad_distance);

    tree->Branch("id", &ev.id, "id/I");
}

void select_mixed_manual()
{
    const std::string input_dir =
        "/public/home/liuz1/work/26.03.18_rest/nubb_ex/output/detsim_normal_root/selected_E1MeV";

    const std::string output_file =
        "/public/home/liuz1/work/26.03.18_rest/nubb_ex/output/detsim_normal_root/selected_E1MeV/detsim_select80K_mixed.root";

    const std::string tree_name = "HitTree";

    // elecsim:
    //std::vector<InputConfig> inputs = {
    //    {"2nubb_ex", input_dir + "/2nubb_ex.root", 24000},  //43.75k in total
    //    {"co60",     input_dir + "/co60.root",      6000},
    //    {"th232",    input_dir + "/th232.root",     6000},
    //    {"u238",     input_dir + "/u238.root",      6000},
    //    {"2nubb_gs", input_dir + "/2nubb_gs.root",  6000}
    //};

    //const Long64_t expected_total = 48000;

    // detsim: 
    std::vector<InputConfig> inputs = {
        {"2nubb_ex", input_dir + "/2nubb_ex.root", 40000},
        {"co60",     input_dir + "/co60.root",      10000},
        {"th232",    input_dir + "/th232.root",     10000},
        {"u238",     input_dir + "/u238.root",      10000},
        {"2nubb_gs", input_dir + "/2nubb_gs.root",  10000}
    };
    const Long64_t expected_total = 80000;




    TStopwatch timer;
    timer.Start();

    TString out_dir = gSystem->DirName(output_file.c_str());
    gSystem->mkdir(out_dir, true);

    TFile* fout = TFile::Open(output_file.c_str(), "RECREATE");
    if (!fout || fout->IsZombie()) {
        std::cerr << "Error: failed to create output file: "
                  << output_file << std::endl;
        return;
    }

    fout->SetCompressionLevel(4);

    HitEvent ev;

    TTree* out_tree = new TTree(
        tree_name.c_str(),
        "Sparse detector hits with hierarchical hex coordinates"
    );

    CreateOutputBranches(out_tree, ev);

    Long64_t total_written = 0;

    for (const auto& cfg : inputs) {
        TFile* fin = TFile::Open(cfg.file_path.c_str(), "READ");
        if (!fin || fin->IsZombie()) {
            std::cerr << "Error: failed to open input file: "
                      << cfg.file_path << std::endl;
            fout->Close();
            return;
        }

        TTree* in_tree = dynamic_cast<TTree*>(fin->Get(tree_name.c_str()));
        if (!in_tree) {
            std::cerr << "Error: tree '" << tree_name
                      << "' was not found in file: "
                      << cfg.file_path << std::endl;
            fin->Close();
            fout->Close();
            return;
        }

        const Long64_t n_entries = in_tree->GetEntries();

        std::cout << "\nInput sample: " << cfg.label << std::endl;
        std::cout << "  File: " << cfg.file_path << std::endl;
        std::cout << "  Available entries: " << n_entries << std::endl;
        std::cout << "  Requested entries: " << cfg.n_select << std::endl;

        if (n_entries < cfg.n_select) {
            std::cerr << "Error: not enough entries in sample "
                      << cfg.label << std::endl;
            std::cerr << "  requested = " << cfg.n_select
                      << ", available = " << n_entries << std::endl;
            fin->Close();
            fout->Close();
            return;
        }

        if (!SetInputBranches(in_tree, ev)) {
            std::cerr << "Error: failed to set input branches for sample "
                      << cfg.label << std::endl;
            fin->Close();
            fout->Close();
            return;
        }

        for (Long64_t i = 0; i < cfg.n_select; ++i) {
            in_tree->GetEntry(i);
            out_tree->Fill();
        }

        total_written += cfg.n_select;

        std::cout << "  Written entries: " << cfg.n_select << std::endl;

        in_tree->ResetBranchAddresses();
        fin->Close();
    }

    fout->cd();
    out_tree->Write("", TObject::kOverwrite);
    fout->Close();

    timer.Stop();

    std::cout << "\nOutput file created:\n  "
              << output_file << std::endl;

    std::cout << "Total written entries = "
              << total_written << " / expected "
              << expected_total << std::endl;

    if (total_written != expected_total) {
        std::cerr << "Warning: total written entries are not equal to "
                  << expected_total << std::endl;
    }

    std::cout << "Elapsed time: ";
    timer.Print();
}
