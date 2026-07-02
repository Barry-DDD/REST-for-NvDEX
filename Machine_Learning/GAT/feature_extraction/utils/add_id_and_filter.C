// add_id_and_filter.C
//
// Read a ROOT file containing the "HitTree" TTree, keep all original
// branches, add a new int branch "id" with a user-specified value, and
// keep only events with g4_sensitive_volume_energy > 1000.0.
//
// Usage:
//   root -l -b -q 'add_id_and_filter.C("input.root", "output.root", 42)'
//   restROOT -l -b -q 'add_id_and_filter.C("input.root", "output.root", 42)'
//
// Arguments:
//   input_file   : path to the input ROOT file
//   output_file  : path to the output ROOT file
//   id_value     : integer value written into the new "id" branch
//   tree_name    : (optional) TTree name, defaults to "HitTree"
//   cut_value    : (optional) g4_sensitive_volume_energy lower bound,
//                  defaults to 1000.0

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TString.h>

#include <iostream>

//root -l -b -q 'add_id_and_filter.C("../output/2nubb_gs_N100_51.root","../output/sel_2nubb_gs_N100_51.root",42,"HitTree",1000.0)'
//root -l -b -q 'add_id_and_filter.C("/public/home/liuz1/work/26.03.18_rest/nubb_ex/output/elecsim_normal_root/2nubb_ex.root","/public/home/liuz1/work/26.03.18_rest/nubb_ex/output/elecsim_normal_root/selected/2nubb_ex.root",1,"HitTree",1000.0)'
int add_id_and_filter(const char* input_file,
                      const char* output_file,
                      int         id_value,
                      const char* tree_name = "HitTree",
                      double      cut_value = 1000.0)
{
    // ---- Open input file ----
    TFile* fin = TFile::Open(input_file, "READ");
    if (!fin || fin->IsZombie()) {
        std::cerr << "[add_id_and_filter] ERROR: cannot open input file: "
                  << input_file << std::endl;
        if (fin) delete fin;
        return 1;
    }

    TTree* tin = dynamic_cast<TTree*>(fin->Get(tree_name));
    if (!tin) {
        std::cerr << "[add_id_and_filter] ERROR: TTree '" << tree_name
                  << "' not found in " << input_file << std::endl;
        fin->Close();
        delete fin;
        return 2;
    }

    // Bind the selection variable. The branch is a plain float scalar.
    Float_t g4_sensitive_volume_energy = 0.f;
    if (tin->SetBranchAddress("g4_sensitive_volume_energy",
                              &g4_sensitive_volume_energy) < 0) {
        std::cerr << "[add_id_and_filter] ERROR: branch "
                     "'g4_sensitive_volume_energy' not found." << std::endl;
        fin->Close();
        delete fin;
        return 3;
    }

    // ---- Create output file and clone the tree structure (0 entries) ----
    TFile* fout = TFile::Open(output_file, "RECREATE");
    if (!fout || fout->IsZombie()) {
        std::cerr << "[add_id_and_filter] ERROR: cannot open output file: "
                  << output_file << std::endl;
        if (fout) delete fout;
        fin->Close();
        delete fin;
        return 4;
    }
    fout->cd();

    // CloneTree(0) copies the full branch structure but no entries.
    TTree* tout = tin->CloneTree(0);
    if (!tout) {
        std::cerr << "[add_id_and_filter] ERROR: CloneTree(0) failed."
                  << std::endl;
        fout->Close();
        delete fout;
        fin->Close();
        delete fin;
        return 5;
    }
    tout->SetName(tree_name);

    // ---- Add the new "id" branch ----
    Int_t id_branch_value = id_value;
    TBranch* b_id = tout->Branch("id", &id_branch_value, "id/I");
    if (!b_id) {
        std::cerr << "[add_id_and_filter] ERROR: failed to add 'id' branch."
                  << std::endl;
        fout->Close();
        delete fout;
        fin->Close();
        delete fin;
        return 6;
    }

    // ---- Event loop with selection ----
    const Long64_t n_in = tin->GetEntries();
    Long64_t n_pass = 0;

    std::cout << "[add_id_and_filter] Input file : " << input_file << std::endl;
    std::cout << "[add_id_and_filter] Tree name  : " << tree_name << std::endl;
    std::cout << "[add_id_and_filter] Events before selection: " << n_in
              << std::endl;
    std::cout << "[add_id_and_filter] Adding branch 'id' with value: "
              << id_value << std::endl;
    std::cout << "[add_id_and_filter] Selection  : "
                 "g4_sensitive_volume_energy > " << cut_value << std::endl;

    for (Long64_t i = 0; i < n_in; ++i) {
        if (tin->GetEntry(i) <= 0) continue;

        if (g4_sensitive_volume_energy > cut_value) {
            // Defensive: keep id constant for every selected event.
            id_branch_value = id_value;
            tout->Fill();
            ++n_pass;
        }
    }

    std::cout << "[add_id_and_filter] Events after  selection: " << n_pass
              << std::endl;
    if (n_in > 0) {
        const double eff = 100.0 * static_cast<double>(n_pass)
                                 / static_cast<double>(n_in);
        std::cout << "[add_id_and_filter] Selection efficiency  : "
                  << eff << " %" << std::endl;
    }

    // ---- Write & close ----
    fout->cd();
    tout->Write("", TObject::kOverwrite);
    fout->Close();
    delete fout;

    fin->Close();
    delete fin;

    std::cout << "[add_id_and_filter] Done. Input entries: " << n_in
              << ", passed (g4_sensitive_volume_energy > " << cut_value
              << "): " << n_pass
              << ", id = " << id_value
              << ", output: " << output_file << std::endl;

    return 0;
}
