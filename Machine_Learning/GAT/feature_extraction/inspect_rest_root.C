#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TObjArray.h"
#include "TKey.h"
#include "TList.h"
#include "TClass.h"
#include "TROOT.h"
#include "TSystem.h"

#include "TRestHitsEvent.h"
#include "TRestG4Event.h"

// -----------------------------------------------------------------------------
// Inspection macro for a REST elecsim ROOT file.
//
// Prints:
//   1. Top-level keys stored in the TFile (trees, metadata, TRestRun, etc.).
//   2. For every TTree found: total entries and the full branch/leaf tree.
//   3. For the "EventTree": pretty summary of the first N events, including
//      the fields exposed by TRestHitsEvent and TRestG4Event.
//
// Usage:
//   restRoot -b -q 'inspect_rest_root.C("../output/elecsim/2nubb_ex/2nubb_ex_N200_33.root")'
//   restRoot -b -q 'inspect_rest_root.C("input.root", 3)'         // print 3 events
//   restRoot -b -q 'inspect_rest_root.C("input.root", 3, 20)'     // print 3 events, 20 hits each
//   INPUT_FILE=path/to.root restRoot -b -q inspect_rest_root.C
// -----------------------------------------------------------------------------

static void print_section(const char* title) {
    std::cout << "\n" << std::string(78, '=') << "\n"
              << " " << title << "\n"
              << std::string(78, '=') << std::endl;
}

static void print_subsection(const char* title) {
    std::cout << "\n" << std::string(78, '-') << "\n"
              << " " << title << "\n"
              << std::string(78, '-') << std::endl;
}

static void print_file_keys(TFile* file) {
    print_section("Top-level keys in TFile");
    TList* keyList = file->GetListOfKeys();
    if (!keyList) {
        std::cout << "  (no keys)" << std::endl;
        return;
    }
    TIter next(keyList);
    TKey* key = 0;
    int idx = 0;
    while ((key = dynamic_cast<TKey*>(next()))) {
        std::cout << "  [" << idx++ << "] "
                  << "name='"  << key->GetName()     << "'  "
                  << "class='" << key->GetClassName() << "'  "
                  << "title='" << key->GetTitle()    << "'  "
                  << "cycle="  << key->GetCycle()
                  << std::endl;
    }
}

static void print_tree_structure(TTree* tree) {
    if (!tree) return;

    std::cout << "\nTree name : "  << tree->GetName()
              << "\nTree title: "  << tree->GetTitle()
              << "\nEntries   : "  << tree->GetEntries()
              << std::endl;

    TObjArray* branches = tree->GetListOfBranches();
    if (!branches) {
        std::cout << "  (no branches)" << std::endl;
        return;
    }

    std::cout << "\n  Branches (" << branches->GetEntries() << "):" << std::endl;
    std::cout << "  " << std::left
              << std::setw(32) << "name"
              << std::setw(28) << "class / type"
              << std::setw(12) << "entries"
              << "title" << std::endl;
    std::cout << "  " << std::string(78, '.') << std::endl;

    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* br = dynamic_cast<TBranch*>(branches->At(i));
        if (!br) continue;

        const char* className = br->GetClassName();
        std::string typeStr = (className && className[0] != '\0') ? className : "";
        if (typeStr.empty()) {
            TLeaf* leaf = br->GetLeaf(br->GetName());
            if (leaf) typeStr = leaf->GetTypeName() ? leaf->GetTypeName() : "";
        }

        std::cout << "  " << std::left
                  << std::setw(32) << br->GetName()
                  << std::setw(28) << typeStr
                  << std::setw(12) << br->GetEntries()
                  << br->GetTitle() << std::endl;

        // Sub-branches, one level deep, to keep output readable.
        TObjArray* sub = br->GetListOfBranches();
        if (sub && sub->GetEntries() > 0) {
            for (int j = 0; j < sub->GetEntries(); ++j) {
                TBranch* sb = dynamic_cast<TBranch*>(sub->At(j));
                if (!sb) continue;

                const char* subClass = sb->GetClassName();
                std::string subType = (subClass && subClass[0] != '\0') ? subClass : "";
                if (subType.empty()) {
                    TLeaf* sleaf = sb->GetLeaf(sb->GetName());
                    if (sleaf) subType = sleaf->GetTypeName() ? sleaf->GetTypeName() : "";
                }

                std::cout << "      -> " << std::left
                          << std::setw(28) << sb->GetName()
                          << std::setw(28) << subType
                          << std::setw(12) << sb->GetEntries()
                          << sb->GetTitle() << std::endl;
            }
        }
    }
}

static void print_all_trees(TFile* file) {
    print_section("Trees found in TFile (structure)");
    TList* keyList = file->GetListOfKeys();
    if (!keyList) return;

    TIter next(keyList);
    TKey* key = 0;
    while ((key = dynamic_cast<TKey*>(next()))) {
        TClass* cl = TClass::GetClass(key->GetClassName());
        if (!cl) continue;
        if (!cl->InheritsFrom(TTree::Class())) continue;

        TTree* tree = dynamic_cast<TTree*>(file->Get(key->GetName()));
        if (!tree) continue;
        print_tree_structure(tree);
    }
}

static void print_hits_event(TRestHitsEvent* hitsEvent, int maxHitsToPrint) {
    if (!hitsEvent) {
        std::cout << "  TRestHitsEvent: <null>" << std::endl;
        return;
    }

    const Int_t nHits = hitsEvent->GetNumberOfHits();
    Double_t totalEnergy = 0.0;
    for (Int_t i = 0; i < nHits; ++i) totalEnergy += hitsEvent->GetEnergy(i);

    std::cout << "  TRestHitsEvent"
              << "\n    id             : " << hitsEvent->GetID()
              << "\n    n_hits         : " << nHits
              << "\n    total_energy   : " << totalEnergy
              << std::endl;

    if (nHits <= 0) return;

    const int nShow = (maxHitsToPrint < 0 || maxHitsToPrint > nHits) ? nHits : maxHitsToPrint;
    std::cout << "    first " << nShow << " hits (x, y, z, energy):" << std::endl;
    std::cout << "      " << std::left
              << std::setw(6)  << "i"
              << std::setw(14) << "x [mm]"
              << std::setw(14) << "y [mm]"
              << std::setw(14) << "z [mm]"
              << "energy [keV]" << std::endl;

    for (int i = 0; i < nShow; ++i) {
        std::cout << "      " << std::left
                  << std::setw(6)  << i
                  << std::setw(14) << hitsEvent->GetX(i)
                  << std::setw(14) << hitsEvent->GetY(i)
                  << std::setw(14) << hitsEvent->GetZ(i)
                  << hitsEvent->GetEnergy(i) << std::endl;
    }
    if (nShow < nHits) {
        std::cout << "      ... (" << (nHits - nShow) << " more hits)" << std::endl;
    }
}

static void print_g4_event(TRestG4Event* g4Event) {
    if (!g4Event) {
        std::cout << "  TRestG4Event  : <null>" << std::endl;
        return;
    }

    const Int_t nPrimaries = g4Event->GetNumberOfPrimaries();
    const Int_t nTracks    = g4Event->GetNumberOfTracks();
    const TVector3 origin  = g4Event->GetPrimaryEventOrigin();

    std::cout << "  TRestG4Event"
              << "\n    id                       : " << g4Event->GetID()
              << "\n    n_primaries              : " << nPrimaries
              << "\n    n_tracks                 : " << nTracks
              << "\n    primary_origin (x,y,z)   : (" << origin.X() << ", " << origin.Y() << ", " << origin.Z() << ")"
              << "\n    total_deposited_energy   : " << g4Event->GetTotalDepositedEnergy()
              << "\n    sensitive_volume_energy  : " << g4Event->GetSensitiveVolumeEnergy()
              << std::endl;

    for (Int_t i = 0; i < nPrimaries; ++i) {
        const TString particleName = g4Event->GetPrimaryEventParticleName(i);
        const TVector3 direction   = g4Event->GetPrimaryEventDirection(i);
        const Double_t energy      = g4Event->GetPrimaryEventEnergy(i);
        std::cout << "    primary[" << i << "]  name='" << particleName
                  << "'  energy=" << energy
                  << "  dir=(" << direction.X() << ", " << direction.Y() << ", " << direction.Z() << ")"
                  << std::endl;
    }
}

int inspect_rest_root(const char* inputFilename,
                      Int_t eventsToPrint = 2,
                      Int_t hitsPerEvent = 10,
                      const char* treeName = "EventTree",
                      const char* hitsBranchName = "TRestHitsEventBranch",
                      const char* g4BranchName = "TRestG4EventBranch") {
    if (!inputFilename || inputFilename[0] == '\0') {
        std::cerr << "ERROR: empty input filename" << std::endl;
        return 1;
    }

    std::cout << "Opening file: " << inputFilename << std::endl;

    TFile* file = TFile::Open(inputFilename, "READ");
    if (!file || file->IsZombie()) {
        std::cerr << "ERROR: cannot open input file: " << inputFilename << std::endl;
        if (file) { file->Close(); delete file; }
        return 2;
    }

    std::cout << "File size on disk : " << file->GetSize() << " bytes" << std::endl;
    std::cout << "Compression level : " << file->GetCompressionLevel() << std::endl;

    print_file_keys(file);
    print_all_trees(file);

    // Detailed event-level inspection of the main tree.
    TTree* tree = dynamic_cast<TTree*>(file->Get(treeName));
    if (!tree) {
        std::cerr << "\nERROR: tree '" << treeName << "' not found. "
                  << "Skipping per-event summary." << std::endl;
        file->Close();
        delete file;
        return 3;
    }

    TBranch* hitsBranch = tree->GetBranch(hitsBranchName);
    TBranch* g4Branch   = tree->GetBranch(g4BranchName);

    TRestHitsEvent* hitsEvent = 0;
    TRestG4Event*   g4Event   = 0;

    if (hitsBranch) tree->SetBranchAddress(hitsBranchName, &hitsEvent);
    else            std::cerr << "\nWARNING: branch '" << hitsBranchName << "' not found." << std::endl;

    if (g4Branch)   tree->SetBranchAddress(g4BranchName,   &g4Event);
    else            std::cerr << "\nWARNING: branch '" << g4BranchName   << "' not found." << std::endl;

    const Long64_t nEntries = tree->GetEntries();
    Long64_t nShow = eventsToPrint;
    if (nShow < 0 || nShow > nEntries) nShow = nEntries;

    print_section("Per-event summary");
    std::cout << "Tree '" << treeName << "' has " << nEntries << " entries. "
              << "Printing first " << nShow << " events, up to "
              << hitsPerEvent << " hits per event." << std::endl;

    for (Long64_t iEvent = 0; iEvent < nShow; ++iEvent) {
        tree->GetEntry(iEvent);

        char header[128];
        snprintf(header, sizeof(header), "Event %lld", (long long)iEvent);
        print_subsection(header);

        print_hits_event(hitsEvent, hitsPerEvent);
        std::cout << std::endl;
        print_g4_event(g4Event);
    }

    file->Close();
    delete file;

    std::cout << "\nDone." << std::endl;
    return 0;
}

// Default interactive entry point (no arguments).
//
// Batch/explicit use:
//   restRoot -b -q 'inspect_rest_root.C("input.root")'
//   restRoot -b -q 'inspect_rest_root.C("input.root", 3, 20)'
//
// Environment-variable driven:
//   INPUT_FILE=... restRoot -b -q inspect_rest_root.C
int inspect_rest_root() {
    const char* envInput = gSystem->Getenv("INPUT_FILE");
    const char* inputFile = (envInput && envInput[0] != '\0')
        ? envInput
        : "../output/elecsim/2nubb_ex/2nubb_ex_N200_33.root";
    return inspect_rest_root(inputFile, 2, 10, "EventTree",
                             "TRestHitsEventBranch", "TRestG4EventBranch");
}
