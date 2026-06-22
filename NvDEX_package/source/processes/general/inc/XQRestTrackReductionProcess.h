///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             XQRestTrackReductionProcess.cxx
///
///             Mar 2020:   Created by LiTao SYSU
//
///_______________________________________________________________________________

#ifndef RestCore_XQRestTrackReductionProcess
#define RestCore_XQRestTrackReductionProcess

#include "TString.h"

#include <TRestTrackEvent.h>
#include "TRestEventProcess.h"

class XQRestTrackReductionProcess : public TRestEventProcess {
   private:
#ifndef __CINT__
    TRestTrackEvent* fInputTrackEvent;   //!
    TRestTrackEvent* fOutputTrackEvent;  //!
#endif

    void InitFromConfigFile();

    void Initialize();

   protected:
     Double_t fStartingDistance;
     Double_t fMinimumDistance;
     Double_t fDistanceFactor;
     Double_t fStartingNodes;
     Double_t fMaxNodes;
     Double_t fHitsEnergy;

   public:
    void InitProcess();
    void BeginOfEventProcess();
    TRestEvent* ProcessEvent(TRestEvent* eventInput);
    void EndOfEventProcess();
    void EndProcess();
    void LoadDefaultConfig();

    void LoadConfig(std::string cfgFilename, std::string name = "");

    void PrintMetadata() {
        BeginPrintProcess();

        std::cout << " Starting distance : " << fStartingDistance << std::endl;
        std::cout << " Minimum distance : " << fMinimumDistance << std::endl;
        std::cout << " Distance step factor : " << fDistanceFactor << std::endl;
        std::cout << " Minimum number of nodes : " << fStartingNodes << std::endl;
        std::cout << " Maximum number of nodes : " << fMaxNodes << std::endl;
        std::cout << " Single Hit Energy Cut : " << fHitsEnergy << " keV. " << std::endl;

        EndPrintProcess();
    }

    TString GetProcessName() { return (TString) "XQRestTrackReductionProcess"; }

    // Constructor
    XQRestTrackReductionProcess();
    XQRestTrackReductionProcess(char* cfgFileName);
    // Destructor
    ~XQRestTrackReductionProcess();

    ClassDef(XQRestTrackReductionProcess,
             1);  // Template for a REST "event process" class inherited from
                  // TRestEventProcess
};
#endif
