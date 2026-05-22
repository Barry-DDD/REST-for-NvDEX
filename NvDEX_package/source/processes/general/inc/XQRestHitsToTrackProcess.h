///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             XQTRestTrackReductionProcess.cxx
///
///             Mar 2020:   Created by LiTao SYSU
//
///_______________________________________________________________________________

#ifndef RestCore_XQRestHitsToTrackProcess
#define RestCore_XQRestHitsToTrackProcess

#include <TRestHitsEvent.h>
#include <TRestTrackEvent.h>
#include "TRestEventProcess.h"

class XQRestHitsToTrackProcess : public TRestEventProcess {
   private:
#ifndef __CINT__
    TRestHitsEvent* fHitsEvent;    //!
    TRestTrackEvent* fTrackEvent;  //!
#endif

    void InitFromConfigFile();

    void Initialize();

  protected:
    TVector2 fEnergyRange;

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

        std::cout << " All Deposited Energy Range : ( " << fEnergyRange.X() << ", " << fEnergyRange.Y() << " )" << std::endl;

        EndPrintProcess();
    }

    TString GetProcessName() { return (TString) "XQRestHitsToTrackProcess"; }

    // Constructor
    XQRestHitsToTrackProcess();
    XQRestHitsToTrackProcess(char* cfgFileName);
    // Destructor
    ~XQRestHitsToTrackProcess();

    ClassDef(XQRestHitsToTrackProcess,
             1);  // Template for a REST "event process" class inherited from
                  // TRestEventProcess
};
#endif
