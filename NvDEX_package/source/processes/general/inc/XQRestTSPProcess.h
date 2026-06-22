///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs Modified
///
///             XQRestTSPProcess.cxx

///             Date : Oct/2019
///             Author : Tao Li, SYSU
///
///_______________________________________________________________________________

#ifndef RestCore_XQRestTSPProcess
#define RestCore_XQRestTSPProcess

#include "TString.h"

#include <TRestHitsEvent.h>
#include <TRestGas.h>
#include "TRestEventProcess.h"

class XQRestTSPProcess : public TRestEventProcess {
   private:
#ifndef __CINT__
    TRestHitsEvent* fInputHitsEvent;   //!
    TRestHitsEvent* fOutputHitsEvent;  //!
#endif

    void InitFromConfigFile();

    void Initialize();

    void LoadDefaultConfig();

    Double_t fSetGap;
    Double_t fNAdjoin;  // unit 1

   protected:
    // add here the members of your event process

   public:
    void InitProcess();
    void BeginOfEventProcess();
    TRestEvent* ProcessEvent(TRestEvent* eventInput);
    void EndOfEventProcess();
    void EndProcess();

    void LoadConfig(std::string cfgFilename, std::string name = "");

    void PrintMetadata() {
        BeginPrintProcess();

        std::cout << " XQRestTSPProcess: SetGap = " << fSetGap << std::endl;

        EndPrintProcess();
    }

    TString GetProcessName() { return (TString) "XQRestTSPProcess"; }

    // Constructor
    XQRestTSPProcess();
    XQRestTSPProcess(char* cfgFileName);
    // Destructor
    ~XQRestTSPProcess();

    ClassDef(XQRestTSPProcess, 1);  // Transform a TRestG4Event event to a
                                        // TRestHitsEvent (hits-collection event)
};
#endif
