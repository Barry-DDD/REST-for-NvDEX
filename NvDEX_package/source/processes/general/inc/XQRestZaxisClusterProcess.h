///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs Modified
///
///             XQRestZaxisClusterProcess.cxx

///             Date : Sep/2019
///             Author : Tao Li, SYSU
///
///_______________________________________________________________________________

#ifndef RestCore_XQRestZaxisClusterProcess
#define RestCore_XQRestZaxisClusterProcess

#include "TString.h"

#include <TRestHitsEvent.h>
#include <TRestGas.h>
#include "TRestEventProcess.h"
//just before TRestSignalToHitsProcess
class XQRestZaxisClusterProcess : public TRestEventProcess {
   private:
#ifndef __CINT__
    TRestHitsEvent* fInputHitsEvent;   //!
    TRestHitsEvent* fOutputHitsEvent;  //!

    TRestGas* fGas;          //!
#endif

    void InitFromConfigFile();

    void Initialize();

    void LoadDefaultConfig();

    Double_t fSetGap;
    Double_t fSampling;       // us
    Double_t fElectricField;  // V/cm
    Double_t fDriftVelocity;  // mm/us

   public:

    TVector3 Get3DMinBoundary( TRestHitsEvent* eventInput );
    TVector3 Get3DMaxBoundary( TRestHitsEvent* eventInput );

    void InitProcess();
    void BeginOfEventProcess();
    TRestEvent* ProcessEvent(TRestEvent* eventInput);
    void EndOfEventProcess();
    void EndProcess();

    void LoadConfig(std::string cfgFilename);

    TRestMetadata* GetProcessMetadata() { return NULL; }

    TString GetProcessName() { return (TString) "XQRestZaxisClusterProcess"; }

    // Constructor
    XQRestZaxisClusterProcess();
    XQRestZaxisClusterProcess(char* cfgFileName);
    // Destructor
    ~XQRestZaxisClusterProcess();

    ClassDef(XQRestZaxisClusterProcess,
             1);
};
#endif
