///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs Modified
///
///             XQRestGaussianSharpeningProcess.cxx

///             Date : Sep/2019
///             Author : Tao Li, SYSU
///
///_______________________________________________________________________________

#ifndef RestCore_XQRestGaussianSharpeningProcess
#define RestCore_XQRestGaussianSharpeningProcess

#include <TRestHitsEvent.h>
#include <TRestGas.h>
#include "TRestEventProcess.h"

class XQRestGaussianSharpeningProcess : public TRestEventProcess {
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
    Double_t fThreEnergy;  // mm/us

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

        std::cout << " XQRestHitsConnectionProcess: sampling = " << fSampling << std::endl;
        std::cout << " XQRestHitsConnectionProcess: SetGap = " << fSetGap << std::endl;
        std::cout << " XQRestHitsConnectionProcess: ElectricField = " << fElectricField << std::endl;
        std::cout << " XQRestHitsConnectionProcess: DriftVelocity = " << fDriftVelocity << std::endl;

        EndPrintProcess();
    }

    TVector3 Get3DMinBoundary( TRestHitsEvent* eventInput );
    TVector3 Get3DMaxBoundary( TRestHitsEvent* eventInput );
    void GaussianSharpening( const TVector3 size,
                          const TVector3 boundary_min,
                          const TVector3 boundary_max,
                          TRestHitsEvent* eventInput,
                          TRestHitsEvent* eventOutput );


    TString GetProcessName() { return (TString) "XQRestGaussianSharpeningProcess"; }

    // Constructor
    XQRestGaussianSharpeningProcess();
    XQRestGaussianSharpeningProcess(char* cfgFileName);
    // Destructor
    ~XQRestGaussianSharpeningProcess();

    ClassDef(XQRestGaussianSharpeningProcess, 1);  // Transform a TRestG4Event event to a
                                        // TRestHitsEvent (hits-collection event)
};
#endif
