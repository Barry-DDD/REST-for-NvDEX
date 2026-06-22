///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs Modified
///
///             XQRestRasterisationProcess.cxx

///             Date : Sep/2019
///             Author : Tao Li, SYSU
///
///_______________________________________________________________________________

#ifndef RestCore_XQRestRasterisationProcess
#define RestCore_XQRestRasterisationProcess

#include "TString.h"

#include <TRestHitsEvent.h>
#include <TRestGas.h>
#include "TRestEventProcess.h"

class XQRestRasterisationProcess : public TRestEventProcess {
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
    Int_t fNumsGap;  // No 1 means analysisProcess
    Double_t fSampling;       // us
    Double_t fElectricField;  // V/cm
    Double_t fDriftVelocity;  // mm/us
    Int_t fModel;  // Simulation: 0 or Analysis: 1


    vector<Double_t> number_x;  //!
    vector<Double_t> number_y;  //!
    vector<Double_t> number_z;  //!
    vector<Double_t> number_e;  //!

   protected:
    // add here the members of your event process
    TVector2 fEnergyRange;

   public:
    void InitProcess();
    void BeginOfEventProcess();
    TRestEvent* ProcessEvent(TRestEvent* eventInput);
    void EndOfEventProcess();
    void EndProcess();

    void LoadConfig(std::string cfgFilename, std::string name = "");

    void PrintMetadata() {
        BeginPrintProcess();

        std::cout << " Electron Sampling rate : " << fSampling << " us" << std::endl;
        std::cout << " Readout Plane Gap : " << fSetGap << " mm" << std::endl;
        std::cout << " Readout Plane Gap nums : " << fNumsGap << std::endl;
        std::cout << " eField : " << fElectricField << " V/cm" << std::endl;
        metadata << "Drift velocity : " << fDriftVelocity << " mm/us" << endl;
        std::cout << " XQRestRasterisationProcess Model(1:Simulation;0:Analysis) : " << fModel << std::endl;
        std::cout << " All Deposited Energy Range : ( " << fEnergyRange.X() << ", " << fEnergyRange.Y() << " )" << std::endl;

        EndPrintProcess();
    }

    TString GetProcessName() { return (TString) "HitsEvent"; }

    // Constructor
    XQRestRasterisationProcess();
    XQRestRasterisationProcess(char* cfgFileName);
    // Destructor
    ~XQRestRasterisationProcess();

    ClassDef(XQRestRasterisationProcess, 1);  // Transform a TRestG4Event event to a
                                        // TRestHitsEvent (hits-collection event)
};
#endif
