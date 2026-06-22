///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestElectronDiffusionProcess.h
///
///_______________________________________________________________________________

#ifndef RestCore_TRestIonDiffusionProcess
#define RestCore_TRestIonDiffusionProcess

#include "TString.h"

#include <TRandom3.h>
#include <TRestGas.h>
#include <TRestHitsEvent.h>
#include <TRestReadout.h>

#include "TRestEventProcess.h"

class TRestIonDiffusionProcess : public TRestEventProcess {
   private:
#ifndef __CINT__
    TRestHitsEvent* fInputHitsEvent;   //!
    TRestHitsEvent* fOutputHitsEvent;  //!

    TRestGas* fGas;          //!
    TRestReadout* fReadout;  //!

    TRandom3* fRandom;  //!
#endif

    void InitFromConfigFile();

    void Initialize();

    void LoadDefaultConfig();

   protected:
    Double_t fElectricField;  // in V/mm
    Double_t fAttachment;     // in mm
    Double_t fGasPressure;    // in atm
    Double_t fWvalue;
    Double_t fLonglDiffCoeff;  // in mm^0.5
    Double_t fTransDiffCoeff;  // in mm^0.5
    Bool_t fPoissonElectronExcitation;
    Bool_t fUnitElectronEnergy;

    Int_t fMaxHits;

    Double_t settingCoeff;

    Double_t fSeed = 0;

   public:
    void InitProcess();
    TRestEvent* ProcessEvent(TRestEvent* eventInput);
    void EndProcess();

    void LoadConfig(std::string cfgFilename, std::string name = "");

    void PrintMetadata() {
        BeginPrintProcess();

        metadata << " eField : " << fElectricField << " V/mm" << endl;
        metadata << " max drift distance : " << "1690" << " mm" << endl;
        metadata << " gas pressure : " << fGasPressure << " atm" << endl;
        metadata << " ionization_energy value : " << "32" << " eV" << endl;

        // metadata << " Maximum number of hits : " << fMaxHits << endl;

        metadata << " seed : " << fSeed << endl;

        EndPrintProcess();
    }

    TString GetProcessName() { return (TString) "IonDiffusion"; }

    Double_t GetElectricField() { return fElectricField; }
    Double_t GetAttachmentCoefficient() { return fAttachment; }
    Double_t GetGasPressure() { return fGasPressure; }

    // Constructor
    TRestIonDiffusionProcess();
    TRestIonDiffusionProcess(char* cfgFileName);
    // Destructor
    ~TRestIonDiffusionProcess();

    ClassDef(TRestIonDiffusionProcess, 1);  // Template for a REST "event process" class inherited from
                                            // TRestEventProcess
};
#endif
