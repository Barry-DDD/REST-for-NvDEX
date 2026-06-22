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

#ifndef RestCore_TRestElectronDiffusionProcess
#define RestCore_TRestElectronDiffusionProcess

#include "TString.h"

#include <TRandom3.h>
#include <TRestGas.h>
#include <TRestHitsEvent.h>
#include <TRestReadout.h>

#include "TRestEventProcess.h"

class TRestElectronDiffusionProcess : public TRestEventProcess {
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
    Double_t fElectricField;
    Double_t fAttachment;
    Double_t fGasPressure;
    Double_t fWvalue;
    Double_t fLonglDiffCoeff;
    Double_t fTransDiffCoeff;

    Int_t fMaxHits;

    Double_t settingCoeff;
    
   public:
    void InitProcess();
    void BeginOfEventProcess();
    TRestEvent* ProcessEvent(TRestEvent* eventInput);
    void EndOfEventProcess();
    void EndProcess();

    void LoadConfig(std::string cfgFilename, std::string name = "");

    void PrintMetadata() {
        BeginPrintProcess();

        metadata << " eField : " << fElectricField << " V/cm" << endl;
        metadata << " attachment coeficient : " << fAttachment << "" << endl;
        metadata << " gas pressure : " << fGasPressure << " atm" << endl;
        metadata << " setting transversal diffusion coefficient : " << settingCoeff << " cm^1/2" << endl;
        metadata << " longitudinal diffusion coefficient : " << fLonglDiffCoeff << " cm^1/2" << endl;
        metadata << " drift distance 780 mm, the longitudinal sigma : " << 10. * TMath::Sqrt(780 / 10.) * fLonglDiffCoeff << " mm" << endl;
        metadata << " transversal diffusion coefficient : " << fTransDiffCoeff << " cm^1/2" << endl;
        metadata << " drift distance 780 mm, the transversal sigma : " << 10. * TMath::Sqrt(780 / 10.) * fTransDiffCoeff << " mm" << endl;
        metadata << " W value : " << fWvalue << " eV" << endl;

        metadata << " Maximum number of hits : " << fMaxHits << endl;

        EndPrintProcess();
    }

    TRestMetadata* GetProcessMetadata() { return fGas; }

    TString GetProcessName() { return (TString) "electronDiffusion"; }

    Double_t GetElectricField() { return fElectricField; }
    Double_t GetAttachmentCoefficient() { return fAttachment; }
    Double_t GetGasPressure() { return fGasPressure; }

    // Constructor
    TRestElectronDiffusionProcess();
    TRestElectronDiffusionProcess(char* cfgFileName);
    // Destructor
    ~TRestElectronDiffusionProcess();

    Double_t SetBoundaries(TRestHitsEvent* evInput);

    ClassDef(TRestElectronDiffusionProcess,
             1);  // Template for a REST "event process" class inherited from
                  // TRestEventProcess
};
#endif
