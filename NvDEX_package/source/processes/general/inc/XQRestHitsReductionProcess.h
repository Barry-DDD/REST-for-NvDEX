//////////////////////////////////////////////////////////////////////////
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             XQRestHitsReductionProcess.h
///
///             Jan 2016:   First concept (Javier Galan)
///             Modified by LiTao SYSU
///
//////////////////////////////////////////////////////////////////////////

#ifndef RestCore_XQRestHitsReductionProcess
#define RestCore_XQRestHitsReductionProcess

#include <TRestHitsEvent.h>
#include "TRestEventProcess.h"

class XQRestHitsReductionProcess : public TRestEventProcess {
   private:
#ifndef __CINT__
    TRestHitsEvent* fInputHitsEvent;   //!
    TRestHitsEvent* fOutputHitsEvent;  //!
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

        std::cout << " Starting distance : " << fStartingDistance << std::endl;
        std::cout << " Minimum distance : " << fMinimumDistance << std::endl;
        std::cout << " Distance step factor : " << fDistanceFactor << std::endl;
        std::cout << " Minimum number of nodes : " << fStartingNodes << std::endl;
        std::cout << " Maximum number of nodes : " << fMaxNodes << std::endl;
        std::cout << " Single Hit Energy Cut : " << fHitsEnergy << " keV. " << std::endl;
        std::cout << " All Deposited Energy Range : ( " << fEnergyRange.X() << ", " << fEnergyRange.Y() << " )" << std::endl;

        EndPrintProcess();
    }

    TString GetProcessName() { return (TString) "XQRestHitsReductionProcess"; }

    //void Reducting( TRestHitsEvent* temp );

    // Constructor
    XQRestHitsReductionProcess();
    XQRestHitsReductionProcess(char* cfgFileName);
    // Destructor
    ~XQRestHitsReductionProcess();

    ClassDef(XQRestHitsReductionProcess,
             1);  // Template for a REST "event process" class inherited from
                  // TRestEventProcess
};
#endif
