///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs Modified
///
///             XQRestHitsConnectionProcess.cxx

///             Date : Sep/2019
///             Author : Tao Li, SYSU
///
///_______________________________________________________________________________

#ifndef RestCore_XQRestHitsConnectionProcess
#define RestCore_XQRestHitsConnectionProcess

#include "TString.h"

#include <TRestHitsEvent.h>
#include <TRestGas.h>
#include "TRestEventProcess.h"

class XQRestHitsConnectionProcess : public TRestEventProcess {
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
    Double_t fSquareSize;
    Double_t fSampling;       // us
    Double_t fElectricField;  // V/cm
    Double_t fDriftVelocity;  // mm/us
    Double_t fMergeNGap;  // unit 1
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

        std::cout << " XQRestHitsConnectionProcess: SetGap = " << fSetGap << std::endl;

        EndPrintProcess();
    }

    void SpatialFiltering(const Int_t max_index,
                  const vector<Int_t> &path,
                  TRestHitsEvent* fInputHitsEvent,
                  TRestHitsEvent* fOutputHitsEvent);
    void dijkstra(const Int_t &beg,//出发点
                  const vector<vector<Double_t> > &adjmap,//邻接矩阵，通过传引用避免拷贝
                  vector<Double_t> &dist,//出发点到各点的最短路径长度
                  vector<Int_t> &path);//路径上到达该点的前一个点

    TString GetProcessName() { return (TString) "XQRestHitsConnectionProcess"; }

    // Constructor
    XQRestHitsConnectionProcess();
    XQRestHitsConnectionProcess(char* cfgFileName);
    // Destructor
    ~XQRestHitsConnectionProcess();

    ClassDef(XQRestHitsConnectionProcess, 1);  // Transform a TRestG4Event event to a
                                        // TRestHitsEvent (hits-collection event)
};
#endif
