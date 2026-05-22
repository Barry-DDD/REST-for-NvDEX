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

#include "XQRestTSPProcess.h"
#include<queue>
using namespace std;

const double cmTomm = 10.;

ClassImp(XQRestTSPProcess)
    //______________________________________________________________________________
    XQRestTSPProcess::XQRestTSPProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestTSPProcess::XQRestTSPProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestTSPProcess::~XQRestTSPProcess() {
    delete fInputHitsEvent;

}

//______________________________________________________________________________
void XQRestTSPProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    cout << "Loading default values" << endl;

    fSetGap = 1;//mm
    fNAdjoin = 2;
}

//______________________________________________________________________________
void XQRestTSPProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;
}

void XQRestTSPProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestTSPProcess::InitProcess() {}

//______________________________________________________________________________
void XQRestTSPProcess::BeginOfEventProcess() { fInputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestTSPProcess::ProcessEvent(TRestEvent* evInput) {
    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);
    /*
    //获取邻接矩阵
    bool adjmapZero_flag = false; //adjmap can`t be zeros
    vector<vector<Double_t> > adjmap(fInputHitsEvent->GetNumberOfHits(), vector<Double_t>(fInputHitsEvent->GetNumberOfHits(),-1));
    for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
        for (int j = 0; j < fInputHitsEvent->GetNumberOfHits(); j++) {
            if( i != j
              && TMath::Abs( fInputHitsEvent->GetX(i) - fInputHitsEvent->GetX(j) ) <= fSetGap * fNAdjoin
              && TMath::Abs( fInputHitsEvent->GetY(i) - fInputHitsEvent->GetY(j) ) <= fSetGap * fNAdjoin
              && TMath::Abs( fInputHitsEvent->GetZ(i) - fInputHitsEvent->GetZ(j) ) <= fSetGap * fNAdjoin){
              //&& TMath::Abs( fInputHitsEvent->GetZ(i) - fInputHitsEvent->GetZ(j) ) <= fSampling * fDriftVelocity * 1.1 * fNAdjoin){
                    adjmap[i][j] = TMath::Sqrt( fInputHitsEvent->GetDistance2(i,j) );
                    adjmapZero_flag = true;
                }
        }
    }
    if(adjmapZero_flag == false){
        cout<< "XQRestTSPProcess : Error : Adjmap Matrix is zeros-matrix !"<<endl;
        cout<< "  Maybe restG4 Distance > SetGap * NAdjoin, Please Change your NAdjoin"<<endl;
        return NULL;
    }








    for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
        if(flag[i] == maxEnergy_index)
            fOutputHitsEvent->AddHit( fInputHitsEvent->GetX(i),  fInputHitsEvent->GetY(i),  fInputHitsEvent->GetZ(i),  fInputHitsEvent->GetEnergy(i));
    }
    */
    cout << "XQRestTSPProcess : Initial number of hits : " << fInputHitsEvent->GetNumberOfHits() << endl;
    cout << "XQRestTSPProcess : Final number of hits : " << fOutputHitsEvent->GetNumberOfHits() << endl;
    return fOutputHitsEvent;
}

//______________________________________________________________________________
void XQRestTSPProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestTSPProcess::EndProcess() {}

//______________________________________________________________________________
void XQRestTSPProcess::InitFromConfigFile() {
    fSetGap = GetDblParameterWithUnits("SetGap");
    fNAdjoin = StringToInteger(GetParameter("NAdjoin", "2"));
}
