///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs Modified
///
///             XQRestMainTrackProcess.cxx

///             Date : Sep/2019
///             Author : Tao Li, SYSU
///
///_______________________________________________________________________________

#include "XQRestMainTrackProcess.h"
#include<queue>
using namespace std;

const double cmTomm = 10.;

ClassImp(XQRestMainTrackProcess)
    //______________________________________________________________________________
    XQRestMainTrackProcess::XQRestMainTrackProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestMainTrackProcess::XQRestMainTrackProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestMainTrackProcess::~XQRestMainTrackProcess() {
    delete fInputHitsEvent;

}

//______________________________________________________________________________
void XQRestMainTrackProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    cout << "Loading default values" << endl;

    fSetGap = 1;//mm
    fNAdjoin = 2;
}

//______________________________________________________________________________
void XQRestMainTrackProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;
}

void XQRestMainTrackProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestMainTrackProcess::InitProcess() {}

//______________________________________________________________________________
void XQRestMainTrackProcess::BeginOfEventProcess() { fInputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestMainTrackProcess::ProcessEvent(TRestEvent* evInput) {
    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);

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
        cout<< "XQRestMainTrackProcess : Error : Adjmap Matrix is zeros-matrix !"<<endl;
        cout<< "  Maybe restG4 Distance > SetGap * NAdjoin, Please Change your NAdjoin"<<endl;
        return NULL;
    }

    //bfs方法
    const int &NODE=adjmap.size();//用邻接矩阵的大小传递顶点个数，减少参数传递
    vector<int> flag(NODE,0);//标志数组，判断是否处理过
    int cluster_num = 1;//表明类
    while( isOver(flag) ){
        queue<int> pt;
        for( int i = 0; i < NODE; i++)
            if(flag[i] == 0){
                pt.push(i); flag[i] = cluster_num;
                //cout<< "起始点："<< i << endl;
                break;
        }

        while( !pt.empty() ){
            int nowNode = pt.front();
            pt.pop();
            for( int i = 0; i < NODE; i++){
                if( adjmap[nowNode][i] > 0 && !flag[i]){
                    pt.push(i);
                    flag[i] = cluster_num;
                    //cout << "序号：" << i << ";属于类别：" << cluster_num <<endl;
                }
            }
        }

        int xq = 0;
        for(int i=0; i < flag.size(); i++){
            if(flag[i] == cluster_num) xq++;
        }
        //cout << "类别:"<<cluster_num << " 合计:"<<xq <<endl;

        cluster_num ++;
    }

    //统计各类沉积能量
    vector<float> energy(cluster_num,0);
    for( int i = 0; i < NODE; i++)
        energy[flag[i]] += fInputHitsEvent->GetEnergy(i);
    int maxEnergy_index = max_element(energy.begin(),energy.end()) - energy.begin();
    //cout << "选取类别:"<<maxEnergy_index << endl;

    for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
        if(flag[i] == maxEnergy_index)
            fOutputHitsEvent->AddHit( fInputHitsEvent->GetX(i),  fInputHitsEvent->GetY(i),  fInputHitsEvent->GetZ(i),  fInputHitsEvent->GetEnergy(i));
    }
    cout << "XQRestMainTrackProcess : Initial number of hits : " << fInputHitsEvent->GetNumberOfHits() << endl;
    cout << "XQRestMainTrackProcess : Final number of hits : " << fOutputHitsEvent->GetNumberOfHits() << endl;
    return fOutputHitsEvent;
}

//______________________________________________________________________________
bool XQRestMainTrackProcess::isOver(const vector<int> &flag) {
    for(int i=0; i < flag.size(); i++){
        if(flag[i] == 0 ) return true;
    }
    return false;
}

//______________________________________________________________________________
void XQRestMainTrackProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestMainTrackProcess::EndProcess() {}

//______________________________________________________________________________
void XQRestMainTrackProcess::InitFromConfigFile() {
    fSetGap = GetDblParameterWithUnits("SetGap");
    fNAdjoin = StringToInteger(GetParameter("NAdjoin", "2"));
}
