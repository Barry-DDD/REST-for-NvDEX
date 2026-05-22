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

#include "XQRestHitsConnectionProcess.h"

using namespace std;

const double cmTomm = 10.;

ClassImp(XQRestHitsConnectionProcess)
    //______________________________________________________________________________
    XQRestHitsConnectionProcess::XQRestHitsConnectionProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestHitsConnectionProcess::XQRestHitsConnectionProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestHitsConnectionProcess::~XQRestHitsConnectionProcess() {
    delete fInputHitsEvent;

}

//______________________________________________________________________________
void XQRestHitsConnectionProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    cout << "Loading default values" << endl;

    fSetGap = 1;//mm
    fSquareSize = 800; //mm
    fSampling = 1;
    fElectricField = 1000;
    fMergeNGap = 5;
    fNAdjoin = 2;
}

//______________________________________________________________________________
void XQRestHitsConnectionProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;
}

void XQRestHitsConnectionProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestHitsConnectionProcess::InitProcess() {
    fGas = (TRestGas*)this->GetGasMetadata();
    if (fGas != NULL) {
        if (fDriftVelocity <= 0) fDriftVelocity = fGas->GetDriftVelocity(fElectricField) * cmTomm;
    } else {
        cout << "REST_WARNING. No TRestGas found in TRestRun." << endl;
    }
}

//______________________________________________________________________________
void XQRestHitsConnectionProcess::BeginOfEventProcess() { fInputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestHitsConnectionProcess::ProcessEvent(TRestEvent* evInput) {

    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);

    Double_t temp_dist = 0;
    vector<Int_t> final_path;
    Int_t max_index = 0;
    Int_t temp_max_index = 0;
    Int_t temp_beg = 0;

    Int_t beg = 0;
    vector<Double_t> dist;
    vector<Int_t> path;

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
        cout<< "XQRestHitsConnectionProcess : Error : Adjmap Matrix is zeros-matrix !"<<endl;
        cout<< "  Maybe restG4 Distance > SetGap * NAdjoin, Please Change your NAdjoin"<<endl;
        return NULL;
    }

    for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
        beg =i;
        vector<Double_t> dist;
        vector<Int_t> path;

        dijkstra(beg,adjmap,dist,path);

        std::vector<double>::iterator biggest = std::max_element(std::begin(dist), std::end(dist));
        max_index = std::distance(std::begin(dist), biggest);

        /*
        printf(
        " XQRestHitsConnectionProcess: "
        "fDriftVelocity %lf mm/us; fSampling %lf MHz\n",
        fDriftVelocity, 1.0 / fSampling );
        cout<<beg<<"到"<<max_index<<"的最短距离为"<<dist[max_index]<<"，反向打印路径：";
        for(int w=max_index; path[w]>=0; w=path[w])
            cout<<w<<"<-";
        cout<<beg<<'\n';
        */

        if( dist[max_index] > temp_dist ){
            temp_dist = dist[max_index];
            final_path = path;
            temp_max_index = max_index;
            temp_beg = beg;
        }
    }

    //空间滤波
    SpatialFiltering(temp_max_index, final_path, fInputHitsEvent, fOutputHitsEvent);

    if (this->GetVerboseLevel() >= REST_Debug) {
        printf(
        " XQRestHitsConnectionProcess: "
        "fDriftVelocity %lf mm/us; fSampling %lf MHz\n",
        fDriftVelocity, 1.0 / fSampling );
        cout<<beg<<"到"<<temp_max_index<<"的最短距离为"<<temp_dist<<"，反向打印路径：";
        for(int w=temp_max_index; final_path[w]>=0; w=final_path[w])
            cout<<w<<"<-";
        cout<<beg<<'\n';
    }

    Int_t initialHits = fInputHitsEvent->GetNumberOfHits();
    Int_t finalHits = fOutputHitsEvent->GetNumberOfHits();
    if(finalHits < 10)
      return NULL;
    cout << "XQRestHitsConnectionProcess : Initial number of hits : " << initialHits << endl;
    cout << "XQRestHitsConnectionProcess : Final number of hits : " << finalHits << endl;

    return fOutputHitsEvent;
}

//______________________________________________________________________________
void XQRestHitsConnectionProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestHitsConnectionProcess::EndProcess() {}

//______________________________________________________________________________
//TODO::堆优化
void XQRestHitsConnectionProcess::dijkstra(const Int_t &beg,//出发点
                  const vector<vector<Double_t> > &adjmap,//邻接矩阵，通过传引用避免拷贝
                  vector<Double_t> &dist,//出发点到各点的最短路径长度
                  vector<Int_t> &path)//路径上到达该点的前一个点
//负边被认作不联通
//福利：这个函数没有用任何全局量，可以直接复制！
{
    const int &NODE=adjmap.size();//用邻接矩阵的大小传递顶点个数，减少参数传递
    dist.assign(NODE,-1);//初始化距离为未知
    path.assign(NODE,-1);//初始化路径为未知
    vector<bool> flag(NODE,0);//标志数组，判断是否处理过
    dist[beg]=0;//出发点到自身路径长度为0
    while(1)
    {
        int v=-1;//初始化为未知
        for(int i=0; i!=NODE; ++i)
            if(!flag[i]&&dist[i]>=0)//寻找未被处理过且
                if(v<0||dist[i]<dist[v])//距离最小的点
                    v=i;
        if(v<0)return;//所有联通的点都被处理过
        flag[v]=1;//标记
        for(int i=0; i!=NODE; ++i)
            if(adjmap[v][i]>=0)//有联通路径且
                if(dist[i]<0||dist[v]+adjmap[v][i]<dist[i])//不满足三角不等式
                {
                    dist[i]=dist[v]+adjmap[v][i];//更新
                    path[i]=v;//记录路径
                }
    }
}

//______________________________________________________________________________

void XQRestHitsConnectionProcess::SpatialFiltering(const Int_t max_index,
                const vector<Int_t> &path,
                TRestHitsEvent* fInputHitsEvent,
                TRestHitsEvent* fOutputHitsEvent)
{


    vector<Double_t> number_x;
    vector<Double_t> number_y;
    vector<Double_t> number_z;
    vector<Double_t> number_e;

    for(int w=max_index; path[w]>=0; w=path[w]){
        Double_t e = 0;
        Double_t x = 0;
        Double_t y = 0;
        Double_t z = 0;
        bool merge = false;
        for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
            if( w != i
              && TMath::Abs( fInputHitsEvent->GetX(w) - fInputHitsEvent->GetX(i) ) <= fSetGap * fMergeNGap
              && TMath::Abs( fInputHitsEvent->GetY(w) - fInputHitsEvent->GetY(i) ) <= fSetGap * fMergeNGap
              && TMath::Abs( fInputHitsEvent->GetZ(w) - fInputHitsEvent->GetZ(i) ) <= fSetGap * fMergeNGap){
              //&& TMath::Abs( fInputHitsEvent->GetZ(w) - fInputHitsEvent->GetZ(i) ) <= fSampling * fDriftVelocity * 1.1 * fMergeNGap){
                  merge = true;
                  e += fInputHitsEvent->GetEnergy(i);
                  x += fInputHitsEvent->GetEnergy(i) * fInputHitsEvent->GetX(i);
                  y += fInputHitsEvent->GetEnergy(i) * fInputHitsEvent->GetY(i);
                  z += fInputHitsEvent->GetEnergy(i) * fInputHitsEvent->GetZ(i);
              }
        }
        merge = false;

        if(merge == true){
            merge = false;
            e += fInputHitsEvent->GetEnergy(w);
            x += fInputHitsEvent->GetX(w) * fInputHitsEvent->GetEnergy(w);
            y += fInputHitsEvent->GetY(w) * fInputHitsEvent->GetEnergy(w);
            z += fInputHitsEvent->GetZ(w) * fInputHitsEvent->GetEnergy(w);
            x /= e;
            y /= e;
            z /= e;
        }else{
            e = fInputHitsEvent->GetEnergy(w);
            x = fInputHitsEvent->GetX(w);
            y = fInputHitsEvent->GetY(w);
            z = fInputHitsEvent->GetZ(w);
        }
        number_x.push_back( x );
        number_y.push_back( y );
        number_z.push_back( z );
        number_e.push_back( e );
    }

    //merge hits
    for (int i = 0; i < number_x.size(); i++) {
        Double_t x = number_x[i];
        Double_t y = number_y[i];
        Double_t z = number_z[i];
        Double_t e = number_e[i];
        for (int j = i +1 ; j < number_x.size(); j++) {
            //cout<< abs(x - number_x[j])<<" "<< abs(y - number_y[j])<<" "<< abs(z - number_z[j])<<" "<<endl;
            if( (abs(x - number_x[j]) < 0.01) && (abs(y - number_y[j]) < 0.01) && (abs(z - number_z[j]) < 0.01) ){
                e += number_e[j];
                number_x.erase( std::begin( number_x) + j );
                number_y.erase( std::begin( number_y) + j );
                number_z.erase( std::begin( number_z) + j );
                number_e.erase( std::begin( number_e) + j );
                j--;
            }
        }
        fOutputHitsEvent->AddHit(x, y, z, e);
    }
}


//______________________________________________________________________________
void XQRestHitsConnectionProcess::InitFromConfigFile() {
    fSampling = GetDblParameterWithUnits("sampling");
    fSetGap = GetDblParameterWithUnits("SetGap");
    fElectricField = GetDblParameterWithUnits("ElectricField");
    fDriftVelocity = StringToDouble(GetParameter("DriftVelocity", "0")) * cmTomm;
    fMergeNGap = StringToInteger(GetParameter("MergeNGap", "5"));
    fNAdjoin = StringToInteger(GetParameter("NAdjoin", "2"));
}
