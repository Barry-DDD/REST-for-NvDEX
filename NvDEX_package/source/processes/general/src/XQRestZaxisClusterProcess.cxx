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

#include "XQRestZaxisClusterProcess.h"
using namespace std;

const double cmTomm = 10.;

ClassImp(XQRestZaxisClusterProcess)
    //______________________________________________________________________________
    XQRestZaxisClusterProcess::XQRestZaxisClusterProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestZaxisClusterProcess::XQRestZaxisClusterProcess(char* cfgFileName) {
    Initialize();
    if (LoadConfigFromFile(cfgFileName) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestZaxisClusterProcess::~XQRestZaxisClusterProcess() {
    delete fOutputHitsEvent;
    delete fInputHitsEvent;
}

void XQRestZaxisClusterProcess::LoadConfig(string cfgFilename) {
    if (LoadConfigFromFile(cfgFilename)) LoadDefaultConfig();
}

void XQRestZaxisClusterProcess::LoadDefaultConfig() {
  SetTitle("Default config");

  cout << "Loading default values" << endl;

  fSetGap = 1;//mm
  fSampling = 1;
  fElectricField = 1000;
}

//______________________________________________________________________________
void XQRestZaxisClusterProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fInputEvent = fInputHitsEvent;
    fOutputEvent = fOutputHitsEvent;
}

//______________________________________________________________________________
void XQRestZaxisClusterProcess::InitProcess() {
    fGas = (TRestGas*)this->GetGasMetadata();
    if (fGas != NULL) {
        if (fDriftVelocity <= 0) fDriftVelocity = fGas->GetDriftVelocity(fElectricField) * cmTomm;
    } else {
        cout << "REST_WARNING. No TRestGas found in TRestRun." << endl;
    }
}

//______________________________________________________________________________
void XQRestZaxisClusterProcess::BeginOfEventProcess() { fOutputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestZaxisClusterProcess::ProcessEvent(TRestEvent* evInput) {
    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);

    TVector3 boundary_min = Get3DMinBoundary(fInputHitsEvent);
    TVector3 boundary_max = Get3DMaxBoundary(fInputHitsEvent);
    Int_t size_x = (Int_t)( ( boundary_max.X() - boundary_min.X() ) / fSetGap ) + 1 ;
    Int_t size_y = (Int_t)( ( boundary_max.Y() - boundary_min.Y() ) / fSetGap ) + 1 ;
    Int_t size_z = (Int_t)( ( boundary_max.Z() - boundary_min.Z() ) / (fSampling * fDriftVelocity) ) + 1 ;
    TVector3 size(size_x, size_y, size_z);


    /******************************set 3D matrix******************************/
    double ***graph3D;//声明指针
    graph3D = (double***)malloc(sizeof(double**)*size.X());//X的长度
    for (int x = 0; x < size.X(); x++)
    {
        graph3D[x] = (double **)malloc(sizeof(double*)*size.Y());//Y的长度
        for (int y = 0;y < size.Y();y++)
        {
            graph3D[x][y] = (double *)malloc(sizeof(double)*size.Z());//Z的长度
        }
    }

    for (int i = 0;i < size.X(); i++)
    {
        for (int j = 0; j < size.Y(); j++)
        {
            for (int k = 0; k < size.Z(); k++)
            {
                graph3D[i][j][k] =  0;
            }
        }
    }
    for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
        Int_t x = (Int_t)( ( fInputHitsEvent->GetX(i) - boundary_min.X() ) / fSetGap );
        Int_t y = (Int_t)( ( fInputHitsEvent->GetY(i) - boundary_min.Y() ) / fSetGap );
        Int_t z = (Int_t)( ( fInputHitsEvent->GetZ(i) - boundary_min.Z() ) / (fSampling * fDriftVelocity) );
        graph3D[x][y][z] =  fInputHitsEvent->GetEnergy(i);
    }
    /*************************************************************************/


    /******************************ZaxisCluster******************************/
    for (int i = 0;i < size.X(); i++)
    {
      cout << "xxxxxxxxxxxx: " << fSetGap * i + boundary_min.X() << endl;
        for (int j = 0; j < size.Y(); j++)
        {
            vector<Int_t> temp_break; //break point
            bool break_flag = false;
            //find break point
            for (int k = 0; k < size.Z() - 1; k++)
            {
                if ( graph3D[i][j][k] > 0.0001
                      && break_flag == false ){
                    temp_break.push_back( k );
                    break_flag = true;
                }
                if ( graph3D[i][j][k] < 0.0001
                      && graph3D[i][j][k+1] < 0.0001
                      && break_flag == true ){
                    temp_break.push_back( k );
                    break_flag = false;
                }
            }
            bool xq = false;
            for (int k = 0; k < size.Z(); k++)
                if( graph3D[i][j][k] != 0 ){
                    cout << k << "  ";
                    xq = true;
                }
            if( xq )
              cout << endl;

            if( temp_break.size() % 2  == 1 )
                temp_break.push_back( size.Z() - 1 );
            if( temp_break.size() % 2  == 1 ){
                cout<< "XQRestZaxisClusterProcess: ("
                  << i << "," << j << "). logic error!" <<endl;
                continue;
            }

            //wave peak
            vector<Int_t> subZ;
            for (int m = 0; m < temp_break.size() / 2; m ++)
            {
                if( temp_break[m*2+1] - temp_break[m*2] < 4){
                  subZ.push_back( temp_break[m*2] );
                    subZ.push_back( temp_break[m*2 + 1] );
                    continue;
                }
                subZ.push_back( temp_break[m*2] );
                for (int k = temp_break[m*2]; k < temp_break[m*2] - 2; k++){
                    if( graph3D[i][j][k] > graph3D[i][j][k+1]
                          && graph3D[i][j][k+1] > graph3D[i][j][k+2]
                          && graph3D[i][j][k+2] < graph3D[i][j][k+3]
                          && graph3D[i][j][k+3] < graph3D[i][j][k+4]){
                        subZ.push_back( k+1 );
                        subZ.push_back( k+1 );
                    }
                }
                subZ.push_back( temp_break[m*2 + 1] );
            }

            xq = false;
            for (int k = 0; k < subZ.size(); k++){
                    cout << "subZ:" << subZ[k] << "  ";
                    xq = true;
                }
            if( xq )
              cout << endl;


            //weight mean
            if( subZ.size() % 2  == 1 ){
                cout<< "XQRestZaxisClusterProcess: logic error!" <<endl;
                return NULL;
            }
            for (int m = 0; m < subZ.size() / 2; m ++){
                Double_t e = 0;
                Double_t x = fSetGap * i + boundary_min.X();
                Double_t y = fSetGap * j + boundary_min.Y();
                Double_t z = 0;
                for (int k = subZ[m*2]; k < subZ[m*2 +1]; k++){
                    e += graph3D[i][j][k];
                    z += ( (fSampling * fDriftVelocity) * k + boundary_min.Z() ) * graph3D[i][j][k];
                }
                fOutputHitsEvent->AddHit(x, y, z/e, e);
            }

        }
    }
    /*************************************************************************/
    cout << boundary_min.X() << " " << boundary_max.X() << "  " << fSetGap <<endl;
    cout << boundary_min.Y() << " " << boundary_max.Y() << "  " << fSetGap <<endl;
    cout << boundary_min.Z() << " " << boundary_max.Z() << "  " << fSampling * fDriftVelocity <<endl;
    cout<< size_x<< "  "<< size_y <<" "<<size_z <<endl;
    cout << "XQRestZaxisClusterProcess : Initial number of hits : " << fInputHitsEvent->GetNumberOfHits() << endl;
    cout << "XQRestZaxisClusterProcess : Final number of hits : " << fOutputHitsEvent->GetNumberOfHits() << endl;

    return fOutputHitsEvent;
}

//______________________________________________________________________________
void XQRestZaxisClusterProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestZaxisClusterProcess::EndProcess() {}

//______________________________________________________________________________
TVector3 XQRestZaxisClusterProcess::Get3DMaxBoundary( TRestHitsEvent* eventInput ){
    Double_t max_x=-1e10,max_y=-1e10,max_z=-1e10;
    for (int i = 0; i < eventInput->GetNumberOfHits(); i++) {
            if( eventInput->GetX(i) > max_x )
                max_x = eventInput->GetX(i);
            if( eventInput->GetY(i) > max_y )
                max_y = eventInput->GetY(i);
            if( eventInput->GetZ(i) > max_z )
                max_z = eventInput->GetZ(i);
    }
    return TVector3(max_x, max_y, max_z);
}

//______________________________________________________________________________
TVector3 XQRestZaxisClusterProcess::Get3DMinBoundary( TRestHitsEvent* eventInput ){
  Double_t min_x=1e10,min_y=1e10,min_z=1e10;
  for (int i = 0; i < eventInput->GetNumberOfHits(); i++) {
          if( eventInput->GetX(i) < min_x )
              min_x = eventInput->GetX(i);
          if( eventInput->GetY(i) < min_y )
              min_y = eventInput->GetY(i);
          if( eventInput->GetZ(i) < min_z )
              min_z = eventInput->GetZ(i);
  }
  return TVector3(min_x, min_y, min_z);
}

//______________________________________________________________________________
void XQRestZaxisClusterProcess::InitFromConfigFile() {
    fSampling = GetDblParameterWithUnits("sampling");
    fSetGap = GetDblParameterWithUnits("SetGap");
    fElectricField = GetDblParameterWithUnits("ElectricField");
    fDriftVelocity = StringToDouble(GetParameter("DriftVelocity", "0")) * cmTomm;
}
