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

#include "XQRestGaussianSharpeningProcess.h"

using namespace std;

const double cmTomm = 10.;

ClassImp(XQRestGaussianSharpeningProcess)
    //______________________________________________________________________________
    XQRestGaussianSharpeningProcess::XQRestGaussianSharpeningProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestGaussianSharpeningProcess::XQRestGaussianSharpeningProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestGaussianSharpeningProcess::~XQRestGaussianSharpeningProcess() {
    delete fInputHitsEvent;

}

//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    cout << "Loading default values" << endl;

    fSetGap = 1;//mm
    fSampling = 1;
    fElectricField = 1000;
    fThreEnergy = 0;
}

//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;
}

void XQRestGaussianSharpeningProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::InitProcess() {
    fGas = (TRestGas*)this->GetGasMetadata();
    if (fGas != NULL) {
        if (fDriftVelocity <= 0) fDriftVelocity = fGas->GetDriftVelocity(fElectricField) * cmTomm;
    } else {
        cout << "REST_WARNING. No TRestGas found in TRestRun." << endl;
    }
}

//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::BeginOfEventProcess() { fInputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestGaussianSharpeningProcess::ProcessEvent(TRestEvent* evInput) {

    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);

    TVector3 boundary_min = Get3DMinBoundary(fInputHitsEvent);
    TVector3 boundary_max = Get3DMaxBoundary(fInputHitsEvent);
    Int_t size_x = (Int_t)( ( boundary_max.X() - boundary_min.X() ) / fSetGap ) + 1 ;
    Int_t size_y = (Int_t)( ( boundary_max.Y() - boundary_min.Y() ) / fSetGap ) + 1 ;
    Int_t size_z = (Int_t)( ( boundary_max.Z() - boundary_min.Z() ) / (fSampling * fDriftVelocity) ) + 1 ;
    TVector3 size(size_x, size_y, size_z);

    GaussianSharpening(size, boundary_min, boundary_max, fInputHitsEvent, fOutputHitsEvent);

    /*
    for (int i = 0; i < fInputHitsEvent->GetNumberOfHits(); i++) {
        Double_t x = fInputHitsEvent->GetX(i);
        Double_t y = fInputHitsEvent->GetY(i);
        Double_t z = fInputHitsEvent->GetZ(i);
        Double_t e = fInputHitsEvent->GetEnergy(i);
        if(e > fThreEnergy)
          fOutputHitsEvent->AddHit(x, y, z, e);
    }
    */

    cout << boundary_min.X() << " " << boundary_max.X() << "  " << fSetGap <<endl;
    cout << boundary_min.Y() << " " << boundary_max.Y() << "  " << fSetGap <<endl;
    cout << boundary_min.Z() << " " << boundary_max.Z() << "  " << fSampling * fDriftVelocity <<endl;
    cout<< size_x<< "  "<< size_y <<" "<<size_z <<endl;
    Int_t initialHits = fInputHitsEvent->GetNumberOfHits();
    cout << "XQRestGaussianSharpeningProcess : Initial number of hits : " << initialHits << endl;
    Int_t finalHits = fOutputHitsEvent->GetNumberOfHits();
    cout << "XQRestGaussianSharpeningProcess : Final number of hits : " << finalHits << endl;
    cout << "XQRestGaussianSharpeningProcess : xBoundary : (" << boundary_min.X() << " "
         << "," << boundary_max.X() << ");" << endl;
    cout << "XQRestGaussianSharpeningProcess : yBoundary : (" << boundary_min.Y() << " "
         << "," << boundary_max.Y() << ");" << endl;
    cout << "XQRestGaussianSharpeningProcess : zBoundary : (" << boundary_min.Z() << " "
         << "," << boundary_max.Z() << ");" << endl;
    return fOutputHitsEvent;
}

//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::EndProcess() {}

//______________________________________________________________________________
TVector3 XQRestGaussianSharpeningProcess::Get3DMaxBoundary( TRestHitsEvent* eventInput ){
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
TVector3 XQRestGaussianSharpeningProcess::Get3DMinBoundary( TRestHitsEvent* eventInput ){
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
void XQRestGaussianSharpeningProcess::GaussianSharpening(const TVector3 size,
                                                        const TVector3 boundary_min,
                                                        const TVector3 boundary_max,
                                                        TRestHitsEvent* eventInput,
                                                        TRestHitsEvent* eventOutput )
{
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

    for (int i = 0; i < eventInput->GetNumberOfHits(); i++) {
        Int_t x = (Int_t)( ( eventInput->GetX(i) - boundary_min.X() ) / fSetGap );
        Int_t y = (Int_t)( ( eventInput->GetY(i) - boundary_min.Y() ) / fSetGap );
        Int_t z = (Int_t)( ( eventInput->GetZ(i) - boundary_min.Z() ) / (fSampling * fDriftVelocity) );
        graph3D[x][y][z] =  eventInput->GetEnergy(i);
    }

    for (int i = 0;i < size.X(); i++)
    {
        for (int j = 0; j < size.Y(); j++)
        {
            for (int k = 0; k < size.Z(); k++)
            {
                eventOutput->AddHit(i, j, k, graph3D[i][j][k]);
            }
        }
    }


    //free
    for (int x = 0; x < size.X(); x++)
    {
        for (int y = 0; y < size.Y(); y++)
        {
            free(graph3D[x][y]);//释放Z
        }
    }
    for (int x = 0; x < size.X(); x++)
    {
        free(graph3D[x]);//释放Y
    }
    free(graph3D);//释放X

}
//______________________________________________________________________________
void XQRestGaussianSharpeningProcess::InitFromConfigFile() {
    fSampling = GetDblParameterWithUnits("sampling");
    fSetGap = GetDblParameterWithUnits("SetGap");
    fElectricField = GetDblParameterWithUnits("ElectricField");
    fDriftVelocity = StringToDouble(GetParameter("DriftVelocity", "0")) * cmTomm;
    fThreEnergy = StringToDouble(GetParameter("ThreEnergy", "0"));
}
