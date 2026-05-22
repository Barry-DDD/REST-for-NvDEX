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

#include "XQRestRasterisationProcess.h"

using namespace std;

const double cmTomm = 10.;

ClassImp(XQRestRasterisationProcess)
    //______________________________________________________________________________
    XQRestRasterisationProcess::XQRestRasterisationProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestRasterisationProcess::XQRestRasterisationProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestRasterisationProcess::~XQRestRasterisationProcess() {
    delete fInputHitsEvent;
    number_x.clear();
    number_y.clear();
    number_z.clear();
    number_e.clear();

}

//______________________________________________________________________________
void XQRestRasterisationProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    cout << "Loading default values" << endl;

    fSetGap = 1;//mm
    fNumsGap = 1; //mm
    fSampling = 1;
    fElectricField = 1000;
    fModel = 1;
    fEnergyRange.Set(2395.0,2520.0);
}

//______________________________________________________________________________
void XQRestRasterisationProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;
}

void XQRestRasterisationProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestRasterisationProcess::InitProcess() {

    fGas = (TRestGas*)this->GetGasMetadata();
    if (fGas != NULL) {
        if (fDriftVelocity <= 0) fDriftVelocity = fGas->GetDriftVelocity(fElectricField) * cmTomm;
    } else {
        cout << "REST_WARNING. No TRestGas found in TRestRun." << endl;
    }
}

//______________________________________________________________________________
void XQRestRasterisationProcess::BeginOfEventProcess() { fInputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestRasterisationProcess::ProcessEvent(TRestEvent* evInput) {
    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);
    //cout << fNumsGap << "  " << fSetGap << endl;
    //if (this->GetVerboseLevel() >= REST_Debug) {

    //能量cut
    if( fInputHitsEvent->GetTotalEnergy() < fEnergyRange.X() ||
        fInputHitsEvent->GetTotalEnergy() > fEnergyRange.Y() ){
        if( this->GetVerboseLevel() >= REST_Debug )
            cout << "XQRestHitsToTrackProcess: Energy over range: " << fInputHitsEvent->GetTotalEnergy() << "keV !"<<endl;
        return NULL;
    }

    Double_t fGap = fSetGap * fNumsGap;
    Double_t ras_x, ras_y, ras_z;

    for (int hit = 0; hit < fInputHitsEvent->GetNumberOfHits(); hit++) {
        Double_t x = fInputHitsEvent->GetX(hit);
        Double_t y = fInputHitsEvent->GetY(hit);
        Double_t z = fInputHitsEvent->GetZ(hit);
        Double_t e = fInputHitsEvent->GetEnergy(hit);

        // x rasterisation
        if ( x < 0 )
            ras_x = ((Int_t)( x / fGap )) * fGap - fGap / 2;
        else
            ras_x = ((Int_t)( x / fGap )) * fGap + fGap / 2;

        // y rasterisation
        if ( y < 0 )
            ras_y = ((Int_t)( y / fGap )) * fGap - fGap / 2;
        else
            ras_y = ((Int_t)( y / fGap )) * fGap + fGap / 2;

        // z rasterisation
        if ( fModel == 1 ) {
            Double_t time = z / fDriftVelocity;
            time = ((Int_t)(time / fSampling)) * fSampling;  // now time is in unit "us", but dispersed
            ras_z = time * fDriftVelocity;
        }else{
            if ( z < 0 )
                ras_z = ((Int_t)( z / fGap )) * fGap - fGap / 2;
            else
                ras_z = ((Int_t)( z / fGap )) * fGap + fGap / 2;
        }

        //merge hits
        bool merge_flag = false;
        for (int i = 0; i < number_x.size(); i++) {
            if( (abs(ras_x - number_x[i]) < 0.01) && (abs(ras_y - number_y[i]) < 0.01) && (abs(ras_z - number_z[i]) < 0.01) ) {
                number_e[i] += e;
                merge_flag = true;
                break;
            }
        }

        if( !merge_flag ){
            number_x.push_back( ras_x );
            number_y.push_back( ras_y );
            number_z.push_back( ras_z );
            number_e.push_back( e );
        }
    }

    for (int i = 0; i < number_x.size(); i++) {
        fOutputHitsEvent->AddHit(number_x[i], number_y[i], number_z[i], number_e[i]);
    }

    if (this->GetVerboseLevel() >= REST_Debug) {
    //if (1) {

        Int_t initialHits = fInputHitsEvent->GetNumberOfHits();
        Int_t finalHits = fOutputHitsEvent->GetNumberOfHits();

        if ( fModel == 1 ) {
            cout << "XQRestRasterisationProcess: Simulate Readout" << endl;
        }else{
            cout << "XQRestRasterisationProcess: Rasterisation for Analysis " << endl;
        }
        cout << " XQRestRasterisationProcess-DriftVelocity: " << fDriftVelocity << "mm/us" << endl;
        cout << " XQRestRasterisationProcess-ElectricField: " << fElectricField << "V/cm" << endl;
        printf(
          " XQRestRasterisationProcess: "
          "fDriftVelocity %lf mm/us; fSampling %lf MHz\n",
          fDriftVelocity, 1.0 / fSampling );
        cout<<"InputHitsEvent->GetTotalEnergy: "<<fInputHitsEvent->GetEnergy()<<endl;
        cout<<"fOutputHitsEvent->GetTotalEnergy: "<<fOutputHitsEvent->GetEnergy()<<endl;
        cout << "XQRestRasterisationProcess : Initial number of hits : " << initialHits << endl;
        cout << "XQRestRasterisationProcess : Final number of hits : " << finalHits << endl;
    }

    number_x.clear();
    number_y.clear();
    number_z.clear();
    number_e.clear();

    return fOutputHitsEvent;
}

//______________________________________________________________________________
void XQRestRasterisationProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestRasterisationProcess::EndProcess() {}

//______________________________________________________________________________
void XQRestRasterisationProcess::InitFromConfigFile() {
  fSampling = GetDblParameterWithUnits("sampling");
  fSetGap = GetDblParameterWithUnits("SetGap");
  fNumsGap = StringToInteger(GetParameter("NumsGap", "1"));
  fElectricField = GetDblParameterWithUnits("ElectricField");
  fDriftVelocity = StringToDouble(GetParameter("driftVelocity", "0")) * cmTomm;
  fModel = StringToInteger(GetParameter("Model", "1")); // Simulation: 1 or Analysis: 0
  fEnergyRange = StringTo2DVector(GetParameter("EnergyRange", "(2395,2520)")); //keV

}
