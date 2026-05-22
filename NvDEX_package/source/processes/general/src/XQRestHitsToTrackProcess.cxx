///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             XQTRestTrackReductionProcess.cxx
///
///             Mar 2020:   Created by LiTao SYSU
//
///_______________________________________________________________________________

#include "XQRestHitsToTrackProcess.h"
using namespace std;

ClassImp(XQRestHitsToTrackProcess)
    //______________________________________________________________________________
    XQRestHitsToTrackProcess::XQRestHitsToTrackProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestHitsToTrackProcess::XQRestHitsToTrackProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName) == -1) LoadDefaultConfig();

    // XQRestHitsToTrackProcess default constructor
}

//______________________________________________________________________________
XQRestHitsToTrackProcess::~XQRestHitsToTrackProcess() {
    delete fHitsEvent;
    delete fTrackEvent;
    // XQRestHitsToTrackProcess destructor
}

void XQRestHitsToTrackProcess::LoadDefaultConfig() {
    SetName("XQRestHitsToTrackProcess");
    SetTitle("Default config");

    fEnergyRange.Set(2.395,2.520);

}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::Initialize() {
    SetSectionName(this->ClassName());

    fHitsEvent = new TRestHitsEvent();
    fTrackEvent = new TRestTrackEvent();

    fOutputEvent = fTrackEvent;
    fInputEvent = fHitsEvent;
}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::LoadConfig(string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::InitProcess() {
    // Function to be executed once at the beginning of process
    // (before starting the process of the events)

    // Start by calling the InitProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::InitProcess();

    cout << __PRETTY_FUNCTION__ << endl;
}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::BeginOfEventProcess() { fTrackEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestHitsToTrackProcess::ProcessEvent(TRestEvent* evInput) {

    fHitsEvent = (TRestHitsEvent*)evInput;
    fTrackEvent->SetEventInfo(fHitsEvent);

    //能量cut
    fHitsEvent = (TRestHitsEvent*)evInput;
    if( fHitsEvent->GetTotalEnergy() < fEnergyRange.X() ||
        fHitsEvent->GetTotalEnergy() > fEnergyRange.Y() ){
        if( this->GetVerboseLevel() >= REST_Debug )
            cout << "XQRestHitsToTrackProcess: Energy over range: " << fHitsEvent->GetTotalEnergy() << "keV !"<<endl;
        return NULL;
    }

    TRestTrack* track = new TRestTrack();
    TRestVolumeHits volHit;

    for (unsigned int nhit = 0; nhit < fHitsEvent->GetNumberOfHits(); nhit++) {
        const Double_t x = fHitsEvent->GetX(nhit);
        const Double_t y = fHitsEvent->GetY(nhit);
        const Double_t z = fHitsEvent->GetZ(nhit);
        const Double_t en = fHitsEvent->GetEnergy(nhit);
        TVector3 pos(x, y, z);
        TVector3 sigma(0., 0., 0.);
        volHit.AddHit(pos, en, sigma);
    }

    track->SetParentID(0);
    track->SetTrackID(fTrackEvent->GetNumberOfTracks() + 1);
    track->SetVolumeHits(volHit);
    volHit.RemoveHits();

    if( track->GetVolumeHits()->GetNumberOfHits() == 0 ){
        if( this->GetVerboseLevel() >= REST_Debug )
            cout << "ERROR: XQRestHitsToTrackProcess: Hits Nums = 0 !"<<endl;
        return NULL;
    }

    fTrackEvent->AddTrack(track);
    fTrackEvent->SetLevels();
    if (fTrackEvent->GetNumberOfTracks() == 0) return NULL;
    if (GetVerboseLevel() >= REST_Debug){
        fTrackEvent->PrintOnlyTracks();
        cout << "\n XQRestHitsToTrackProcess: \n HitEvent Hits nums: " << fHitsEvent->GetNumberOfHits() << endl;
        cout << "TrackEvent Hits nums: " <<
            fTrackEvent->GetTrackById(fTrackEvent->GetNumberOfTracks() )->GetVolumeHits()->GetNumberOfHits() <<
            "; Track nums: " << fTrackEvent->GetNumberOfTracks() << endl;
    }
    delete track;
    return fTrackEvent;
}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::EndProcess() {
    // Function to be executed once at the end of the process
    // (after all events have been processed)

    // Start by calling the EndProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::EndProcess();
}

//______________________________________________________________________________
void XQRestHitsToTrackProcess::InitFromConfigFile() {
  fEnergyRange = StringTo2DVector(GetParameter("EnergyRange", "(2.395,2.520)")); //MeV
}
