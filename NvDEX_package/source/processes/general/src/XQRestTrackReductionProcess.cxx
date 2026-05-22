///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             XQRestTrackReductionProcess.cxx
///
///             Mar 2020:   Created by LiTao SYSU
//
///_______________________________________________________________________________

#include "XQRestTrackReductionProcess.h"
using namespace std;

ClassImp(XQRestTrackReductionProcess)
    //______________________________________________________________________________
    XQRestTrackReductionProcess::XQRestTrackReductionProcess() {
    Initialize();
}

//______________________________________________________________________________
XQRestTrackReductionProcess::XQRestTrackReductionProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
XQRestTrackReductionProcess::~XQRestTrackReductionProcess() {
    delete fInputTrackEvent;
    delete fOutputTrackEvent;
}

void XQRestTrackReductionProcess::LoadDefaultConfig() {
    SetName("trackReductionProcess");
    SetTitle("Default config");

    fStartingDistance = 0.5;
    fMinimumDistance = 3;
    fDistanceFactor = 1.5;
    fStartingNodes = 300;
    fMaxNodes = 30;
    fHitsEnergy = 1;
}

//______________________________________________________________________________
void XQRestTrackReductionProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputTrackEvent = new TRestTrackEvent();
    fOutputTrackEvent = new TRestTrackEvent();

    fOutputEvent = fOutputTrackEvent;
    fInputEvent = fInputTrackEvent;
}

void XQRestTrackReductionProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
void XQRestTrackReductionProcess::InitProcess() { cout << __PRETTY_FUNCTION__ << endl; }

//______________________________________________________________________________
void XQRestTrackReductionProcess::BeginOfEventProcess() { fOutputTrackEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* XQRestTrackReductionProcess::ProcessEvent(TRestEvent* evInput) {
    fInputTrackEvent = (TRestTrackEvent*)evInput;
    fOutputTrackEvent->SetEventInfo(fInputTrackEvent);
    // Copying the input tracks to the output track
    for (int tck = 0; tck < fInputTrackEvent->GetNumberOfTracks(); tck++)
        fOutputTrackEvent->AddTrack(fInputTrackEvent->GetTrack(tck));

    if (this->GetVerboseLevel() >= REST_Debug) fInputTrackEvent->PrintOnlyTracks();

    Int_t ParentID = 1;
    TRestTrack* track = fInputTrackEvent->GetTrackById(ParentID);
    TRestVolumeHits* hits = track->GetVolumeHits();

    TRestTrack *outputTrack = new TRestTrack();
    TRestVolumeHits* outputHits = outputTrack->GetVolumeHits();

    if (this->GetVerboseLevel() >= REST_Debug)
        cout << "XQRestTrackReductionProcess. Reducing hits in track id : " << track->GetTrackID() << endl;

    Int_t counts = 0;
    Double_t init_distance = fStartingDistance;
    while(counts< 5){
        TRestVolumeHits *tempHits = new TRestVolumeHits();
        // Copying the input hits event to the output hits event
        for (int h = 0; h < hits->GetNumberOfHits(); h++)
          if( hits->GetEnergy(h) > fHitsEnergy ){
            tempHits->AddHit( hits->GetX(h), hits->GetY(h), hits->GetZ(h),
                            hits->GetEnergy(h), 0, 0, 0 );
          }

        // Reducing the hits
        if(tempHits->GetNumberOfHits() > fStartingNodes &&
              tempHits->GetNumberOfHits() < fMaxNodes){
            //cout << counts << endl;
            for (int h = 0; h < tempHits->GetNumberOfHits(); h++)
                outputHits->AddHit( tempHits->GetX(h), tempHits->GetY(h), tempHits->GetZ(h),
                                tempHits->GetEnergy(h), 0, 0, 0 );
            break;
        }

        Double_t distance = init_distance;
        while ( distance < fMinimumDistance || tempHits->GetNumberOfHits() > fMaxNodes ) {
            Bool_t merged = true;
            while (merged) {
                merged = false;
                for (int i = 0; i < tempHits->GetNumberOfHits(); i++) {
                    for (int j = i + 1; j < tempHits->GetNumberOfHits(); j++) {
                        if (tempHits->GetDistance2(i, j) < distance * distance) {
                            tempHits->MergeHits(i, j);
                            merged = true;
                        }
                    }
                }
            }
            distance *= fDistanceFactor;
        }
        if(tempHits->GetNumberOfHits() > fStartingNodes &&
              tempHits->GetNumberOfHits() < fMaxNodes){
            //cout << counts << endl;
            for (int h = 0; h < tempHits->GetNumberOfHits(); h++)
                outputHits->AddHit( tempHits->GetX(h), tempHits->GetY(h), tempHits->GetZ(h),
                                tempHits->GetEnergy(h), 0, 0, 0 );
            delete tempHits;
            break;
        }
        init_distance /= 1.2;
        counts ++;
    }

    if( outputHits->GetNumberOfHits() == 0 ){
        if( this->GetVerboseLevel() >= REST_Debug )
            cout << "ERROR: XQRestTrackReductionProcess: Hits Nums = 0 !"<<endl;
        return NULL;
    }

    outputTrack->SetParentID(ParentID);
    outputTrack->SetTrackID(fOutputTrackEvent->GetNumberOfTracks() + 1);
    fOutputTrackEvent->AddTrack(outputTrack);
    fOutputTrackEvent->SetLevels();
    delete outputTrack;

    if (this->GetVerboseLevel() >= REST_Debug)
      cout << "\n XQRestTrackReductionProcess: \n TrackEvent Hits nums: " <<
          fOutputTrackEvent->GetTrackById(fOutputTrackEvent->GetNumberOfTracks() )->GetVolumeHits()->GetNumberOfHits() <<
          "; Track nums: " << fOutputTrackEvent->GetNumberOfTracks() << endl;
    return fOutputTrackEvent;
}

//______________________________________________________________________________
void XQRestTrackReductionProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void XQRestTrackReductionProcess::EndProcess() {}

//______________________________________________________________________________
void XQRestTrackReductionProcess::InitFromConfigFile() {
  fStartingDistance = GetDblParameterWithUnits("startingDistance");
  fMinimumDistance = GetDblParameterWithUnits("minimumDistance");
  fDistanceFactor = StringToDouble(GetParameter("distanceStepFactor"));
  fStartingNodes = StringToDouble(GetParameter("StartingNodes"));
  fMaxNodes = StringToDouble(GetParameter("maxNodes"));
  fHitsEnergy = StringToDouble(GetParameter("HitsEnergy"));
}
