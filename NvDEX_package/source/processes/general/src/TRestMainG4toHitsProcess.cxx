///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestMainG4toHitsProcess.cxx
///
///
///             Simple process to convert a TRestG4Event class into a
///    		    TRestHitsEvent, that is, we just "extract" the hits
///    information
///             Date : oct/2016
///             Author : I. G. Irastorza
///
///_______________________________________________________________________________

#include "TRestMainG4toHitsProcess.h"
using namespace std;

ClassImp(TRestMainG4toHitsProcess)
    //______________________________________________________________________________
    TRestMainG4toHitsProcess::TRestMainG4toHitsProcess() {
    Initialize();
}

//______________________________________________________________________________
TRestMainG4toHitsProcess::TRestMainG4toHitsProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
TRestMainG4toHitsProcess::~TRestMainG4toHitsProcess() {
    delete fG4Event;
    delete fHitsEvent;
}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    cout << "G4 to hits metadata not found. Loading default values" << endl;

    fTrackNum = 0;
}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::Initialize() {
    SetSectionName(this->ClassName());

    fG4Event = new TRestG4Event();
    fHitsEvent = new TRestHitsEvent();

    fOutputEvent = fHitsEvent;
    fInputEvent = fG4Event;
}

void TRestMainG4toHitsProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::InitProcess() {
    //    TRestEventProcess::ReadObservables();

    fG4Metadata = (TRestG4Metadata*)GetGeant4Metadata();

    for (unsigned int n = 0; n < fVolumeSelection.size(); n++) {
        if (fG4Metadata->GetActiveVolumeID(fVolumeSelection[n]) >= 0)
            fVolumeId.push_back(fG4Metadata->GetActiveVolumeID(fVolumeSelection[n]));
        else if (GetVerboseLevel() >= REST_Warning)
            cout << "TRestMainG4toHitsProcess. volume name : " << fVolumeSelection[n]
                 << " not found and will not be added." << endl;
    }
}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::BeginOfEventProcess() { fHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* TRestMainG4toHitsProcess::ProcessEvent(TRestEvent* evInput) {
    fG4Event = (TRestG4Event*)evInput;

    fHitsEvent->SetRunOrigin(fG4Event->GetRunOrigin());
    fHitsEvent->SetSubRunOrigin(fG4Event->GetSubRunOrigin());
    fHitsEvent->SetID(fG4Event->GetID());
    fHitsEvent->SetSubID(fG4Event->GetSubID());
    fHitsEvent->SetSubEventTag(fG4Event->GetSubEventTag());
    fHitsEvent->SetTimeStamp(fG4Event->GetTimeStamp());
    fHitsEvent->SetState(fG4Event->isOk());

    Int_t i, j;
    Double_t x, y, z, E;

    if(fTrackNum == 2){// 0vbb
        if (this->GetVerboseLevel() >= REST_Debug)
            cout<<"save track0-1 hits"<<endl;
      	Int_t track_No = 1;
      	Double_t tempE = fG4Event->GetTrack(1)->GetEnergy();
      	for(i=2;i<fG4Event->GetNumberOfTracks();i++){
        		Double_t temp = fG4Event->GetTrack(i)->GetEnergy();
        		if(tempE < temp ){
        				tempE = temp;
        				track_No = i;
        		}
      	}
        for (j = fG4Event->GetTrack(0)->GetNumberOfHits()-1; j>=0; j--) {
            // read x,y,z and E of every hit in the G4 event
            x = fG4Event->GetTrack(0)->GetHits()->fX[j];
            y = fG4Event->GetTrack(0)->GetHits()->fY[j];
            z = fG4Event->GetTrack(0)->GetHits()->fZ[j];
            E = fG4Event->GetTrack(0)->GetHits()->fEnergy[j];

            Bool_t addHit = true;
            if (fVolumeId.size() > 0) {
                addHit = false;
                for (unsigned int n = 0; n < fVolumeId.size(); n++)
                    if (fG4Event->GetTrack(0)->GetHits()->GetVolumeId(j) == fVolumeId[n]) addHit = true;
            }

            // and write them in the output hits event:
            if (addHit && E > 0) fHitsEvent->AddHit(x, y, z, E);
      	}
        for (j = 0; j < fG4Event->GetTrack(track_No)->GetNumberOfHits(); j++) {
            // read x,y,z and E of every hit in the G4 event
            x = fG4Event->GetTrack(track_No)->GetHits()->fX[j];
            y = fG4Event->GetTrack(track_No)->GetHits()->fY[j];
            z = fG4Event->GetTrack(track_No)->GetHits()->fZ[j];
            E = fG4Event->GetTrack(track_No)->GetHits()->fEnergy[j];
            Bool_t addHit = true;
            if (fVolumeId.size() > 0) {
                addHit = false;
                for (unsigned int n = 0; n < fVolumeId.size(); n++)
                    if (fG4Event->GetTrack(track_No)->GetHits()->GetVolumeId(j) == fVolumeId[n]) addHit = true;
            }
            // and write them in the output hits event:
            if (addHit && E > 0) fHitsEvent->AddHit(x, y, z, E);
        }
    }else if(fTrackNum == 1){// electron
        if (this->GetVerboseLevel() >= REST_Debug)
            cout<<"save Track0 hits"<<endl;
        Int_t track_No = 0;
        Double_t tempE = fG4Event->GetTrack(0)->GetEnergy();
        for(i=1;i<fG4Event->GetNumberOfTracks();i++){
            Double_t temp = fG4Event->GetTrack(i)->GetEnergy();
            if(tempE < temp ){
                tempE = temp;
                track_No = i;
            }
        }
        for (j = 0; j < fG4Event->GetTrack(track_No)->GetNumberOfHits(); j++) {
            // read x,y,z and E of every hit in the G4 event
            x = fG4Event->GetTrack(track_No)->GetHits()->fX[j];
            y = fG4Event->GetTrack(track_No)->GetHits()->fY[j];
            z = fG4Event->GetTrack(track_No)->GetHits()->fZ[j];
            E = fG4Event->GetTrack(track_No)->GetHits()->fEnergy[j];
            Bool_t addHit = true;
            if (fVolumeId.size() > 0) {
                addHit = false;
                for (unsigned int n = 0; n < fVolumeId.size(); n++)
                    if (fG4Event->GetTrack(track_No)->GetHits()->GetVolumeId(j) == fVolumeId[n]) addHit = true;
            }

            // and write them in the output hits event:
            if (addHit && E > 0) fHitsEvent->AddHit(x, y, z, E);
        }
    }else if(fTrackNum == 0){ // all
      if (this->GetVerboseLevel() >= REST_Debug)
          cout<<"save all hits"<<endl;
      for (i = 0; i < fG4Event->GetNumberOfTracks(); i++) {
          for (j = 0; j < fG4Event->GetTrack(i)->GetNumberOfHits(); j++) {
              // read x,y,z and E of every hit in the G4 event
              x = fG4Event->GetTrack(i)->GetHits()->fX[j];
              y = fG4Event->GetTrack(i)->GetHits()->fY[j];
              z = fG4Event->GetTrack(i)->GetHits()->fZ[j];
              E = fG4Event->GetTrack(i)->GetHits()->fEnergy[j];
              Bool_t addHit = true;
              if (fVolumeId.size() > 0) {
                  addHit = false;
                  for (unsigned int n = 0; n < fVolumeId.size(); n++)
                      if (fG4Event->GetTrack(i)->GetHits()->GetVolumeId(j) == fVolumeId[n]) addHit = true;
              }
              // and write them in the output hits event:
              if (addHit && E > 0) fHitsEvent->AddHit(x, y, z, E);
          }
      }
    }

    if (this->GetVerboseLevel() >= REST_Debug) {
        cout << "TRestMainG4toHitsProcess. Hits added : " << fHitsEvent->GetNumberOfHits() << endl;
        cout << "TRestMainG4toHitsProcess. Hits total energy : " << fHitsEvent->GetEnergy() << endl;
    }

    return fHitsEvent;
}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::EndProcess() {
    // Function to be executed once at the end of the process
    // (after all events have been processed)

    // Start by calling the EndProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::EndProcess();
}

//______________________________________________________________________________
void TRestMainG4toHitsProcess::InitFromConfigFile() {
    size_t position = 0;
    string addVolumeDefinition;
    fTrackNum = StringToInteger(GetParameter("TrackNum"));
    while ((addVolumeDefinition = GetKEYDefinition("addVolume", position)) != "")
        fVolumeSelection.push_back(GetFieldValue("name", addVolumeDefinition));


}
