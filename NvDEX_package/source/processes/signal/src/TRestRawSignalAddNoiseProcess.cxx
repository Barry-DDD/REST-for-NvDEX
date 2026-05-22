///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestRawSignalAddNoiseProcess.cxx
///
///             February 2016: Javier Gracia
///_______________________________________________________________________________

#include "TRestRawSignalAddNoiseProcess.h"
using namespace std;

#include <TRestFFT.h>

#include <TFile.h>

ClassImp(TRestRawSignalAddNoiseProcess)
    //______________________________________________________________________________
    TRestRawSignalAddNoiseProcess::TRestRawSignalAddNoiseProcess() {
    Initialize();
}

//______________________________________________________________________________
TRestRawSignalAddNoiseProcess::TRestRawSignalAddNoiseProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName) == -1) LoadDefaultConfig();

    PrintMetadata();
    // TRestRawSignalAddNoiseProcess default constructor
}

//______________________________________________________________________________
TRestRawSignalAddNoiseProcess::~TRestRawSignalAddNoiseProcess() {
    delete fOutputSignalEvent;
    delete fInputSignalEvent;
    // TRestRawSignalAddNoiseProcess destructor
}

void TRestRawSignalAddNoiseProcess::LoadDefaultConfig() {
    SetName("addSignalNoiseProcess-Default");
    SetTitle("Default config");
    fNoiseMean = 0;

    fNoiseLevel = 1;
}

//______________________________________________________________________________
void TRestRawSignalAddNoiseProcess::Initialize() {
    SetSectionName(this->ClassName());
    fNoiseMean = 0;
    fNoiseLevel = 1;

    fInputSignalEvent = new TRestRawSignalEvent();
    fOutputSignalEvent = new TRestRawSignalEvent();

    fInputEvent = fInputSignalEvent;
    fOutputEvent = fOutputSignalEvent;
}

void TRestRawSignalAddNoiseProcess::LoadConfig(string cfgFilename, string name) {
    if (LoadConfigFromFile(cfgFilename, name) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
void TRestRawSignalAddNoiseProcess::InitProcess() {
    // Function to be executed once at the beginning of process
    // (before starting the process of the events)

    // Start by calling the InitProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::InitProcess();
}

//______________________________________________________________________________
void TRestRawSignalAddNoiseProcess::BeginOfEventProcess() { fOutputSignalEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* TRestRawSignalAddNoiseProcess::ProcessEvent(TRestEvent* evInput) {
    fInputSignalEvent = (TRestRawSignalEvent*)evInput;

    // cout<<"Number of signals "<< fInputSignalEvent->GetNumberOfSignals()<<
    // endl;

    if (fInputSignalEvent->GetNumberOfSignals() <= 0) return NULL;

    for (int n = 0; n < fInputSignalEvent->GetNumberOfSignals(); n++) {
        TRestRawSignal noiseSignal;

        // Asign ID and add noise
        fInputSignalEvent->GetSignal(n)->GetWhiteNoiseSignal(&noiseSignal, fNoiseMean, fNoiseLevel);
        noiseSignal.SetSignalID(fInputSignalEvent->GetSignal(n)->GetSignalID());

        fOutputSignalEvent->AddSignal(noiseSignal);
    }
    // Draw Signals in every channels
    for (int n = 0; n < fOutputSignalEvent->GetNumberOfSignals(); n++) {
        TRestRawSignal outSignal = *fOutputSignalEvent->GetSignal(n);
        Int_t nBins = outSignal.GetNumberOfPoints();
        std::vector<double> x(nBins);
        for (int i = 0; i < nBins; i++) {
            x[i] = i* 0.004; 
        }
        std::vector<double> out(nBins);
        for (int i = 0; i < nBins; i++) {
            out[i] = outSignal.GetData(i);
        }

        


        // if (n<50)
        // {   
        //     TCanvas *c1 = new TCanvas(Form("c1_%d", n), Form("c1_%d", n), 800, 600);
        //     TGraph *graph1 = new TGraph(nBins, &x[0], &out[0]);
        //     graph1->SetTitle(" ");
        //     graph1->GetXaxis()->SetTitle("Time/s");
        //     graph1->GetYaxis()->SetTitle("U/mV");
        //     // graph1->GetYaxis()->SetRangeUser(700, 1300);
        //     // graph1->GetXaxis()->SetRangeUser(2.8, 3.6);
        //     graph1->Draw("AL");
        //     c1->Draw();
        //     c1->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/SignalAddnoise_%d.png", n));
        // }

    }


    return fOutputSignalEvent;
}

//______________________________________________________________________________
void TRestRawSignalAddNoiseProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void TRestRawSignalAddNoiseProcess::EndProcess() {
    // Function to be executed once at the end of the process
    // (after all events have been processed)

    // Start by calling the EndProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::EndProcess();
}

//______________________________________________________________________________
void TRestRawSignalAddNoiseProcess::InitFromConfigFile() {
    fNoiseMean = StringToDouble(GetParameter("noiseMean"));
    fNoiseLevel = StringToDouble(GetParameter("noiseLevel"));
}
