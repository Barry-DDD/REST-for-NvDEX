/*************************************************************************
 * This file is part of the REST software framework.                     *
 *                                                                       *
 * Copyright (C) 2016 GIFNA/TREX (University of Zaragoza)                *
 * For more information see http://gifna.unizar.es/trex                  *
 *                                                                       *
 * REST is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or     *
 * (at your option) any later version.                                   *
 *                                                                       *
 * REST is distributed in the hope that it will be useful,               *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          *
 * GNU General Public License for more details.                          *
 *                                                                       *
 * You should have a copy of the GNU General Public License along with   *
 * REST in $REST_PATH/LICENSE.                                           *
 * If not, see http://www.gnu.org/licenses/.                             *
 * For the list of contributors see $REST_PATH/CREDITS.                  *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
/// The TRestRawSignalShapingProcess allows to convolute a given time signal
/// response with the TRestRawSignal input signals found inside the input
/// TRestRawSignalEvent. This process may serve for conditioning the signals
/// produced in a MonteCarlo simulation and emulate those from real
/// electronics.
///
/// For the moment, only gaussian response has been implemented.
///
/// The different parameters allowed in this process are:
///
/// * **shapingType**: It defines the type of convolution to be performed.
///     - gaus : It produces a gausian convolution.
///     - shaper : It produces a shaping following traditional shaper
///               waveforms.
///     - shaperSin : It produces a shaping following traditional shaper
///               waveforms, it includes a sinusoidal effect.
///     - responseFile : A file providing a user provided response (TODO).
///
/// * **shapingTime** : The standard deviation of the gaussian convolution,
///                     or the shaping time on shaper models. Defined in
///                     samples unit.
/// * **gain** : A factor to amplify or attenuate the signal.
/// * **responseFile** : A response file to be used in case the shapingType
/// is defined to use a response file.
///
///--------------------------------------------------------------------------
///
/// RESTsoft - Software for Rare Event Searches with TPCs
///
/// History of developments:
///
/// 2016-March: First implementation
///             Xinglong
///
/// 2018-March: Transfered to TRestRawSignal
///             Javier Galan
///
/// \class      TRestRawSignalShapingProcess
/// \author     Xinglong
/// \author     Javier Galan
///
/// <hr>
///
#include "TRestRawSignalShapingProcess.h"
using namespace std;

#include <TFile.h>
#include <TMath.h>
#include "TCanvas.h"
#include "TGraph.h"

ClassImp(TRestRawSignalShapingProcess)

    ///////////////////////////////////////////////
    /// \brief Default constructor
    ///
    TRestRawSignalShapingProcess::TRestRawSignalShapingProcess()
{
    Initialize();
}

///////////////////////////////////////////////
/// \brief Constructor loading data from a config file
///
/// If no configuration path is defined using TRestMetadata::SetConfigFilePath
/// the path to the config file must be specified using full path, absolute or
/// relative.
///
/// The default behaviour is that the config file must be specified with
/// full path, absolute or relative.
///
/// \param cfgFileName A const char* giving the path to an RML file.
///
TRestRawSignalShapingProcess::TRestRawSignalShapingProcess(char *cfgFileName)
{
    Initialize();

    if (LoadConfigFromFile(cfgFileName) == -1)
        LoadDefaultConfig();

    PrintMetadata();
}

///////////////////////////////////////////////
/// \brief Default destructor
///
TRestRawSignalShapingProcess::~TRestRawSignalShapingProcess()
{
    delete fOutputSignalEvent;
    delete fInputSignalEvent;
}

///////////////////////////////////////////////
/// \brief Function to load the default config in absence of RML input
///
void TRestRawSignalShapingProcess::LoadDefaultConfig()
{
    SetName("rawSignalShapingProcess-Default");
    SetTitle("Default config");
}

///////////////////////////////////////////////
/// \brief Function to initialize input/output event members and define the
/// section name
///
void TRestRawSignalShapingProcess::Initialize()
{
    SetSectionName(this->ClassName());

    fInputSignalEvent = new TRestRawSignalEvent();
    fOutputSignalEvent = new TRestRawSignalEvent();

    fInputEvent = fInputSignalEvent;
    fOutputEvent = fOutputSignalEvent;
}

///////////////////////////////////////////////
/// \brief Function to load the configuration from an external configuration
/// file.
///
/// If no configuration path is defined in TRestMetadata::SetConfigFilePath
/// the path to the config file must be specified using full path, absolute or
/// relative.
///
/// \param cfgFileName A const char* giving the path to an RML file.
/// \param name The name of the specific metadata. It will be used to find the
/// correspondig TRestGeant4AnalysisProcess section inside the RML.
///
void TRestRawSignalShapingProcess::LoadConfig(string cfgFilename, string name)
{
    if (LoadConfigFromFile(cfgFilename, name) == -1)
        LoadDefaultConfig();
}

///////////////////////////////////////////////
/// \brief Process initialization. Observable names are interpreted and auxiliar
/// observable members, related to VolumeEdep, MeanPos, TracksCounter, TrackEDep
/// observables defined in TRestGeant4AnalysisProcess are filled at this stage.
///
void TRestRawSignalShapingProcess::InitProcess()
{
    /*
     * NOT IMPLEMENTED. TODO To use a generic response from a
     * predefined TRestSignal
     *
     * For the moment we do only a gausian shaping"
     * /

    responseSignal = new TRestRawSignal();

    if( fShapingType == "responseFile" )
    {
        TString fullPathName = (TString) getenv("REST_PATH") + "/data/signal/" +
    fResponseFilename; TFile *f = new TFile(fullPathName); responseSignal =
    (TRestRawSignal *) f->Get("signal Response"); f->Close();
    }

    if( GetVerboseLevel() >= REST_Debug )
    {
        CreateCanvas();
        fCanvas->Draw();
    }

    if( fShapingType == "gaus" )
    {
        responseSignal->InitGaussian( 100, 100, 30, 200 );

        if( GetVerboseLevel() >= REST_Debug )
        {
            responseSignal->GetGraph()->Draw();
            fCanvas->Update();
            GetChar();
        }
    }
    */
}

///////////////////////////////////////////////
/// \brief Function to include required initialization before each event starts
/// to process.
///
void TRestRawSignalShapingProcess::BeginOfEventProcess() { fOutputSignalEvent->Initialize(); }

///////////////////////////////////////////////
/// \brief The main processing event function
///
TRestEvent *TRestRawSignalShapingProcess::ProcessEvent(TRestEvent *evInput)
{
    fInputSignalEvent = (TRestRawSignalEvent *)evInput;

    if (fInputSignalEvent->GetNumberOfSignals() <= 0)
        return NULL;

    double *rsp;
    Int_t Nr = 0;

    if (fShapingType == "gaus")
    {
        Double_t amp = fShapingGain;
        Int_t cBin = (Int_t)(fShapingTime * 3.5);
        Nr = 2 * cBin;
        Double_t sigma = fShapingTime;

        rsp = new double[Nr];
        for (int i = 0; i < Nr; i++)
            rsp[i] = (amp * TMath::Exp(-0.5 * (i - cBin) * (i - cBin) / sigma / sigma));
    }
    else if (fShapingType == "shaper")
    {
        Nr = (Int_t)(5 * fShapingTime);

        rsp = new double[Nr];
        for (int i = 0; i < Nr; i++)
        {
            Double_t coeff = ((Double_t)i) / fShapingTime;
            rsp[i] = (fShapingGain * TMath::Exp(-3. * coeff) * coeff * coeff * coeff);
        }
    }
    else if (fShapingType == "shaperSin")
    {
        Nr = (Int_t)(5 * fShapingTime);

        rsp = new double[Nr];
        for (int i = 0; i < Nr; i++)
        {
            Double_t coeff = ((Double_t)i) / fShapingTime;
            rsp[i] = (fShapingGain * TMath::Exp(-3. * coeff) * coeff * coeff * coeff * sin(coeff));
        }
    }
    else if (fShapingType == "response")
    {
        Nr = (Int_t)(5 * fShapingTime);
        rsp = new double[Nr];
        for (int i = 0; i < Nr; i++)
        {
            rsp[i] = fShapingGain * (TMath::Exp(-i / fShapingTime));
        }
    }
    else
    {
        if (GetVerboseLevel() >= REST_Warning)
            cout << "REST WARNING. Shaping type : " << fShapingType << " is not defined!!" << endl;
        return NULL;
    }

    // Making sure that rsp integral is 1.
    Double_t sum = 0;
    for (int n = 0; n < Nr; n++) sum += rsp[n];
    for (int n = 0; n < Nr; n++) rsp[n] = rsp[n] / sum;

    Double_t totalIons = 0;
    Double_t totalEnergy = 0;
    Double_t totalData = 0;
    // Int_t NumberOfSignal = fInputSignalEvent->GetNumberOfSignals();
    // cout<<"Signal Numbers:"<<NumberOfSignal<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
    for (int n = 0; n < fInputSignalEvent->GetNumberOfSignals(); n++)
    {
        // cout<<"Signal Number:"<<n<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
        int temp = 0;
        TRestRawSignal shapingSignal = TRestRawSignal();
        TRestRawSignal inSignal = *fInputSignalEvent->GetSignal(n);
        Int_t nBins = inSignal.GetNumberOfPoints();
        // cout<<"nBins:"<<nBins<<endl;
        // std::vector<double> x(nBins);
        // double t = 0;
        // for (int i = 0; i < nBins; i++)
        // {

        //     x[i] = t;
        //     t++;
        // }
        // std::vector<double> M(nBins);
        // for (int i = 0; i < nBins; i++)
        // {
        //     M[i] = inSignal.GetData(i)*1000/21.9;
        //     totalIons+=M[i];
        // }

        vector<double> out(nBins);
        for (int m = 0; m < nBins; m++)
        {
            if (inSignal.GetData(m) > 0)
            {
                for (int n = 0; n < Nr && m + n < nBins; n++)
                    out[m + n] += rsp[n] * inSignal.GetData(m);
            }
        }
        // bool aboveThreshold = false;
        // double threshold = 1;
        // std::vector<int> boundaryPositions(4, 0);

        // int index = 0;
        // for (int i = 0; i < nBins; i++)
        // {
        //     if (M[i] > threshold && !aboveThreshold)
        //     {
        //         boundaryPositions[index] = i;
        //         aboveThreshold = true;
        //         index++;
        //     }
        //     else if (M[i] <= threshold && aboveThreshold)
        //     {
        //         boundaryPositions[index] = i;
        //         aboveThreshold = false;
        //         index++;
        //     }

        //     if (index >= 4)
        //     {
        //         break;
        //     }
        // }

        // if (boundaryPositions[0] * boundaryPositions[1] * boundaryPositions[2] * boundaryPositions[3] != 0)
        // {
        //     double peaklenth1 = (boundaryPositions[1] - boundaryPositions[0]) * 4.0; // in ms
        //     double peaklenth2 = (boundaryPositions[3] - boundaryPositions[2]) * 4.0;
        //     double signallenth = (boundaryPositions[3] - boundaryPositions[0]) * 4.0;

        //     std::ofstream OutFile1("peaklenth1.txt", std::ios::app);
        //     OutFile1 << peaklenth1 << std::endl;
        //     OutFile1.close();
        //     std::ofstream OutFile2("peaklenth2.txt", std::ios::app);
        //     OutFile2 << peaklenth2 << std::endl;
        //     OutFile2.close();
        //     std::ofstream OutFile3("signallenth.txt", std::ios::app);
        //     OutFile3 << signallenth << std::endl;
        //     OutFile3.close();
        // }

        // cout << "channel id :" << n << endl;
        // TCanvas *c1 = new TCanvas(Form("c1_%d", n), Form("c1_%d", n), 800, 600);
        // TGraph *graph1 = new TGraph(nBins, &x[0], &out[0]);
        // graph1->SetTitle(" ");
        // graph1->GetXaxis()->SetTitle("Time/s");
        // graph1->GetYaxis()->SetTitle("U/mV");
        // graph1->Draw("AL");
        // c1->Draw();
        // c1->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/SignalShaping_%d.png", n));

        // TCanvas *c2 = new TCanvas(Form("c2_%d", n), Form("c2_%d", n), 800, 600);
        // TGraph *graph2 = new TGraph(nBins, &x[0], &rsp[0]);
        // graph2->SetTitle(Form("ResponseFunction_%d", n));
        // graph2->GetXaxis()->SetTitle("Time");
        // graph2->Draw("AL");
        // c2->Draw();
        // c2->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/ResponseFunction_%d.png", n));

        // TCanvas *c3 = new TCanvas(Form("c3_%d", n), Form("c3_%d", n), 800, 600);
        // TGraph *graph3 = new TGraph(nBins, &x[0], &M[0]);
        // graph3->SetTitle(" ");
        // graph3->GetXaxis()->SetTitle("Time(us)");
        // graph3->GetXaxis()->SetRangeUser(0, 200);
        // graph3->GetYaxis()->SetTitle("N");
        // graph3->Draw("AL");
        // c3->Draw();
        // c3->SaveAs(Form("/home/rest/rest_workspace/TEST/Xe136 test/signal/Signal_%d.png", n));

        for (int i = 0; i < nBins; i++)
        {
            shapingSignal.AddPoint(out[i]);
        }
        shapingSignal.SetSignalID(inSignal.GetSignalID());

        fOutputSignalEvent->AddSignal(shapingSignal);
    }

    delete[] rsp;
    
    // cout << "totalIons:" << totalIons << "<<!<<" << endl;
    // cout << "totalEnergy:" << totalEnergy << "<<!<<" << endl;
    // cout << "totalData:" << totalData << "<<!<<" << endl;
    // cout << " ! " << fInputSignalEvent->GetIntegral() << " ! " << fOutputSignalEvent->GetIntegral() <<endl;
    return fOutputSignalEvent;
}

///////////////////////////////////////////////
/// \brief Function to include required actions after each event has been
/// processed.
///
void TRestRawSignalShapingProcess::EndOfEventProcess() {}

///////////////////////////////////////////////
/// \brief Function to include required actions after all events have been
/// processed.
///
void TRestRawSignalShapingProcess::EndProcess()
{
    // Function to be executed once at the end of the process
    // (after all events have been processed)

    // Start by calling the EndProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::EndProcess();
}

///////////////////////////////////////////////
/// \brief Function to read input parameters from the RML
/// TRestGeant4AnalysisProcess metadata section
///
void TRestRawSignalShapingProcess::InitFromConfigFile()
{
    // It is not used for the moment
    fResponseFilename = GetParameter("responseFile");

    // gaus, responseFile, etc
    fShapingType = GetParameter("shapingType", "gaus");

    fShapingTime = StringToDouble(GetParameter("shapingTime", "10"));

    fShapingGain = StringToDouble(GetParameter("gain", "1"));
}
