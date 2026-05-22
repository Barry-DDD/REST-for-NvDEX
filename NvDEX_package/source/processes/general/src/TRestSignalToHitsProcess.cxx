///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestSignalToHitsProcess.cxx
///
///             jan 2016:  Javier Galan
///_______________________________________________________________________________

#include "TRestSignalToHitsProcess.h"

#include <TRestDetectorSetup.h>
#include <TFile.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TF1Convolution.h>
#include <TRestFFT.h>
#include <TStyle.h>
#include <TVirtualFitter.h>
#include <Math/MinimizerOptions.h>
#include <TMarker.h>
#include <TLatex.h>

using namespace std;

const double cmTomm = 10.;

ClassImp(TRestSignalToHitsProcess)
    //______________________________________________________________________________
    TRestSignalToHitsProcess::TRestSignalToHitsProcess()
{
    Initialize();
}

//______________________________________________________________________________
TRestSignalToHitsProcess::TRestSignalToHitsProcess(char *cfgFileName)
{
    Initialize();

    LoadConfig(cfgFileName);

    // TRestSignalToHitsProcess default constructor
}

//______________________________________________________________________________
TRestSignalToHitsProcess::~TRestSignalToHitsProcess()
{
    delete fHitsEvent;
    delete fSignalEvent;
    // TRestSignalToHitsProcess destructor
}

void TRestSignalToHitsProcess::LoadDefaultConfig()
{
    SetName("signalToHitsProcess-Default");
    SetTitle("Default config");

    cout << "Signal to hits metadata not found. Loading default values" << endl;

    fElectricField = 1000;
    fGasPressure = 10;
    fEnergyRange.Set(2395.0, 2520.0);
}

void TRestSignalToHitsProcess::LoadConfig(std::string cfgFilename, std::string name)
{
    if (LoadConfigFromFile(cfgFilename, name))
        LoadDefaultConfig();

    // If the parameters have no value it tries to obtain it from detector setup

    if (fElectricField == PARAMETER_NOT_FOUND_DBL)
    {
        TRestDetectorSetup *detSetup = (TRestDetectorSetup *)this->GetDetectorSetup();
        if (detSetup != NULL)
        {
            fElectricField = detSetup->GetFieldInVPerCm();
            cout << "SignalToHitsProcess : Obtainning electric field from detector "
                    "setup : "
                 << fElectricField << " V/cm" << endl;
        }
    }
}

//______________________________________________________________________________
void TRestSignalToHitsProcess::Initialize()
{
    SetSectionName(this->ClassName());

    fHitsEvent = new TRestHitsEvent();
    fSignalEvent = new TRestSignalEvent();

    fInputEvent = fSignalEvent;
    fOutputEvent = fHitsEvent;

    fGas = NULL;
    fReadout = NULL;
}

//______________________________________________________________________________
void TRestSignalToHitsProcess::InitProcess()
{
    // Function to be executed once at the beginning of process
    // (before starting the process of the events)

    // Start by calling the InitProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::InitProcess();

    fGas = (TRestGas *)this->GetGasMetadata();
    if (fGas != NULL)
    {
        if (fGasPressure <= 0)
            fGasPressure = fGas->GetPressure();
        else
            fGas->SetPressure(fGasPressure);

        if (fDriftVelocity <= 0)
            fDriftVelocity = fGas->GetDriftVelocity(fElectricField) * cmTomm;
    }
    else
    {
        cout << "REST_WARNING. No TRestGas found in TRestRun." << endl;
    }

    fReadout = (TRestReadout *)this->GetReadoutMetadata();

    if (fReadout == NULL)
    {
        cout << "REST ERRORRRR : Readout has not been initialized" << endl;
        exit(-1);
    }
}

//______________________________________________________________________________
void TRestSignalToHitsProcess::BeginOfEventProcess() { fHitsEvent->Initialize(); }

//______________________________________________________________________________

Double_t TRestSignalToHitsProcess::fitFunction(Double_t *x, Double_t *par)
{
    Double_t peak1_value = par[0];
    Double_t peak2_value = par[1];                                                             // in V
    Double_t peak1_position = par[2];                                                          // in 4ms
    Double_t peak2_position = par[3];                                                          // in 4ms
    Double_t DriftLence = (par[3] - par[2]) * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004; // in cm
    Double_t peak_width1 = std::sqrt(2 * 0.00127 * DriftLence / 23.6305) / 23.6305 * 250;      // of SeF5 in 4ms
    Double_t peak_width2 = std::sqrt(2 * 0.00118 * DriftLence / 21.8923) / 21.8923 * 250;      // of SeF6 in 4ms
    Double_t peak1 = peak1_value / (std::sqrt(2 * 3.14159) * peak_width1) * std::exp(-0.5 * std::pow((x[0] - peak1_position) / peak_width1, 2));
    Double_t peak2 = peak2_value / (std::sqrt(2 * 3.14159) * peak_width2) * std::exp(-0.5 * std::pow((x[0] - peak2_position) / peak_width2, 2));
    return (peak1 + peak2);
}

Double_t TRestSignalToHitsProcess::fitFunction4Peaks(Double_t *x, Double_t *par)
{
    // 第一组双峰
    Double_t peak1_value = par[0];
    Double_t peak2_value = par[1];    // in V
    Double_t peak1_position = par[2]; // in 4ms
    Double_t peak2_position = par[3]; // in 4ms

    Double_t DriftLence1 = (peak2_position - peak1_position) * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004; // in cm
    Double_t peak_width1 = std::sqrt(2 * 0.00127 * DriftLence1 / 23.6305) / 23.6305 * 250;                      // of SeF5 in 4ms
    Double_t peak_width2 = std::sqrt(2 * 0.00118 * DriftLence1 / 21.8923) / 21.8923 * 250;                      // of SeF6 in 4ms

    Double_t peak1 = peak1_value / (std::sqrt(2 * 3.14159) * peak_width1) * std::exp(-0.5 * std::pow((x[0] - peak1_position) / peak_width1, 2));
    Double_t peak2 = peak2_value / (std::sqrt(2 * 3.14159) * peak_width2) * std::exp(-0.5 * std::pow((x[0] - peak2_position) / peak_width2, 2));

    // 第二组双峰
    Double_t peak3_value = par[4];
    Double_t peak4_value = par[5];    // in V
    Double_t peak3_position = par[6]; // in 4ms
    Double_t peak4_position = par[7]; // in 4ms

    Double_t DriftLence2 = (peak4_position - peak3_position) * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004; // in cm
    Double_t peak_width3 = std::sqrt(2 * 0.00127 * DriftLence2 / 23.6305) / 23.6305 * 250;                      // of SeF5 in 4ms
    Double_t peak_width4 = std::sqrt(2 * 0.00118 * DriftLence2 / 21.8923) / 21.8923 * 250;                      // of SeF6 in 4ms

    Double_t peak3 = peak3_value / (std::sqrt(2 * 3.14159) * peak_width3) * std::exp(-0.5 * std::pow((x[0] - peak3_position) / peak_width3, 2));
    Double_t peak4 = peak4_value / (std::sqrt(2 * 3.14159) * peak_width4) * std::exp(-0.5 * std::pow((x[0] - peak4_position) / peak_width4, 2));

    return (peak1 + peak2 + peak3 + peak4);
}

TRestEvent *TRestSignalToHitsProcess::ProcessEvent(TRestEvent *evInput)
{

    fSignalEvent = (TRestSignalEvent *)evInput;

    if (GetVerboseLevel() >= REST_Debug)
        fSignalEvent->PrintEvent();

    fHitsEvent->SetID(fSignalEvent->GetID());
    fHitsEvent->SetSubID(fSignalEvent->GetSubID());
    fHitsEvent->SetTimeStamp(fSignalEvent->GetTimeStamp());
    fHitsEvent->SetSubEventTag(fSignalEvent->GetSubEventTag());

    Int_t numberOfSignals = fSignalEvent->GetNumberOfSignals();
    Int_t planeID, readoutChannel = -1, readoutModule;
    Double_t totalEnergy = 0;

    Double_t initialt1 = 0;
    Double_t initialt2 = 0;
    Double_t initialA1 = 0;
    Double_t initialA2 = 0;

    Double_t initialt3 = 0;
    Double_t initialt4 = 0;
    Double_t initialA3 = 0;
    Double_t initialA4 = 0;

    Int_t countofchannel = 0;

    for (int n = 0; n < numberOfSignals; n++)
    {
        // cout << "Signal Number :" << n << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
        TRestSignal *sgnl = fSignalEvent->GetSignal(n);
        Int_t signalID = sgnl->GetSignalID();
        Double_t Max_value = sgnl->GetMaxPeakValue();
        Double_t Pedestal = 0;
        // for (int i = 0; i < 100; i++)
        // {
        //     Pedestal += sgnl->GetData(i) / 100;
        // }
        // Double_t Maxwithoutbaseline = Max_value - Pedestal;
        // if (Maxwithoutbaseline < 7.5)
        //     continue;
        // countofchannel += 1;
        // cout << "Maxwithoutbaseline = " << Maxwithoutbaseline << endl;
        // Int_t maxIndex = sgnl->GetMaxIndex(0, sgnl->GetNumberOfPoints());
        // Int_t start = maxIndex - 500;
        // Int_t end = maxIndex + 1000;
        // if (start < 0)
        //     start = 0;
        // if (end > sgnl->GetNumberOfPoints())
        //     end = sgnl->GetNumberOfPoints();
        // TRestSignal *newSignal = new TRestSignal();
        // for (int i = start; i < end; i++)
        // for (int i = 0; i < 5000; i++)
        // {
        //     newSignal->AddPoint(i, sgnl->GetData(i));
        // }
        Int_t nBins = sgnl->GetNumberOfPoints();
        // cout << "points = " << nBins << endl;

        // TRestFFT *fft = new TRestFFT();

        // fft->SetNfft(nBins);

        // fft->ForwardSignalFFT(newSignal, 0, 0);
        // double samplingRate = 250;   // 4ms 采样间隔，对应采样率 250Hz
        // double maxFrequency = 10000; // 设定要关注的最大频率，比如100Hz

        // int cutFrequencyHz = 50;                                       // 静态截止频率(Hz)
        // int cutFrequencyIndex = cutFrequencyHz * nBins / samplingRate; // 转换为索引
        // fft->ApplyLowPassFilter(cutFrequencyIndex);

        // fft->BackwardFFT();

        // fft->GetSignal(newSignal);
        // delete fft;
        // std::vector<double> time;
        // std::vector<double> signal;

        // for (int i = 0; i < nBins; i++)
        // {
        //     time.push_back(i * (1.0 / samplingRate)); // 时间轴
        //     signal.push_back(newSignal->GetData(i));  // 滤波后的信号值
        // }

        if (GetVerboseLevel() >= REST_Debug)
            cout << "Searching readout coordinates for signal ID : " << signalID << endl;

        fReadout->GetPlaneModuleChannel(signalID, planeID, readoutModule, readoutChannel);

        if (readoutChannel == -1)
        {
            cout << "REST Warning : Readout channel not found for daq ID : " << signalID << endl;
            continue;
        }
        /////////////////////////////////////////////////////////////////////////

        TRestReadoutPlane *plane = fReadout->GetReadoutPlaneWithID(planeID);

        // For the moment this will only be valid for a TPC with its axis (field
        // direction) being in z
        Double_t fieldZDirection = plane->GetPlaneVector().Z();
        Double_t zPosition = plane->GetPosition().Z();
        Double_t x = plane->GetX(readoutModule, readoutChannel);
        Double_t y = plane->GetY(readoutModule, readoutChannel);

        if (fSignalToHitMethod == "onlyIntegral")
        {
            // cout << "signal number :" << n << endl;
            Double_t time = sgnl->GetMaxPeakTime();

            Double_t distanceToPlane = time * fDriftVelocity / 10;

            Double_t z = zPosition + fieldZDirection * distanceToPlane;
            Double_t energy = 0;
            for (int i = 0; i < nBins; i++)
            {
                energy += (sgnl->GetData(i)); //
            }
            totalEnergy += energy;
            fHitsEvent->AddHit(x, y, z, energy, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);
            // cout<<"pixel energy = "<<energy<<endl;
        }
        else if (fSignalToHitMethod == "fitting")
        {
            std::vector<double> Sgl(nBins);
            for (int i = 0; i < nBins; i++)
            {
                Sgl[i] = sgnl->GetData(i); // mV
            }
            TH1D *hist1 = new TH1D("hist1", "hist1", Sgl.size(), 0, Sgl.size());
            for (int i = 0; i < Sgl.size(); i++)
            {
                hist1->SetBinContent(i + 1, Sgl[i]);
            }

            std::vector<double> originsgl(nBins);
            for (int i = 0; i < nBins; i++)
            {
                originsgl[i] = (sgnl->GetData(i) - Pedestal); // mV
            }
            TH1D *hist2 = new TH1D("hist2", " ", originsgl.size(), 0, originsgl.size());
            for (int i = 0; i < originsgl.size(); i++)
            {
                hist2->SetBinContent(i + 1, originsgl[i]);
            }

            // ================== 寻峰参数设置 ==================
            const int fFittingWindow = 5;              // 峰搜索窗口
            const double thresholdPercentage = 0.2;    // 相对阈值 (相对最大峰)
            const int minPeakSeparation = 10;          // 最小峰间隔 (bin)
            const double mergeThresholdFraction = 0.1; // 相对强度差阈值 (合并用)

            // ================== 寻找峰 ==================
            std::vector<int> peakIndices;
            std::vector<double> peakValues;

            for (int k = 0; k < nBins; k++)
            {
                bool isPeak = true;
                for (int offset = 1; offset <= fFittingWindow; ++offset)
                {
                    if (k - offset < 0 || k + offset >= nBins)
                        continue;                                             // 边界保护
                    if (Sgl[k] < Sgl[k - offset] || Sgl[k] < Sgl[k + offset]) // 改成 < 避免漏检测
                    {
                        isPeak = false;
                        break;
                    }
                }
                if (isPeak)
                {
                    peakIndices.push_back(k);
                    peakValues.push_back(Sgl[k]);
                }
            }

            // ================== 阈值筛选 ==================
            
                double maxPeakValue = *std::max_element(peakValues.begin(), peakValues.end());
                for (int i = (int)peakValues.size() - 1; i >= 0; i--)
                {
                    if (peakValues[i] < thresholdPercentage * maxPeakValue)
                    {
                        peakValues.erase(peakValues.begin() + i);
                        peakIndices.erase(peakIndices.begin() + i);
                    }
                }
            

            // ================== 合并过近的峰 ==================
            int i = 0;
            while (i < (int)peakIndices.size() - 1)
            {
                int dist = peakIndices[i + 1] - peakIndices[i];
                double diff = std::abs(peakValues[i] - peakValues[i + 1]);
                double mergeThreshold = mergeThresholdFraction * (*std::max_element(peakValues.begin(), peakValues.end()));

                if (dist < minPeakSeparation && diff < mergeThreshold)
                {
                    if (peakValues[i] >= peakValues[i + 1])
                    {
                        peakValues.erase(peakValues.begin() + i + 1);
                        peakIndices.erase(peakIndices.begin() + i + 1);
                    }
                    else
                    {
                        peakValues.erase(peakValues.begin() + i);
                        peakIndices.erase(peakIndices.begin() + i);
                    }
                }
                else
                {
                    ++i;
                }
            }

            // ================== 打印结果 ==================
            // cout << "Detected peak number = " << peakIndices.size() << endl;
            // for (size_t j = 0; j < peakIndices.size(); ++j)
            // {
            //     cout << "Peak " << j + 1
            //          << " : position = " << peakIndices[j]
            //          << ", value = " << peakValues[j] << endl;
            // }

            // ================== 构建时间轴 ==================
            // std::vector<double> t1(nBins);
            // for (int i = 0; i < nBins; ++i)
            //     t1[i] = i * 4e-3; // 4ms间隔

            // ================== 绘图 ==================
            // TCanvas *c3 = new TCanvas(Form("c3_%d", n), Form("c3_%d", n), 800, 600);
            // TGraph *graph3 = new TGraph(nBins, &t1[0], &Sgl[0]);
            // graph3->SetTitle("Processed Signal with Peaks");
            // graph3->GetXaxis()->SetTitle("Time (s)");
            // graph3->GetYaxis()->SetTitle("Voltage (V)");
            // graph3->SetLineColor(kBlue);
            // graph3->Draw("AL");

            // // 标注所有峰值
            // for (size_t i = 0; i < peakIndices.size(); ++i)
            // {
            //     TMarker *marker = new TMarker(t1[peakIndices[i]], Sgl[peakIndices[i]], 20);
            //     marker->SetMarkerColor(kRed);
            //     marker->SetMarkerStyle(20);
            //     marker->Draw("SAME");
            // }

            // // 保存图片
            // c3->Draw();
            // c3->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/Signalwithpeak_%d.png", n));

            if (peakIndices.size() == 2)
            {

                initialt1 = peakIndices[0]; // in 4ms
                initialt2 = peakIndices[1]; // in 4ms
                initialA1 = peakValues[0];  // in mV
                initialA2 = peakValues[1];  // in mV

                TF1 *fitFunc = new TF1("fitFunc", TRestSignalToHitsProcess::fitFunction, 0, 5000, 4);
                TF1 *responseFunction = new TF1("responseFunction", "(1.0 - 1.0 / (TMath::Exp(x / 1e-05) + 1.0)) * TMath::Exp(-x / 300)", 0, 5000);

                TF1Convolution *convolution = new TF1Convolution(fitFunc, responseFunction, 0, 5000, true);
                TF1 *convolutedFunction = new TF1("convolutedFunction", *convolution, 0, 5000, convolution->GetNpar());
                convolutedFunction->SetNpx(100000);

                convolutedFunction->SetParameter(0, initialA1);

                convolutedFunction->SetParameter(1, (initialA2 - initialA1));

                convolutedFunction->SetParameter(2, initialt1);
                convolutedFunction->SetParameter(3, initialt2);
                convolutedFunction->SetParName(0, "A1(mV)");
                convolutedFunction->SetParName(1, "A2-A1(mV)");
                convolutedFunction->SetParName(2, "T1");
                convolutedFunction->SetParName(3, "T2");
                // TFitResultPtr r = hist2->Fit("convolutedFunction", "QS0W");
                // draw
                // TCanvas *c3 = new TCanvas("c3", "Fit Result", 800, 600);
                // hist2->Draw();
                // gStyle->SetOptFit(1);
                // convolutedFunction->SetLineColor(kRed);
                // convolutedFunction->Draw("same");
                // c3->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/Signalwithfit_%d.png", n));
                // delete c3;

                Double_t chargeVal = (convolutedFunction->GetParameter(0) + convolutedFunction->GetParameter(1));       // in mV
                Double_t driftTimeDiffVal = convolutedFunction->GetParameter(3) - convolutedFunction->GetParameter(2);  // in 4ms
                Double_t distanceToPlane = driftTimeDiffVal * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004 * cmTomm; // in mm
                Double_t z = zPosition + fieldZDirection * distanceToPlane;
                Double_t energy = chargeVal / 221 / 1.602176634 * 1e04 * 24.8 / 1000; // mV->keV

                fHitsEvent->AddHit(x, y, z, energy, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);
                delete hist1;
                delete hist2;
                delete convolutedFunction;
            }
            else if (peakIndices.size() == 3)
            {
                initialt1 = peakIndices[0]; // in 4ms
                initialt2 = peakIndices[1]; // in 4ms
                initialt3 = peakIndices[1]; // in 4ms
                initialt4 = peakIndices[2]; // in 4ms
                initialA1 = peakValues[0];  // in mV
                initialA2 = peakValues[1];  // in mV
                initialA3 = peakValues[1];  // in mV
                initialA4 = peakValues[2];  // in mV

                TF1 *fitFunc = new TF1("fitFunc", TRestSignalToHitsProcess::fitFunction4Peaks, 0, 5000, 8);
                TF1 *responseFunction = new TF1("responseFunction", "(1.0 - 1.0 / (TMath::Exp(x / 1e-05) + 1.0)) * TMath::Exp(-x / 300)", 0, 5000);

                TF1Convolution *convolution = new TF1Convolution(fitFunc, responseFunction, 0, 5000, true);
                TF1 *convolutedFunction = new TF1("convolutedFunction", *convolution, 0, 5000, convolution->GetNpar());
                convolutedFunction->SetNpx(100000);

                convolutedFunction->SetParameter(0, initialA1);
                convolutedFunction->SetParameter(1, (initialA2 - initialA1));
                convolutedFunction->SetParameter(2, initialt1);
                convolutedFunction->SetParameter(3, initialt2);
                convolutedFunction->SetParameter(4, initialA3);
                convolutedFunction->SetParameter(5, (initialA4 - initialA3));
                convolutedFunction->SetParameter(6, initialt3);
                convolutedFunction->SetParameter(7, initialt4);
                convolutedFunction->SetParName(0, "A1(mV)");
                convolutedFunction->SetParName(1, "A2-A1(mV)");
                convolutedFunction->SetParName(2, "T1");
                convolutedFunction->SetParName(3, "T2");
                convolutedFunction->SetParName(4, "A3(mV)");
                convolutedFunction->SetParName(5, "A4-A3(mV)");
                convolutedFunction->SetParName(6, "T3");
                convolutedFunction->SetParName(7, "T4");
                // TFitResultPtr r = hist2->Fit("convolutedFunction", "QS0W");

                // draw
                //  TCanvas *c3 = new TCanvas("c3", "Fit Result", 800, 600);
                //  hist2->Draw();
                //  gStyle->SetOptFit(1);
                //  convolutedFunction->SetLineColor(kRed);
                //  convolutedFunction->Draw("same");
                //  c3->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/Signalwithfit_%d.png", n));
                //  delete c3;

                Double_t chargeVal = (convolutedFunction->GetParameter(0) + convolutedFunction->GetParameter(1));       // in mV
                Double_t driftTimeDiffVal = convolutedFunction->GetParameter(3) - convolutedFunction->GetParameter(2);  // in 4ms
                Double_t distanceToPlane = driftTimeDiffVal * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004 * cmTomm; // in mm
                Double_t z_1 = zPosition + fieldZDirection * distanceToPlane;
                Double_t energy_1 = chargeVal / 221 / 1.602176634 * 1e04 * 24.8 / 1000; // V->keV

                fHitsEvent->AddHit(x, y, z_1, energy_1, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);

                chargeVal = (convolutedFunction->GetParameter(4) + convolutedFunction->GetParameter(5));       // in mV
                driftTimeDiffVal = convolutedFunction->GetParameter(7) - convolutedFunction->GetParameter(6);  // in 4ms
                distanceToPlane = driftTimeDiffVal * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004 * cmTomm; // in mm
                Double_t z_2 = zPosition + fieldZDirection * distanceToPlane;
                Double_t energy_2 = chargeVal / 221 / 1.602176634 * 1e04 * 24.8 / 1000; // V->keV

                fHitsEvent->AddHit(x, y, z_2, energy_2, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);

                delete hist1;
                delete hist2;
                delete convolutedFunction;
            }
            else if (peakIndices.size() == 4)
            {
                initialt1 = peakIndices[0]; // in 4ms
                initialt2 = peakIndices[1]; // in 4ms
                initialt3 = peakIndices[2]; // in 4ms
                initialt4 = peakIndices[3]; // in 4ms
                initialA1 = peakValues[0];  // in mV
                initialA2 = peakValues[1];  // in mV
                initialA3 = peakValues[2];  // in mV
                initialA4 = peakValues[3];  // in mV

                // std::cout << "Initial peak positions (in 4ms): "
                //           << "t1 = " << initialt1 << ", "
                //           << "t2 = " << initialt2 << ", "
                //           << "t3 = " << initialt3 << ", "
                //           << "t4 = " << initialt4 << std::endl;

                // std::cout << "Initial peak amplitudes (in mV): "
                //           << "A1 = " << initialA1 << ", "
                //           << "A2 = " << initialA2 << ", "
                //           << "A3 = " << initialA3 << ", "
                //           << "A4 = " << initialA4 << std::endl;

                TF1 *fitFunc = new TF1("fitFunc", TRestSignalToHitsProcess::fitFunction4Peaks, 0, 5000, 8);
                TF1 *responseFunction = new TF1("responseFunction", "(1.0 - 1.0 / (TMath::Exp(x / 1e-05) + 1.0)) * TMath::Exp(-x / 300)", 0, 5000);

                TF1Convolution *convolution = new TF1Convolution(fitFunc, responseFunction, 0, 5000, true);
                TF1 *convolutedFunction = new TF1("convolutedFunction", *convolution, 0, 5000, convolution->GetNpar());
                convolutedFunction->SetNpx(100000);

                convolutedFunction->SetParameter(0, initialA1);
                convolutedFunction->SetParameter(1, (initialA2 - initialA1));
                convolutedFunction->SetParameter(2, initialt1);
                convolutedFunction->SetParameter(3, initialt2);
                convolutedFunction->SetParameter(4, initialA3);
                convolutedFunction->SetParameter(5, (initialA4 - initialA3));
                convolutedFunction->SetParameter(6, initialt3);
                convolutedFunction->SetParameter(7, initialt4);
                convolutedFunction->SetParName(0, "A1(mV)");
                convolutedFunction->SetParName(1, "A2-A1(mV)");
                convolutedFunction->SetParName(2, "T1");
                convolutedFunction->SetParName(3, "T2");
                convolutedFunction->SetParName(4, "A3(mV)");
                convolutedFunction->SetParName(5, "A4-A3(mV)");
                convolutedFunction->SetParName(6, "T3");
                convolutedFunction->SetParName(7, "T4");
                // TFitResultPtr r = hist2->Fit("convolutedFunction", "QS0W");

                // draw
                // TCanvas *c3 = new TCanvas("c3", "Fit Result", 800, 600);
                // hist2->Draw();
                // gStyle->SetOptFit(1);
                // convolutedFunction->SetLineColor(kRed);
                // convolutedFunction->Draw("same");
                // c3->SaveAs(Form("/home/rest/rest_workspace/TEST/signalshaping/Signalwithfit_%d.png", n));
                // delete c3;

                Double_t chargeVal = (convolutedFunction->GetParameter(0) + convolutedFunction->GetParameter(1));       // in mV
                Double_t driftTimeDiffVal = convolutedFunction->GetParameter(3) - convolutedFunction->GetParameter(2);  // in 4ms
                Double_t distanceToPlane = driftTimeDiffVal * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004 * cmTomm; // in mm
                Double_t z_1 = zPosition + fieldZDirection * distanceToPlane;
                Double_t energy_1 = chargeVal / 221 / 1.602176634 * 1e04 * 24.8 / 1000; // V->keV

                fHitsEvent->AddHit(x, y, z_1, energy_1, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);
                cout<<"z_1 = "<<z_1<<endl;
                cout<<"energy_1 = "<<energy_1<<endl;

                chargeVal = (convolutedFunction->GetParameter(4) + convolutedFunction->GetParameter(5));       // in mV
                driftTimeDiffVal = convolutedFunction->GetParameter(7) - convolutedFunction->GetParameter(6);  // in 4ms
                distanceToPlane = driftTimeDiffVal * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004 * cmTomm; // in mm
                Double_t z_2 = zPosition + fieldZDirection * distanceToPlane;
                Double_t energy_2 = chargeVal / 221 / 1.602176634 * 1e04 * 24.8 / 1000; // V->keV

                fHitsEvent->AddHit(x, y, z_2, energy_2, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);
                cout<<"z_2 = "<<z_2<<endl;
                cout<<"energy_2 = "<<energy_2<<endl;

                delete hist1;
                delete hist2;
                delete convolutedFunction;
            }
            else
            {
                initialA1 = maxPeakValue / 3;
                initialA2 = maxPeakValue;

                initialt2 = sgnl->GetMaxPeakTime();;

                TF1 *fitFunc = new TF1("fitFunc", TRestSignalToHitsProcess::fitFunction, 0, 5000, 4);
                TF1 *responseFunction = new TF1("responseFunction", "(1.0 - 1.0 / (TMath::Exp(x / 1e-05) + 1.0)) * TMath::Exp(-x / 300)", 0, 5000);

                TF1Convolution *convolution = new TF1Convolution(fitFunc, responseFunction, 0, 5000, true);
                TF1 *convolutedFunction = new TF1("convolutedFunction", *convolution, 0, 5000, convolution->GetNpar());
                convolutedFunction->SetNpx(100000);

                convolutedFunction->SetParameter(0, initialA1);

                convolutedFunction->SetParameter(1, (initialA2 - initialA1));

                convolutedFunction->SetParameter(2, initialt1);
                convolutedFunction->SetParameter(3, initialt2);
                convolutedFunction->SetParName(0, "A1(mV)");
                convolutedFunction->SetParName(1, "A2-A1(mV)");
                convolutedFunction->SetParName(2, "T1");
                convolutedFunction->SetParName(3, "T2");
                // TFitResultPtr r = hist2->Fit("convolutedFunction", "QS0W");

                Double_t chargeVal = (convolutedFunction->GetParameter(0) + convolutedFunction->GetParameter(1));       // in mV
                Double_t driftTimeDiffVal = convolutedFunction->GetParameter(3) - convolutedFunction->GetParameter(2);  // in 4ms
                Double_t distanceToPlane = driftTimeDiffVal * 21.8923 * 23.6305 / (23.6305 - 21.8923) * 0.004 * cmTomm; // in mm
                Double_t z = zPosition + fieldZDirection * distanceToPlane;
                Double_t energy = chargeVal / 221 / 1.602176634 * 1e04 * 24.8 / 1000; // mV->keV

                fHitsEvent->AddHit(x, y, z, energy, 0, 0, (Short_t)readoutModule, (Short_t)readoutChannel);
                delete hist1;
                delete hist2;
                delete convolutedFunction;
            }
        }
    }
    // cout << "countofchannel = " << countofchannel << endl;
    // cout << "totalEnergy = " << totalEnergy << endl;
    // std::ofstream zOutFile("totalE_value_withoutcut_20241204.txt", std::ios::app);
    // zOutFile << totalEnergy << std::endl;
    // zOutFile.close();
    if (this->GetVerboseLevel() >= REST_Debug)
    {
        cout << "TRestSignalToHitsProcess. Hits added : " << fHitsEvent->GetNumberOfHits() << endl;
        cout << "TRestSignalToHitsProcess. Hits total energy : " << fHitsEvent->GetEnergy() << endl;
    }

    if (this->GetVerboseLevel() >= REST_Debug)
    {
        fHitsEvent->PrintEvent(300);
        GetChar();
    }

    if (fHitsEvent->GetNumberOfHits() <= 0)
    {
        if (this->GetVerboseLevel() == REST_Debug)
            cout << "TRestSignalToHitsProcess: Number of hits: " << fHitsEvent->GetNumberOfHits() << " !" << endl;
        return NULL;
    }

    if (fHitsEvent->GetTotalEnergy() < fEnergyRange.X() || fHitsEvent->GetTotalEnergy() > fEnergyRange.Y())
    {
        if (this->GetVerboseLevel() == REST_Debug)
            cout << "TRestSignalToHitsProcess: Energy over range: " << fHitsEvent->GetTotalEnergy() << "keV !"
                 << endl;
        return NULL;
    }

    return fHitsEvent;
}

//______________________________________________________________________________
void TRestSignalToHitsProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void TRestSignalToHitsProcess::EndProcess()
{
    // Function to be executed once at the end of the process
    // (after all events have been processed)

    // Start by calling the EndProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::EndProcess();
}

//______________________________________________________________________________
void TRestSignalToHitsProcess::InitFromConfigFile()
{
    fElectricField = GetDblParameterWithUnits("electricField");
    fGasPressure = StringToDouble(GetParameter("gasPressure", "-1"));
    fFittingWindow = StringToDouble(GetParameter("fittingWindow", "5"));
    fDriftVelocity = StringToDouble(GetParameter("driftVelocity", "0")) * cmTomm;
    fSignalToHitMethod = GetParameter("method", "all");
    fEnergyRange = StringTo2DVector(GetParameter("EnergyRange", "(2395,2520)")); // keV
}
