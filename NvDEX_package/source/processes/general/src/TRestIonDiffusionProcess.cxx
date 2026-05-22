///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestIonDiffusionProcess.cxx
///
///
///             First implementation of Ion diffusion process into REST_v2
///             Date : Oct/2023
///             Author : H.Barry
///
///_______________________________________________________________________________

#include "TRestIonDiffusionProcess.h"
using namespace std;

ClassImp(TRestIonDiffusionProcess);

//______________________________________________________________________________
TRestIonDiffusionProcess::TRestIonDiffusionProcess() { Initialize(); }

//______________________________________________________________________________
TRestIonDiffusionProcess::TRestIonDiffusionProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName)) LoadDefaultConfig();
}

//______________________________________________________________________________
TRestIonDiffusionProcess::~TRestIonDiffusionProcess() {
    delete fOutputHitsEvent;
    delete fInputHitsEvent;
}

void TRestIonDiffusionProcess::LoadDefaultConfig() {
    SetTitle("Default config");

    fElectricField = 1000;
    fAttachment = 0;
    fGasPressure = 1;
}

//______________________________________________________________________________
void TRestIonDiffusionProcess::Initialize() {
    SetSectionName(this->ClassName());

    fElectricField = 0;
    fAttachment = 0;
    fGasPressure = 1;

    fTransDiffCoeff = 0;
    fLonglDiffCoeff = 0;
    fWvalue = 0;
    settingCoeff = 0;

    fOutputHitsEvent = new TRestHitsEvent();
    fInputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;

    fGas = NULL;
    fReadout = NULL;

    fRandom = new TRandom3(fSeed);
}

void TRestIonDiffusionProcess::LoadConfig(string cfgFilename, string name) {
    if (LoadConfigFromFile(cfgFilename, name)) LoadDefaultConfig();
}

//______________________________________________________________________________
void TRestIonDiffusionProcess::InitProcess() {
    fGas = (TRestGas*)GetGasMetadata();
    if (fGas == NULL) {
        cout << "REST WARNING : Gas has not been initialized" << endl;
    } else {
        if (fGasPressure <= 0)
            fGasPressure = fGas->GetPressure();
        else
            fGas->SetPressure(fGasPressure);

        if (fWvalue <= 0) fWvalue = fGas->GetWvalue();

        if (fLonglDiffCoeff <= 0)
            fLonglDiffCoeff = fGas->GetLongitudinalDiffusion(fElectricField);  // (cm)^1/2

        if (fTransDiffCoeff <= 0)
            fTransDiffCoeff = fGas->GetTransversalDiffusion(fElectricField);  // (cm)^1/2
    }

    if (settingCoeff) {
        Double_t coeff = settingCoeff / fTransDiffCoeff;
        fTransDiffCoeff = settingCoeff;
        fLonglDiffCoeff = fLonglDiffCoeff * coeff;
    }

    fReadout = (TRestReadout*)GetReadoutMetadata();
    if (fReadout == NULL) {
        cout << "REST ERRORRRR : Readout has not been initialized" << endl;
        exit(-1);
    }
}

//______________________________________________________________________________
TRestEvent* TRestIonDiffusionProcess::ProcessEvent(TRestEvent* evInput) {
    fInputHitsEvent = (TRestHitsEvent*)evInput;
    fOutputHitsEvent->SetEventInfo(fInputHitsEvent);

    Int_t nHits = fInputHitsEvent->GetNumberOfHits();
    if (nHits <= 0) return NULL;

    const Double_t Boltzmann_const = 1.380e-23;
    const Double_t elementary_charge = 1.602e-19;  // in C
    const Double_t temperature = 293.1500;         // in K
    const Double_t ionization_energy = 32;       // in eV
    const Double_t Fano = 0.19;                    // for Se at 10bar
    Double_t totalEnergy = 0 ;

    Double_t drift_mu0_neg[2] = {0.466, 0.503};  
    // Double_t drift_mu0_pos[6] = {0.466, 0.503, 0.546, 0.599, 0.663, 0.746}; // TODO
    Double_t drift_v_neg[2];
    Double_t Dt_neg[2];
    Double_t Dl_neg[2];
    Double_t distance_trans_neg[2];
    Double_t distance_long_neg[2];
    Double_t particle_select_neg;
    Int_t particle_type_rm_neg;

    for (int n = 0; n < nHits; n++) {
        TRestHits* hits = fInputHitsEvent->GetHits();
        Double_t eDep = hits->GetEnergy(n);
        Double_t x = hits->GetX(n);
        Double_t y = hits->GetY(n);
        Double_t z = hits->GetZ(n);
        Double_t t = hits->GetTime(n);
        // std::cout << "Input hit info - x: " << x << ", y: " << y << ", z: " << z
        //   << ", E: " << eDep << ", t: " << t << " ns" << std::endl;
        TRestReadoutPlane* plane = &(*fReadout)[0];
        Double_t L = plane->GetDistanceTo(x, y, z);
        Double_t mIon = eDep * 1000 / ionization_energy;
        Double_t sIon = sqrt(Fano * mIon);
        Int_t nIon = (int)fRandom->Gaus(mIon, sIon);
        if (nIon < 0)  
            nIon = 0;
        if (nIon == 0) continue;

        for (int k = 0; k < nIon; k++) {
            Double_t particle_select_neg = fRandom->Uniform(0.0, 1.0);
            
            if (particle_select_neg < 0.666)  // 66.6% SeF6; 33.3% SeF5;
            {
                particle_type_rm_neg = 0;
            } else {
                particle_type_rm_neg = 1;
            }
            for (int i = 0; i < 2; i++) {
                drift_v_neg[i] =
                    (1. / fGasPressure) * (temperature / 273) * drift_mu0_neg[i] * fElectricField;
                Dt_neg[i] =
                    (drift_mu0_neg[i] * Boltzmann_const * temperature) / (elementary_charge * fGasPressure);
                Dl_neg[i] =
                    (drift_mu0_neg[i] * Boltzmann_const * temperature) / (elementary_charge * fGasPressure);
                distance_trans_neg[i] = sqrt((2. * Dt_neg[i] * L / 10) / drift_v_neg[i]);
                distance_long_neg[i] = sqrt((2. * Dl_neg[i] * L / 10) / drift_v_neg[i]);
            }

            // nagetive Ion
            // gauss random
            Double_t xDiff = x + fRandom->Gaus(0, 10 * distance_trans_neg[particle_type_rm_neg]);
            Double_t yDiff = y + fRandom->Gaus(0, 10 * distance_trans_neg[particle_type_rm_neg]);
            Double_t zDiff = z + fRandom->Gaus(0, 10 * distance_long_neg[particle_type_rm_neg]);
            // fOutputHitsEvent->AddHit(xDiff, yDiff, zDiff,ionization_energy * REST_Units::keV / REST_Units::eV, particle_type_rm_neg);
            fOutputHitsEvent->AddHit(xDiff, yDiff, zDiff,ionization_energy * REST_Units::keV / REST_Units::eV,particle_type_rm_neg,t,-1, -1);
            
        }
        // totalEnergy+=ionization_energy*nIon/1000;
    }
    // cout<<"realE_value = "<<totalEnergy<<endl;
    // std::ofstream zOutFile("realE_value.txt", std::ios::app);
    // zOutFile << totalEnergy << std::endl;
    // zOutFile.close();
    return fOutputHitsEvent;
}

//______________________________________________________________________________
void TRestIonDiffusionProcess::EndProcess() {}

//______________________________________________________________________________
void TRestIonDiffusionProcess::InitFromConfigFile() {
    // TODO add pressure units
    fGasPressure = GetDblParameterWithUnits("gasPressure", -1.);
    fElectricField = GetDblParameterWithUnits("electricField", -1.);
    fWvalue = GetDblParameterWithUnits("Wvalue", (double)0) * REST_Units::eV;
    fAttachment = StringToDouble(GetParameter("attachment", "0"));
    fLonglDiffCoeff = StringToDouble(GetParameter("longitudinalDiffusionCoefficient", "-1"));
    if (fLonglDiffCoeff == -1)
        fLonglDiffCoeff = StringToDouble(GetParameter("longDiff", "-1"));
    else {
        warning << "longitudinalDiffusionCoeffient is now OBSOLETE! It will soon dissapear." << endl;
        warning << " Please use the shorter form of this parameter : longDiff" << endl;
    }

    fTransDiffCoeff = StringToDouble(GetParameter("transversalDiffusionCoefficient", "-1"));
    if (fTransDiffCoeff == -1)
        fTransDiffCoeff = StringToDouble(GetParameter("transDiff", "-1"));
    else {
        warning << "transversalDiffusionCoeffient is now OBSOLETE! It will soon dissapear." << endl;
        warning << " Please use the shorter form of this parameter : transDiff" << endl;
    }
    fMaxHits = StringToInteger(GetParameter("maxHits", "1000"));
    fSeed = StringToDouble(GetParameter("seed", "0"));
    fPoissonElectronExcitation = StringToBool(GetParameter("poissonElectronExcitation", "false"));
    fUnitElectronEnergy = StringToBool(GetParameter("unitElectronEnergy", "false"));
}
