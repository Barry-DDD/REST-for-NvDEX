///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestHitsAnalysisProcess.cxx
///
///
///             First implementation of hits analysis process into REST_v2
///             Date : may/2016
///             Author : J. Galan
///
///_______________________________________________________________________________

#include "TRestHitsAnalysisProcess.h"
using namespace std;

ClassImp(TRestHitsAnalysisProcess)
    //______________________________________________________________________________
    TRestHitsAnalysisProcess::TRestHitsAnalysisProcess() {
    Initialize();
}

//______________________________________________________________________________
TRestHitsAnalysisProcess::TRestHitsAnalysisProcess(char* cfgFileName) {
    Initialize();

    if (LoadConfigFromFile(cfgFileName) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
TRestHitsAnalysisProcess::~TRestHitsAnalysisProcess() {
    delete fInputHitsEvent;
    delete fOutputHitsEvent;
}

void TRestHitsAnalysisProcess::LoadDefaultConfig() { SetTitle("Default config"); }

//______________________________________________________________________________
void TRestHitsAnalysisProcess::Initialize() {
    SetSectionName(this->ClassName());

    fInputHitsEvent = new TRestHitsEvent();
    fOutputHitsEvent = new TRestHitsEvent();

    fOutputEvent = fOutputHitsEvent;
    fInputEvent = fInputHitsEvent;

    fPrismFiducial = false;
    fCylinderFiducial = false;
}

void TRestHitsAnalysisProcess::LoadConfig(std::string cfgFilename, std::string name) {
    if (LoadConfigFromFile(cfgFilename, name) == -1) LoadDefaultConfig();
}

//______________________________________________________________________________
void TRestHitsAnalysisProcess::InitProcess() { TRestEventProcess::ReadObservables(); }

//______________________________________________________________________________
void TRestHitsAnalysisProcess::BeginOfEventProcess() { fOutputHitsEvent->Initialize(); }

//______________________________________________________________________________
TRestEvent* TRestHitsAnalysisProcess::ProcessEvent(TRestEvent* evInput) {
    fInputHitsEvent = (TRestHitsEvent*)evInput;

    string obsName;

    TRestHits* hits = fInputHitsEvent->GetHits();

    Double_t x_min = 1E10;
    Double_t x_max = -1E10;
    Double_t y_min = 1E10;
    Double_t y_max = -1E10;
    Double_t z_min = 1E10;
    Double_t z_max = -1E10;
    for (int n = 0; n < hits->GetNumberOfHits(); n++) {
        Double_t eDep = hits->GetEnergy(n);

        Double_t x = hits->GetX(n);
        Double_t y = hits->GetY(n);
        Double_t z = hits->GetZ(n);
        Double_t t = hits->GetTime(n);
        Short_t readoutModule = hits->GetModule(n);
        Short_t readoutChannel = hits->GetChannel(n);
        fOutputHitsEvent->AddHit(x, y, z, eDep, 0, t, readoutModule, readoutChannel);
        if( x_min > x ) x_min = x;
        if( x_max < x ) x_max = x;
        if( y_min > y ) y_min = y;
        if( y_max < y ) y_max = y;
        if( z_min > z ) z_min = z;
        if( z_max < z ) z_max = z;
    }

    Double_t x_range = x_max - x_min;
    obsName = "x_min";
    SetObservableValue(obsName, x_min);
    obsName = "x_max";
    SetObservableValue(obsName, x_max);
    obsName = "x_range";
    SetObservableValue(obsName, x_range);

    Double_t y_range = y_max - y_min;
    obsName = "y_min";
    SetObservableValue(obsName, y_min);
    obsName = "y_max";
    SetObservableValue(obsName, y_max);
    obsName = "y_range";
    SetObservableValue(obsName, y_range);
    
    Double_t z_range = z_max - z_min;
    obsName = "z_min";
    SetObservableValue(obsName, z_min);
    obsName = "z_max";
    SetObservableValue(obsName, z_max);
    obsName = "z_range";
    SetObservableValue(obsName, z_range);

    if (fOutputHitsEvent->GetNumberOfHits() == 0) return NULL;

    Double_t energy = fOutputHitsEvent->GetEnergy();
    TVector3 meanPosition = fOutputHitsEvent->GetMeanPosition();
    Double_t sigmaX = fOutputHitsEvent->GetSigmaX();
    Double_t sigmaY = fOutputHitsEvent->GetSigmaY();
    Double_t sigmaXY2 = fOutputHitsEvent->GetSigmaXY2();
    Double_t sigmaZ2 = fOutputHitsEvent->GetSigmaZ2();
    Double_t skewXY = fOutputHitsEvent->GetSkewXY();
    Double_t skewZ = fOutputHitsEvent->GetSkewZ();
    Double_t energyX = fOutputHitsEvent->GetEnergyX();
    Double_t energyY = fOutputHitsEvent->GetEnergyY();
    Double_t maxEnergy = fOutputHitsEvent->GetMaximumHitEnergy();
    Double_t minEnergy = fOutputHitsEvent->GetMinimumHitEnergy();
    Double_t meanEnergy = fOutputHitsEvent->GetMeanHitEnergy();
    Int_t nHits = fOutputHitsEvent->GetNumberOfHits();
    Int_t nHitsX = fOutputHitsEvent->GetNumberOfHitsX();
    Int_t nHitsY = fOutputHitsEvent->GetNumberOfHitsY();

    obsName = "nHits";
    SetObservableValue(obsName, nHits);

    obsName = "nHitsX";
    SetObservableValue(obsName, nHitsX);

    obsName = "nHitsY";
    SetObservableValue(obsName, nHitsY);

    obsName = "balanceXYnHits";
    SetObservableValue(obsName, (nHitsX - nHitsY) / (nHitsX + nHitsY));

    obsName = "nHitsSizeXY";
    if ((nHits == nHitsX) || (nHits == nHitsY))
        SetObservableValue(obsName, nHits);
    else
        SetObservableValue(obsName, TMath::Sqrt(nHitsX * nHitsX + nHitsY * nHitsY));

    // Checking hits inside fiducial cylinder
    if (fCylinderFiducial) {
        TVector3 meanPositionInCylinder =
            fOutputHitsEvent->GetMeanPositionInCylinder(fFid_x0, fFid_x1, fFid_R);

        Int_t isInsideCylinder = 0;
        if (fOutputHitsEvent->isHitsEventInsideCylinder(fFid_x0, fFid_x1, fFid_R)) isInsideCylinder = 1;

        Int_t nCylVol = fOutputHitsEvent->GetNumberOfHitsInsideCylinder(fFid_x0, fFid_x1, fFid_R);

        Double_t enCylVol = fOutputHitsEvent->GetEnergyInCylinder(fFid_x0, fFid_x1, fFid_R);

        obsName = "isInsideCylindricalVolume";
        SetObservableValue(obsName, isInsideCylinder);

        obsName = "nInsideCylindricalVolume";
        SetObservableValue(obsName, nCylVol);

        obsName = "energyInsideCylindricalVolume";
        SetObservableValue(obsName, enCylVol);

        // mean positions
        obsName = "xMeanInCylinder";
        SetObservableValue(obsName, meanPositionInCylinder.X());

        obsName = "yMeanInCylinder";
        SetObservableValue(obsName, meanPositionInCylinder.Y());

        obsName = "zMeanInCylinder";
        SetObservableValue(obsName, meanPositionInCylinder.Z());
    }

    // Checking hits inside fiducial prism
    if (fPrismFiducial) {
        TVector3 meanPositionInPrism =
            fOutputHitsEvent->GetMeanPositionInPrism(fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta);

        Int_t isInsidePrism = 0;
        if (fOutputHitsEvent->isHitsEventInsidePrism(fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta))
            isInsidePrism = 1;

        Int_t nPrismVol =
            fOutputHitsEvent->GetNumberOfHitsInsidePrism(fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta);

        Double_t enPrismVol =
            fOutputHitsEvent->GetEnergyInPrism(fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta);

        obsName = "isInsidePrismVolume";
        SetObservableValue(obsName, isInsidePrism);

        obsName = "nInsidePrismVolume";
        SetObservableValue(obsName, nPrismVol);

        obsName = "energyInsidePrismVolume";
        SetObservableValue(obsName, enPrismVol);

        // Mean Positions

        obsName = "xMeanInPrism";
        SetObservableValue(obsName, meanPositionInPrism.X());

        obsName = "yMeanInPrism";
        SetObservableValue(obsName, meanPositionInPrism.Y());

        obsName = "zMeanInPrism";
        SetObservableValue(obsName, meanPositionInPrism.Z());
    }

    ///////////////////////////////////////

    if (fCylinderFiducial) {
        // Adding distances to cylinder wall
        Double_t dToCylWall =
            fOutputHitsEvent->GetClosestHitInsideDistanceToCylinderWall(fFid_x0, fFid_x1, fFid_R);
        Double_t dToCylTop =
            fOutputHitsEvent->GetClosestHitInsideDistanceToCylinderTop(fFid_x0, fFid_x1, fFid_R);
        Double_t dToCylBottom =
            fOutputHitsEvent->GetClosestHitInsideDistanceToCylinderBottom(fFid_x0, fFid_x1, fFid_R);

        obsName = "distanceToCylinderWall";
        SetObservableValue(obsName, dToCylWall);
        obsName = "distanceToCylinderTop";
        SetObservableValue(obsName, dToCylTop);
        obsName = "distanceToCylinderBottom";
        SetObservableValue(obsName, dToCylBottom);
    }

    if (fPrismFiducial) {
        // Adding distances to prism wall
        Double_t dToPrismWall = fOutputHitsEvent->GetClosestHitInsideDistanceToPrismWall(
            fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta);
        Double_t dToPrismTop = fOutputHitsEvent->GetClosestHitInsideDistanceToPrismTop(
            fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta);
        Double_t dToPrismBottom = fOutputHitsEvent->GetClosestHitInsideDistanceToPrismBottom(
            fFid_x0, fFid_x1, fFid_sX, fFid_sY, fFid_theta);

        obsName = "distanceToPrismWall";
        SetObservableValue(obsName, dToPrismWall);

        obsName = "distanceToPrismTop";
        SetObservableValue(obsName, dToPrismTop);

        obsName = "distanceToPrismBottom";
        SetObservableValue(obsName, dToPrismBottom);
    }

    ///////////////////////////////////////

    obsName = "energy";
    SetObservableValue(obsName, energy);
    obsName = "energyX";
    SetObservableValue(obsName, energyX);
    obsName = "energyY";
    SetObservableValue(obsName, energyY);
    obsName = "balanceXYenergy";
    SetObservableValue(obsName, (energyX - energyY) / (energyX + energyY));

    obsName = "maxHitEnergy";
    SetObservableValue(obsName, maxEnergy);
    obsName = "minHitEnergy";
    SetObservableValue(obsName, minEnergy);
    obsName = "meanHitEnergy";
    SetObservableValue(obsName, meanEnergy);
    obsName = "meanHitEnergyBalance";
    SetObservableValue(obsName, meanEnergy / energy);

    obsName = "xMean";
    SetObservableValue(obsName, meanPosition.X());

    obsName = "yMean";
    SetObservableValue(obsName, meanPosition.Y());

    obsName = "zMean";
    SetObservableValue(obsName, meanPosition.Z());
    obsName = "xy2Sigma";
    SetObservableValue(obsName, sigmaXY2);
    obsName = "xySigmaBalance";
    SetObservableValue(obsName, (sigmaX - sigmaY) / (sigmaX + sigmaY));

    obsName = "z2Sigma";
    SetObservableValue(obsName, sigmaZ2);

    obsName = "xySkew";
    SetObservableValue(obsName, skewXY);
    obsName = "zSkew";
    SetObservableValue(obsName, skewZ);

    if (GetVerboseLevel() >= REST_Extreme) {
        fOutputHitsEvent->PrintEvent(1000);
        GetChar();
    }

    return fOutputHitsEvent;
}

//______________________________________________________________________________
void TRestHitsAnalysisProcess::EndOfEventProcess() {}

//______________________________________________________________________________
void TRestHitsAnalysisProcess::EndProcess() {
    // Function to be executed once at the end of the process
    // (after all events have been processed)

    // Start by calling the EndProcess function of the abstract class.
    // Comment this if you don't want it.
    // TRestEventProcess::EndProcess();
}

//______________________________________________________________________________
void TRestHitsAnalysisProcess::InitFromConfigFile() {
    fFid_x0 = Get3DVectorParameterWithUnits("fiducial_x0", TVector3(0, 0, 0));
    fFid_x1 = Get3DVectorParameterWithUnits("fiducial_x1", TVector3(0, 0, 0));

    fFid_R = GetDblParameterWithUnits("fiducial_R", 1);
    fFid_sX = GetDblParameterWithUnits("fiducial_sX", 1);
    fFid_sY = GetDblParameterWithUnits("fiducial_sY", 1);
    fFid_theta = StringToDouble(GetParameter("fiducial_theta", "0"));

    if (GetParameter("cylinderFiducialization", "false") == "true") fCylinderFiducial = true;

    if (GetParameter("prismFiducialization", "false") == "true") fPrismFiducial = true;

    if (fCylinderFiducial) cout << "Cylinder fiducial active" << endl;
    if (fPrismFiducial) cout << "Prism fiducial active" << endl;
}
