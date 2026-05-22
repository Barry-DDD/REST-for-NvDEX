#ifndef RestCore_TRestParticleSource
#define RestCore_TRestParticleSource

#include <iostream>

#include <TString.h>
#include <TVector2.h>
#include <TVector3.h>
#include "TObject.h"
//
#include <TRestParticle.h>

class TRestParticleSource : public TRestParticle {
   protected:
    TString fAngularDistType;
    TString fEnergyDistType;
    TVector2 fEnergyRange;

    TString fSpectrumFilename;
    TString fSpectrumName;

    TString fAngularFilename;
    TString fAngularName;

    // -----------------------------
    // planeSector angular distribution parameters
    // -----------------------------
    TVector3 fPlaneNormal;  // plane normal
    TVector3 fPlaneRef;     // reference direction in the plane (phi=0)
    Double_t fPhiMin;       // radians (planeSector)
    Double_t fPhiMax;       // radians (planeSector)

    // -----------------------------
    // angle angular distribution parameters (NEW)
    // theta/phi范围定义 (球坐标) [radians]
    // -----------------------------
    Double_t fAngleThetaMin = 0.0;
    Double_t fAngleThetaMax = 0.0;
    Double_t fAnglePhiMin   = 0.0;
    Double_t fAnglePhiMax   = 0.0;

   public:
    // -----------------------------
    // Basic getters
    // -----------------------------
    TString GetParticle() const { return fParticleName; }
    TString GetAngularDistType() const { return fAngularDistType; }
    TVector3 GetDirection() const { return fDirection; }

    // -----------------------------
    // planeSector getters
    // -----------------------------
    TVector3 GetPlaneNormal() const { return fPlaneNormal; }
    TVector3 GetPlaneRef() const { return fPlaneRef; }
    Double_t GetPhiMin() const { return fPhiMin; }
    Double_t GetPhiMax() const { return fPhiMax; }

    // -----------------------------
    // angle getters (NEW)
    // -----------------------------
    Double_t GetAngleThetaMin() const { return fAngleThetaMin; }
    Double_t GetAngleThetaMax() const { return fAngleThetaMax; }
    Double_t GetAnglePhiMin() const { return fAnglePhiMin; }
    Double_t GetAnglePhiMax() const { return fAnglePhiMax; }

    // -----------------------------
    // Energy getters
    // -----------------------------
    TString GetEnergyDistType() const { return fEnergyDistType; }
    TVector2 GetEnergyRange() const { return fEnergyRange; }
    Double_t GetMinEnergy() const { return fEnergyRange.X(); }
    Double_t GetMaxEnergy() const { return fEnergyRange.Y(); }

    // -----------------------------
    // Spectrum getters
    // -----------------------------
    TString GetSpectrumFilename() const { return fSpectrumFilename; }
    TString GetSpectrumName() const { return fSpectrumName; }

    TString GetAngularFilename() const { return fAngularFilename; }
    TString GetAngularName() const { return fAngularName; }

    // -----------------------------
    // Setters
    // -----------------------------
    void SetAngularDistType(TString type) { fAngularDistType = type; }

    // planeSector setters
    void SetPlaneNormal(TVector3 n) { fPlaneNormal = n; }
    void SetPlaneRef(TVector3 r) { fPlaneRef = r; }
    void SetPhiMin(Double_t v) { fPhiMin = v; }
    void SetPhiMax(Double_t v) { fPhiMax = v; }

    // angle setters (NEW)
    void SetAngleThetaRange(Double_t min, Double_t max)
    {
        if (min > max) { Double_t tmp = min; min = max; max = tmp; }
        fAngleThetaMin = min;
        fAngleThetaMax = max;
    }

    void SetAnglePhiRange(Double_t min, Double_t max)
    {
        if (min > max) { Double_t tmp = min; min = max; max = tmp; }
        fAnglePhiMin = min;
        fAnglePhiMax = max;
    }

    void SetEnergyDistType(TString type) { fEnergyDistType = type; }
    void SetEnergyRange(TVector2 range) { fEnergyRange = range; }

    void SetSpectrumFilename(TString spctFilename) { fSpectrumFilename = spctFilename; }
    void SetSpectrumName(TString spctName) { fSpectrumName = spctName; }

    void SetAngularFilename(TString angFilename) { fAngularFilename = angFilename; }
    void SetAngularName(TString angName) { fAngularName = angName; }

    void PrintParticleSource();

    // Constructor / Destructor
    TRestParticleSource();
    virtual ~TRestParticleSource();

    ClassDef(TRestParticleSource, 3);
};
#endif
