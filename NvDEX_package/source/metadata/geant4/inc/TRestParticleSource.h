//////////////////////////////////////////////////////////////////////////
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestParticleSource.h
///
///             Class to store a particle definition
///
///             jul 2015:   First concept
///                 Created as part of the conceptualization of existing REST
///                 software.
///                 J. Galan
///
//////////////////////////////////////////////////////////////////////////

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

    TVector3 fPlaneNormal = TVector3(0, 0, 1);
    TVector3 fPlaneRef = TVector3(1, 0, 0);
    TVector2 fPlanePhiRange = TVector2(0, 0);
    TVector2 fAngleThetaRange = TVector2(0, 0);
    TVector2 fAnglePhiRange = TVector2(0, 0);

   public:
    TString GetParticle() { return fParticleName; }
    TString GetAngularDistType() { return fAngularDistType; }
    TVector3 GetDirection() { return fDirection; }
    TString GetEnergyDistType() { return fEnergyDistType; }
    TVector2 GetEnergyRange() { return fEnergyRange; }
    Double_t GetMinEnergy() { return fEnergyRange.X(); }
    Double_t GetMaxEnergy() { return fEnergyRange.Y(); }

    TString GetSpectrumFilename() { return fSpectrumFilename; }
    TString GetSpectrumName() { return fSpectrumName; }

    TString GetAngularFilename() { return fAngularFilename; }
    TString GetAngularName() { return fAngularName; }
    TVector3 GetPlaneNormal() { return fPlaneNormal; }
    TVector3 GetPlaneRef() { return fPlaneRef; }
    Double_t GetPhiMin() { return fPlanePhiRange.X(); }
    Double_t GetPhiMax() { return fPlanePhiRange.Y(); }
    Double_t GetAngleThetaMin() { return fAngleThetaRange.X(); }
    Double_t GetAngleThetaMax() { return fAngleThetaRange.Y(); }
    Double_t GetAnglePhiMin() { return fAnglePhiRange.X(); }
    Double_t GetAnglePhiMax() { return fAnglePhiRange.Y(); }

    void SetAngularDistType(TString type) { fAngularDistType = type; }
    void SetEnergyDistType(TString type) { fEnergyDistType = type; }
    void SetEnergyRange(TVector2 range) { fEnergyRange = range; }

    void SetSpectrumFilename(TString spctFilename) { fSpectrumFilename = spctFilename; }
    void SetSpectrumName(TString spctName) { fSpectrumName = spctName; }

    void SetAngularFilename(TString angFilename) { fAngularFilename = angFilename; }
    void SetAngularName(TString angName) { fAngularName = angName; }
    void SetPlaneNormal(const TVector3& normal) { fPlaneNormal = normal; }
    void SetPlaneRef(const TVector3& ref) { fPlaneRef = ref; }
    void SetPhiMin(Double_t phi) { fPlanePhiRange = TVector2(phi, fPlanePhiRange.Y()); }
    void SetPhiMax(Double_t phi) { fPlanePhiRange = TVector2(fPlanePhiRange.X(), phi); }
    void SetAngleThetaRange(Double_t minTheta, Double_t maxTheta) {
        fAngleThetaRange = TVector2(minTheta, maxTheta);
    }
    void SetAnglePhiRange(Double_t minPhi, Double_t maxPhi) { fAnglePhiRange = TVector2(minPhi, maxPhi); }

    void PrintParticleSource();

    // Construtor
    TRestParticleSource();
    // Destructor
    virtual ~TRestParticleSource();

    ClassDef(TRestParticleSource, 3);
};
#endif
