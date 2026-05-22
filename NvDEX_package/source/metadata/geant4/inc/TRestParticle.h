#ifndef RestCore_TRestParticle
#define RestCore_TRestParticle

#include <iostream>
#include <TString.h>
#include <TVector3.h>
#include "TObject.h"

class TRestParticle : public TObject {
  protected:
    TString fParticleName;
    Double_t fExcitationLevel = 0;
    TVector3 fDirection;  // 保留原来方向定义
    Double_t fEnergy = 0;
    Int_t fCharge = 0;

    // 用三个 double 存储位置
    Double_t fPosX = 0;
    Double_t fPosY = 0;
    Double_t fPosZ = 0;

    // 新增：粒子产生时间（单位ns）
    Double_t fTime = 0;

  public:
    // --- Getter 接口 ---
    TString GetParticleName() const { return fParticleName; }
    Double_t GetExcitationLevel() const { return fExcitationLevel; }
    Double_t GetEnergy() const { return fEnergy; }
    TVector3 GetMomentumDirection() const { return fDirection; }
    Int_t GetParticleCharge() const { return fCharge; }

    Double_t GetPosX() const { return fPosX; }
    Double_t GetPosY() const { return fPosY; }
    Double_t GetPosZ() const { return fPosZ; }
    Double_t GetTime() const { return fTime; }  //获取粒子产生时间

    // --- Setter 接口 ---
    void SetPosition(Double_t x, Double_t y, Double_t z) {
        fPosX = x; fPosY = y; fPosZ = z;
    }

    void SetDirection(const TVector3& dir) { fDirection = dir; }

    void SetParticleName(const TString& name) { fParticleName = name; }
    void SetExcitationLevel(Double_t level) {
        fExcitationLevel = level < 0 ? 0 : level;
    }
    void SetEnergy(Double_t energy) { fEnergy = energy; }
    void SetParticleCharge(Int_t charge) { fCharge = charge; }
    void SetTime(Double_t time) { fTime = time; }  //设置粒子产生时间

    // TRestParticle 的所有信息（包括时间）
    void SetParticle(const TRestParticle& ptcle) {
        fParticleName = ptcle.fParticleName;
        fExcitationLevel = ptcle.fExcitationLevel;
        fEnergy = ptcle.fEnergy;
        fCharge = ptcle.fCharge;
        fDirection = ptcle.fDirection;
        fPosX = ptcle.fPosX;
        fPosY = ptcle.fPosY;
        fPosZ = ptcle.fPosZ;
        fTime = ptcle.fTime;  
    }

    TRestParticle();
    virtual ~TRestParticle();

    ClassDef(TRestParticle, 4);  
};

#endif
