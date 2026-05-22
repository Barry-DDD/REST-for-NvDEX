///______________________________________________________________________________
///______________________________________________________________________________
///______________________________________________________________________________
///
///
///             RESTSoft : Software for Rare Event Searches with TPCs
///
///             TRestG4Hits.h
///
///             Base class from which to inherit all other event classes in REST
///
///             jul 2015:   First concept
///                 Created as part of the conceptualization of existing REST
///                 software.
///                 J. Galan
///_______________________________________________________________________________

#ifndef RestCore_TRestG4Hits
#define RestCore_TRestG4Hits

#include <iostream>

#include <TArrayI.h>
#include <TArrayD.h>     // 用于 fTime
#include <TRestHits.h>
#include "TObject.h"

class TRestG4Hits : public TRestHits {
   protected:
    TArrayI fVolumeID;     // 存储每个 hit 的体积 ID
    TArrayI fProcessID;    // 存储每个 hit 的生成过程 ID
    TArrayD fTime;         // 每个 hit 的时间（单位：ns）

   public:
    // 获取器
    Int_t GetProcess(int n) { return fProcessID[n]; }
    Int_t GetHitProcess(int n) { return fProcessID[n]; }
    Int_t GetHitVolume(int n) { return fVolumeID[n]; }
    Int_t GetVolumeId(int n) { return fVolumeID[n]; }
    Double_t GetHitTime(int n) const {
        if (n < 0 || n >= fTime.GetSize()) {
            std::cerr << "GetHitTime: index " << n << " out of bounds (size: " << fTime.GetSize() << ")" << std::endl;
            return -1; 
        }
        return fTime[n];
    }// << 新增
    
    void SetHitTime(int n, Double_t t) { fTime[n] = t; }               // << 新增

    // 添加 hit 的方法
    void AddG4Hit(TVector3 pos, Double_t en, Int_t process, Int_t volume,
                  Double_t pre_en, Double_t pos_en);                   // 原有
    void AddG4Hit(Double_t X, Double_t Y, Double_t Z, Double_t en,
                  Int_t process, Int_t volume);                        // 原有

    void AddG4Hit(TVector3 pos, Double_t en, Int_t process, Int_t volume,    // << 新增
                  Double_t pre_en, Double_t pos_en, Double_t time);          // 带时间的重载版本

    void RemoveG4Hits();  // 清空所有 hit 信息，包括时间等

    // 工具函数
    Double_t GetEnergyInVolume(Int_t volID);
    TVector3 GetMeanPositionInVolume(Int_t volID);
    TVector3 GetFirstPositionInVolume(Int_t volID);
    TVector3 GetLastPositionInVolume(Int_t volID);

    // 构造与析构
    TRestG4Hits();
    virtual ~TRestG4Hits();

    ClassDef(TRestG4Hits, 3);  // 版本从 2 升级为 3
};

#endif
