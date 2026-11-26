#ifndef NVDEX_BACKGROUND_CONFIG_H
#define NVDEX_BACKGROUND_CONFIG_H

#include <TEnv.h>
#include <TString.h>
#include <TSystem.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nvdex {
inline constexpr int energyBins = 4000;
inline constexpr double zoomHalfWidthKeV = 200.;
inline constexpr double secondsPerYear = 365. * 24 * 3600;

struct DetectorConfig {
  TString key;
  TString isotope;
  TString gammaFile;
  TString co60File;
  TString twoNuFile;
  double qbb;
  double massKg;
  double roiHalfWidth;
  double fwhmFraction;
  double gammaIndex;
  double twoNuDecaysPerKgYear;
  double pileupWindowSeconds;
  double overlapVolumeFraction;

  double roiLow() const { return qbb - roiHalfWidth; }
  double roiHigh() const { return qbb + roiHalfWidth; }
  bool isSelenium() const { return key == "Se82"; }
};

inline double readNumber(const TEnv& settings, const TString& key, bool allowZero = false)
{
  const std::string input = settings.GetValue(key, "");
  std::size_t parsed = 0;
  double value = 0;
  try {
    value = std::stod(input, &parsed);
  } catch (const std::logic_error&) {
    throw std::runtime_error(std::string("Expected a number for ") + key.Data());
  }
  if (parsed != input.size() || !std::isfinite(value) || value < 0 || (!allowZero && value == 0))
    throw std::runtime_error(std::string("Invalid value for ") + key.Data());
  return value;
}

inline DetectorConfig readDetector(const TEnv& settings, const TString& directory,
                                   const TString& key, double overlap)
{
  auto filePath = [&](const char* name) {
    const TString path = settings.GetValue(key + "." + name, "");
    return path.IsNull() || gSystem->IsAbsoluteFileName(path) ? path : directory + "/" + path;
  };
  auto number = [&](const char* name, bool allowZero = false) {
    return readNumber(settings, key + "." + name, allowZero);
  };

  const bool selenium = key == "Se82";
  DetectorConfig detector{
      key, selenium ? "^{82}Se" : "^{136}Xe",
      filePath("GammaSpectrum"), filePath("Co60Spectrum"),
      filePath(selenium ? "TwoNuSpectrum" : "Decay0File"),
      number("QKeV"), number("MassKg"), number("RoiHalfWidthKeV"),
      number("FwhmFraction"), number("NaturalGammaIndex", true),
      number("TwoNuDecaysPerKgYear"), number("PileupWindowSeconds", true), overlap
  };
  if (detector.roiLow() < 0 || detector.roiHigh() > energyBins ||
      detector.roiHalfWidth >= zoomHalfWidthKeV)
    throw std::runtime_error(std::string("Invalid ROI for ") + key.Data());
  if (detector.gammaFile.IsNull() || detector.co60File.IsNull() ||
      (selenium && detector.twoNuFile.IsNull()))
    throw std::runtime_error(std::string("Missing spectrum path for ") + key.Data());
  return detector;
}

inline std::array<DetectorConfig, 2> readConfig(const TString& path)
{
  TEnv settings;
  if (settings.ReadFile(path, kEnvLocal) != 0)
    throw std::runtime_error(std::string("Cannot read background settings: ") + path.Data());

  const TString directory = gSystem->DirName(path);
  const double overlap = readNumber(settings, "Pileup.OverlapVolumeFraction", true);
  if (overlap > 1) throw std::runtime_error("Pileup.OverlapVolumeFraction must be at most 1");

  return {readDetector(settings, directory, "Se82", overlap),
          readDetector(settings, directory, "Xe136", overlap)};
}
}

#endif
