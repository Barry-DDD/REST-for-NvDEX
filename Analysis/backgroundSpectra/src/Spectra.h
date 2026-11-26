#ifndef NVDEX_BACKGROUND_SPECTRA_H
#define NVDEX_BACKGROUND_SPECTRA_H

#include "Config.h"

#include <TGraph.h>
#include <TH1D.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

namespace nvdex {
using Histogram = std::unique_ptr<TH1D>;

inline constexpr std::array<std::pair<int, int>, 6> pileupPairs{{
    {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}
}};
inline constexpr std::size_t numComponents = 3 + pileupPairs.size();

struct Spectra {
  std::array<Histogram, numComponents> components;
  double nominalGammaIndex = 0;
  double gammaScale = 0;
  double annualRoiGamma = 0;
};

inline Histogram makeHistogram(const TString& name, double energyShift = 0.)
{
  auto histogram = std::make_unique<TH1D>(name, "", energyBins, -energyShift, energyBins - energyShift);
  histogram->SetDirectory(nullptr);
  return histogram;
}

inline void readGammaTemplate(TH1D& histogram, const TString& path)
{
  std::ifstream input(path.Data());
  for (int bin = 1; bin <= histogram.GetNbinsX(); ++bin) {
    double count;
    if (!(input >> count) || !std::isfinite(count) || count < 0)
      throw std::runtime_error(std::string("Invalid or missing gamma template: ") + path.Data());
    histogram.SetBinContent(bin, count);
  }
  if (!(input >> std::ws).eof())
    throw std::runtime_error(std::string("Expected exactly 4000 counts in ") + path.Data());
}

inline void readSeTwoNu(TH1D& histogram, const TString& path)
{
  std::ifstream input(path.Data());
  TGraph shape;
  double energy, weight, unused;
  while (input >> energy) {
    if (!(input >> weight >> unused) || !std::isfinite(energy) || !std::isfinite(weight))
      throw std::runtime_error(std::string("Invalid Se 2nu template: ") + path.Data());
    shape.SetPoint(shape.GetN(), energy * 1000., weight);
  }
  if (!input.eof() || shape.GetN() < 2)
    throw std::runtime_error(std::string("Missing or empty Se 2nu template: ") + path.Data());

  for (int bin = 1; bin <= histogram.GetNbinsX(); ++bin)
    histogram.SetBinContent(bin, shape.Eval(histogram.GetBinCenter(bin)));
}

inline void readDecay0(TH1D& histogram, const TString& path)
{
  std::ifstream input(path.Data());
  std::string line;
  for (int row = 0; row < 22; ++row) {
    if (!std::getline(input, line))
      throw std::runtime_error(std::string("Missing DECAY0 input or incomplete header: ") + path.Data());
  }

  int eventNumber;
  while (input >> eventNumber) {
    double eventTime;
    int particles;
    if (!(input >> eventTime >> particles) || particles != 2)
      throw std::runtime_error(std::string("Expected a two-electron DECAY0 event in ") + path.Data());

    constexpr double electronMassMeV = 0.511;
    double energyMeV = 0;
    for (int particle = 0; particle < particles; ++particle) {
      int geantCode;
      double px, py, pz, timeShift;
      if (!(input >> geantCode >> px >> py >> pz >> timeShift) || geantCode != 3)
        throw std::runtime_error(std::string("Invalid DECAY0 electron record in ") + path.Data());
      energyMeV += std::sqrt(px * px + py * py + pz * pz + electronMassMeV * electronMassMeV)
                   - electronMassMeV;
    }
    if (!std::isfinite(energyMeV))
      throw std::runtime_error(std::string("Invalid energy in DECAY0 event file: ") + path.Data());
    histogram.Fill(1000. * energyMeV);
  }
  if (!input.eof())
    throw std::runtime_error(std::string("Invalid DECAY0 event file: ") + path.Data());
}

inline void fillPrimakoffRosen(TH1D& histogram, double qbb)
{
  // The Primakoff-Rosen sum-energy spectrum uses electron-mass units.
  constexpr double electronMassKeV = 511.;
  for (int bin = 1; bin <= histogram.GetNbinsX(); ++bin) {
    const double energy = histogram.GetBinCenter(bin);
    if (energy >= qbb) break;
    const double t = energy / electronMassKeV;
    const double remaining = (qbb - energy) / electronMassKeV;
    const double polynomial = (((t + 10.) * t + 40.) * t + 60.) * t + 30.;
    histogram.SetBinContent(bin, t * polynomial * std::pow(remaining, 5));
  }
}

inline Histogram convolve(const TH1D& first, const TH1D& second)
{
  auto combined = makeHistogram(TString::Format("%s_%s_pileup", first.GetName(), second.GetName()));
  for (int bin = 1; bin <= energyBins; ++bin) {
    double count = 0;
    for (int j = 1; j <= bin; ++j)
      count += first.GetBinContent(j) * second.GetBinContent(bin - j);
    combined->SetBinContent(bin, count);
  }
  return combined;
}

inline Histogram smear(const TH1D& original, const DetectorConfig& detector)
{
  auto smeared = makeHistogram(TString(original.GetName()) + "_smeared");
  std::vector<double> counts(energyBins + 1);
  for (int source = 1; source <= energyBins; ++source) {
    const double weight = original.GetBinContent(source);
    if (weight == 0) continue;

    const double energy = original.GetBinCenter(source);
    const double sigma = detector.fwhmFraction * std::sqrt(detector.qbb * energy) / 2.355;
    const double amplitude = weight * original.GetBinWidth(source) / (std::sqrt(2. * 3.1415927) * sigma);
    for (int dest = std::max(1, source - 100); dest <= std::min(energyBins, source + 100); ++dest) {
      const double delta = (original.GetBinCenter(dest) - energy) / sigma;
      counts[dest] += amplitude * std::exp(-0.5 * delta * delta);
    }
  }
  for (int bin = 1; bin <= energyBins; ++bin)
    smeared->SetBinContent(bin, counts[bin]);
  return smeared;
}

inline double roiCounts(const TH1D& histogram, const DetectorConfig& detector)
{
  const auto& axis = *histogram.GetXaxis();
  const double low = detector.roiLow();
  const double high = detector.roiHigh();
  const int first = std::max(1, axis.FindFixBin(low));
  const int last = std::min(axis.GetNbins(), axis.FindFixBin(high));
  double count = 0;
  for (int bin = first; bin <= last; ++bin) {
    const double overlap = std::min(high, axis.GetBinUpEdge(bin)) -
                           std::max(low, axis.GetBinLowEdge(bin));
    count += histogram.GetBinContent(bin) * std::max(0., overlap) / axis.GetBinWidth(bin);
  }
  return count;
}

inline Spectra calculateSpectra(const DetectorConfig& detector)
{
  std::array<Histogram, 3> single{{
      makeHistogram(detector.key + "_2nu"),
      makeHistogram(detector.key + "_gamma"),
      makeHistogram(detector.key + "_co60")
  }};
  readGammaTemplate(*single[1], detector.gammaFile);
  readGammaTemplate(*single[2], detector.co60File);

  if (detector.isSelenium()) {
    readSeTwoNu(*single[0], detector.twoNuFile);
  } else if (detector.twoNuFile.IsNull()) {
    fillPrimakoffRosen(*single[0], detector.qbb);
  } else {
    readDecay0(*single[0], detector.twoNuFile);
  }
  const double totalTwoNu = single[0]->Integral();
  if (!(totalTwoNu > 0) || !std::isfinite(totalTwoNu))
    throw std::runtime_error(std::string("Empty or invalid 2nu template for ") + detector.key.Data());
  single[0]->Scale(detector.twoNuDecaysPerKgYear * detector.massKg / totalTwoNu);

  Spectra result;
  for (std::size_t index = 0; index < single.size(); ++index)
    result.components[index] = smear(*single[index], detector);

  const double exposurePerYear = detector.massKg * (detector.roiHigh() - detector.roiLow());
  const double counts = roiCounts(*result.components[1], detector);
  const double target = detector.gammaIndex * exposurePerYear;
  if (!std::isfinite(counts) || (counts <= 0 && target > 0))
    throw std::runtime_error(std::string("Invalid natural gamma ROI for ") + detector.key.Data());
  result.nominalGammaIndex = counts / exposurePerYear;
  result.gammaScale = target > 0 ? target / counts : 0.;
  if (!std::isfinite(result.gammaScale))
    throw std::runtime_error(std::string("Invalid gamma scale for ") + detector.key.Data());

  single[1]->Scale(result.gammaScale);
  result.components[1]->Scale(result.gammaScale);
  result.annualRoiGamma = roiCounts(*result.components[1], detector);
  if (std::abs(result.annualRoiGamma - target) > 1.e-10 * std::max(1., target))
    throw std::runtime_error(std::string("Gamma ROI normalization failed for ") + detector.key.Data());

  const double overlap = detector.pileupWindowSeconds * detector.overlapVolumeFraction / secondsPerYear;
  for (std::size_t index = 0; index < pileupPairs.size(); ++index) {
    const auto& pair = pileupPairs[index];
    auto pileup = convolve(*single[pair.first], *single[pair.second]);
    pileup->Scale(overlap);
    result.components[index + single.size()] = smear(*pileup, detector);
  }
  return result;
}
}

#endif
