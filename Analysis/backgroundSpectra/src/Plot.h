#ifndef NVDEX_BACKGROUND_PLOT_H
#define NVDEX_BACKGROUND_PLOT_H

#include "Spectra.h"

#include <TBox.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TStyle.h>

namespace nvdex {
enum class View { Full, Roi };

struct CurveStyle {
  const char* label;
  const char* color;
  int line;
};

inline constexpr std::array<CurveStyle, numComponents> curveStyles{{
    {"2#nu#beta#beta", "#a43a82", 1},
    {"Natural #gamma", "#0068a5", 1},
    {"^{60}Co #gamma", "#4d5359", 1},
    {"2#nu#beta#beta + 2#nu#beta#beta", "#168664", 2},
    {"2#nu#beta#beta + natural #gamma", "#c64f24", 2},
    {"2#nu#beta#beta + ^{60}Co #gamma", "#97633c", 2},
    {"Natural #gamma + natural #gamma", "#614992", 7},
    {"Natural #gamma + ^{60}Co #gamma", "#007c83", 7},
    {"^{60}Co #gamma + ^{60}Co #gamma", "#73787d", 3}
}};

inline constexpr double plotLeft = 0.145;
inline constexpr double plotRight = 0.955;
inline constexpr double plotBottom = 0.244;
inline constexpr double plotTop = 0.838;
inline constexpr int yMinExponent = -10;

inline void note(double x, double y, const TString& value, int size,
                 int font = 43, int align = 12, double angle = 0.)
{
  TLatex label;
  label.SetNDC();
  label.SetTextFont(font);
  label.SetTextSize(size);
  label.SetTextAlign(align);
  label.SetTextAngle(angle);
  label.SetTextColor(TColor::GetColor("#18212b"));
  label.DrawLatex(x, y, value);
}

inline int sharedUpperExponent(const std::array<DetectorConfig, 2>& detectors,
                               const std::array<Spectra, 2>& results, View view)
{
  double maximum = 0;
  for (std::size_t index = 0; index < detectors.size(); ++index) {
    const auto& detector = detectors[index];
    for (const auto& spectrum : results[index].components) {
      for (int bin = 1; bin <= energyBins; ++bin) {
        if (view == View::Roi &&
            std::abs(spectrum->GetBinCenter(bin) - detector.qbb) > zoomHalfWidthKeV)
          continue;
        const double density = spectrum->GetBinContent(bin) /
                               (detector.massKg * spectrum->GetBinWidth(bin));
        maximum = std::max(maximum, density);
      }
    }
  }
  return static_cast<int>(std::ceil(std::log10(std::max(1.e-8, 1.12 * maximum))));
}

inline void drawRoiBand(const DetectorConfig& detector, double shift, double yMin, double yMax)
{
  const double left = detector.roiLow() - shift;
  const double right = detector.roiHigh() - shift;
  TBox band;
  band.SetFillColor(TColor::GetColor("#edf2f8"));
  band.SetLineColor(band.GetFillColor());
  band.DrawBox(left, yMin, right, yMax);

  TLine edge;
  edge.SetLineColor(TColor::GetColor("#8093ad"));
  edge.SetLineStyle(3);
  edge.DrawLine(left, yMin, left, yMax);
  edge.DrawLine(right, yMin, right, yMax);
}

inline void drawLogTicks(double xMin, double xMax, int maxExponent)
{
  TLine line;
  const double width = xMax - xMin;
  for (int exponent = yMinExponent; exponent <= maxExponent; ++exponent) {
    const double y = std::pow(10., exponent);
    const bool major = exponent % 2 == 0;
    if (major && exponent > yMinExponent && exponent < maxExponent) {
      line.SetLineColor(TColor::GetColor("#dce2e8"));
      line.SetLineStyle(1);
      line.DrawLine(xMin, y, xMax, y);
    }

    line.SetLineColor(kBlack);
    line.DrawLine(xMin, y, xMin + (major ? 0.012 : 0.006) * width, y);
    if (!major) continue;
    const double position = plotBottom + (plotTop - plotBottom) *
                           (exponent - yMinExponent) / (maxExponent - yMinExponent);
    note(plotLeft - 0.012, position, TString::Format("10^{%d}", exponent), 20, 43, 32);
  }
}

inline void drawPage(TCanvas& canvas, const DetectorConfig& detector, const Spectra& spectra,
                     View view, int maxExponent)
{
  canvas.cd();
  canvas.SetFillColor(kWhite);
  canvas.SetLeftMargin(plotLeft);
  canvas.SetRightMargin(1. - plotRight);
  canvas.SetBottomMargin(plotBottom);
  canvas.SetTopMargin(1. - plotTop);
  canvas.SetLogy();
  canvas.SetTicks(0, 0);

  const bool zoom = view == View::Roi;
  const double shift = zoom ? detector.qbb : 0.;
  const double xMin = zoom ? -zoomHalfWidthKeV : 0.;
  const double xMax = zoom ? zoomHalfWidthKeV : energyBins;
  const double yMin = std::pow(10., yMinExponent);
  const double yMax = std::pow(10., maxExponent);
  auto* frame = canvas.DrawFrame(xMin, yMin, xMax, yMax);

  auto& xaxis = *frame->GetXaxis();
  xaxis.SetLabelFont(43);
  xaxis.SetLabelSize(21);
  xaxis.SetTickLength(0.018);
  xaxis.SetNdivisions(zoom ? 408 : 508, false);
  frame->GetYaxis()->SetLabelSize(0);
  frame->GetYaxis()->SetTickLength(0);

  drawRoiBand(detector, shift, yMin, yMax);
  drawLogTicks(xMin, xMax, maxExponent);

  TLegend legend(plotLeft, 0.066, plotRight, 0.155);
  legend.SetBorderSize(0);
  legend.SetFillStyle(0);
  legend.SetNColumns(3);
  legend.SetMargin(0.17);
  legend.SetTextFont(43);
  legend.SetTextSize(18);
  for (std::size_t index = 0; index < spectra.components.size(); ++index) {
    const auto& rate = *spectra.components[index];
    const auto& style = curveStyles[index];
    auto display = makeHistogram(TString(rate.GetName()) + "_density", shift);
    display->SetLineColor(TColor::GetColor(style.color));
    display->SetLineStyle(style.line);
    display->SetLineWidth(index == 1 ? 3 : 2);
    for (int bin = 1; bin <= energyBins; ++bin) {
      const double density = rate.GetBinContent(bin) / (detector.massKg * rate.GetBinWidth(bin));
      display->SetBinContent(bin, std::max(0., density));
    }
    legend.AddEntry(display->DrawCopy("HIST SAME"), style.label, "l");
  }
  canvas.RedrawAxis();
  TLine topBorder;
  topBorder.DrawLine(xMin, yMax, xMax, yMax);
  legend.DrawClone();

  note(0.045, (plotBottom + plotTop) / 2.,
       "Background rate [counts keV^{-1} kg^{-1} yr^{-1}]", 24, 43, 22, 90.);
  note((plotLeft + plotRight) / 2., 0.179, zoom ? "E - Q_{#beta#beta} [keV]" : "Energy [keV]",
       23, 43, 22);
  note(plotLeft, 0.950, detector.isotope + (zoom ? " background in the signal region"
                                                : " background spectrum"), 32, 63);
  note(plotLeft, 0.899, TString::Format("%g kg detector gas   |   Q_{#beta#beta} = %g keV"
                                       "   |   %g%% FWHM at Q_{#beta#beta}",
       detector.massKg, detector.qbb, detector.fwhmFraction * 100.), 19);
  note(plotLeft, 0.028, TString::Format("Shaded ROI: %g-%g keV    |    Natural #gamma: %.6g counts yr^{-1} in ROI%s",
       detector.roiLow(), detector.roiHigh(), spectra.annualRoiGamma,
       detector.isSelenium() || !detector.twoNuFile.IsNull() ? "" : "    |    2#nu#beta#beta: Primakoff-Rosen"),
       16);
}

inline void writePdf(const TString& path, const std::array<DetectorConfig, 2>& detectors,
                     const std::array<Spectra, 2>& results)
{
  gStyle->SetOptStat(0);
  gStyle->SetPaperSize(27., 19.);
  gStyle->SetLineScalePS(2.);
  const int fullMax = sharedUpperExponent(detectors, results, View::Full);
  const int roiMax = sharedUpperExponent(detectors, results, View::Roi);

  for (std::size_t index = 0; index < detectors.size(); ++index) {
    for (const auto view : {View::Full, View::Roi}) {
      const bool full = view == View::Full;
      const bool first = index == 0 && full;
      const bool last = index == detectors.size() - 1 && !full;
      TCanvas canvas(detectors[index].key + (full ? "_full" : "_roi"), "", 1200, 845);
      drawPage(canvas, detectors[index], results[index], view, full ? fullMax : roiMax);
      canvas.Print(path + (first ? "(" : last ? ")" : ""), last ? "EmbedFonts" : "pdf");
    }
  }
}
}

#endif
