#include "src/Config.h"
#include "src/Plot.h"
#include "src/Spectra.h"

#include <iomanip>
#include <iostream>

void compareBackgroundSpectra(const char* configPath = nullptr,
                              const char* outputPdf = "background_Se82_Xe136.pdf")
{
  const TString config = configPath ? TString(configPath)
      : TString(gSystem->DirName(__FILE__)) + "/backgroundLevels.conf";
  const TString output = outputPdf;
  if (!output.EndsWith(".pdf"))
    throw std::runtime_error("Output filename must end with .pdf");

  const std::unique_ptr<char[]> ghostscript(gSystem->Which(gSystem->Getenv("PATH"), "gs",
                                                         kExecutePermission));
  if (!ghostscript)
    throw std::runtime_error("Ghostscript (gs) is required to embed the PDF fonts");

  const auto detectors = nvdex::readConfig(config);
  const std::array<nvdex::Spectra, 2> spectra{{
      nvdex::calculateSpectra(detectors[0]), nvdex::calculateSpectra(detectors[1])
  }};

  for (std::size_t index = 0; index < detectors.size(); ++index) {
    const auto& detector = detectors[index];
    const auto& spectrum = spectra[index];
    std::cout << std::setprecision(9) << detector.key << " natural gamma, "
              << detector.roiLow() << "-" << detector.roiHigh() << " keV, "
              << detector.massKg << " kg: " << spectrum.nominalGammaIndex << " -> "
              << detector.gammaIndex << " counts/(keV kg yr), scale " << spectrum.gammaScale
              << ", " << spectrum.annualRoiGamma << " counts/yr in ROI\n";
  }
  nvdex::writePdf(output, detectors, spectra);
}
