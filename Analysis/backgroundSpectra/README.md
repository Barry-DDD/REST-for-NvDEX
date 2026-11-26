# NvDEx background spectra for ⁸²Se and ¹³⁶Xe

Original code is from **Zhaokan Cheng**. 

## Generate the figure

Run from the project directory with ROOT and Ghostscript (`gs` on `PATH`). Ghostscript embeds the plot fonts in the PDF.

```bash
root -n -l -b -q compareBackgroundSpectra.C
```

The output is `background_Se82_Xe136.pdf` with four pages in this order:

1. ⁸²Se full spectrum
2. ⁸²Se signal region
3. ¹³⁶Xe full spectrum
4. ¹³⁶Xe signal region

Every page uses counts per keV, kg of detector gas, and year. The full-range pages share a vertical scale, as do the signal-region pages. The signal-region pages use `E − Qββ` on the horizontal axis. Shading marks the integration window.

The Se pages show both single-event and pile-up components. The Xe pages show the three single-event components. Set `Xe136.ShowPileup` to `1` to display the calculated Xe pile-up curves and their legend entries.

## Configure the inputs

Edit `backgroundLevels.conf`. Each isotope lists its input spectra, detector mass, Qββ, ROI half width, resolution and coincidence time. `FwhmFraction` is the fractional FWHM at Qββ. `NaturalGammaIndex` is the target mean rate across the ROI, in counts/(keV kg yr). The terminal output gives the input index, applied scale and resulting annual count for each isotope.

| Isotope | Mass (kg) | ROI (keV) | Natural gamma index (counts/(keV kg yr)) | Counts in ROI per year |
| --- | ---: | ---: | ---: | ---: |
| ⁸²Se | 100 | 2981–3011 | 0.0004126 | 1.2378 |
| ¹³⁶Xe | 53 | 2445.7–2470.3 | 0.65552 | 854.666976 |

The gamma templates have 4,000 entries over 0–4,000 keV, one annual count per 1 keV bin. The Se two-neutrino spectrum uses the first two columns of its source file as energy in MeV and relative weight. Xe uses the Primakoff–Rosen two-neutrino shape by default. Set `Xe136.Decay0File` to use a DECAY0 two-electron event file with the original 22-line header format.

Both default configurations use the same gamma templates. Separate input paths can be supplied for detector-specific templates. The natural gamma scaling applies before the pile-up calculation so each gamma-containing term is recomputed. The two-neutrino decay rates retain the respective legacy normalizations.

Input paths are relative to their configuration file. To run another configuration with a separate output:

```bash
root -n -l -b -q 'compareBackgroundSpectra.C("alternative.conf","alternative.pdf")'
```

## Project layout

| Path | Purpose |
| --- | --- |
| `compareBackgroundSpectra.C` | ROOT entry point for the four-page PDF |
| `backgroundLevels.conf` | Default spectra, detector parameters and background levels |
| `src/Config.h` | Configuration parsing and validation |
| `src/Spectra.h` | Template loading, convolutions, energy smearing and ROI normalization |
| `src/Plot.h` | Page layout, axes, curve styles and PDF export |
| `data/` | Active input spectra |
| `deprected/` | Archived macros, plots, backups and inputs for the earlier layouts |
