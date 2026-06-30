# Phase C detector-MC prep configs

These files are scaffolding for Task 8 Phase C preparation. They are not the
Phase C topology/reconstruction analysis.

Use `Gas`, the physical volume, for restG4 generation and storage. The logical
volume is `gasVolume`.

The medium switch is made through the top GDML selected by the run RML:

- `workspace/Geometry/main_xetma1pct10atm.gdml`
- `workspace/Geometry/main_sef6.gdml`

The source scaffold includes `phaseC_g4_sef6_smoke.rml` as the small
second-medium parse/tracking check.

The intended XML-entity form, using `&gasMaterial;` inside
`<materialref ref="..."/>`, was tested but REST/Geant4 did not expand the
entity in that attribute. The working layout is explicit per-medium top GDML
files that share the same `materials.xml` and geometry catalog but define the
`gasVolume` material reference locally.

For the retained Phase C signal files, keep:

- `TRestSignalEventBranch`
- a filled `AnalysisTree` with truth and detector-level observables

Drop:

- full `TRestG4EventBranch`
- full `TRestHitsEventBranch`

This is done in `TRestProcessRunner` with:

```xml
<parameter name="inputAnalysis" value="off"/>
<parameter name="inputEvent" value="off"/>
<parameter name="outputEvent" value="on"/>
```
