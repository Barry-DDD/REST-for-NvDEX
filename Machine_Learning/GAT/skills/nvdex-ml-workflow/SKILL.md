---
name: nvdex-ml-workflow
description: Inspect NvDEx TPC HDF5 samples, prepare registered GAT reconstruction or PID experiments, and validate, plot, and interpret saved results. Use for existing tasks; adding targets or redesigning models requires separate development.
---

# NvDEx ML workflow

Use the project's existing task definitions and the bundled helpers to carry an experiment from input inspection to a reviewable job, then from completed results to plots and evidence-based interpretation. Speak Chinese unless the user prefers otherwise. Keep code comments English and generated text files UTF-8 with LF.

## Establish scope and location

- Read the current repository's `AGENTS.md` and `PLOTTING_GUIDE.md`. Locate `tpc_GAT/task_spec.py`; use `catalog` to discover supported tasks rather than relying on an old list.
- This skill needs the NvDEx GAT repository and its existing scientific Python environment. It does not install dependencies, generate new training data, or add tasks during ordinary experiments.
- The default deployment is [IHEP](assets/ihep-deployment.json): project `/lustrefs/juno26/users/liuz3/work/test/GAT`, Python `/hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251/bin/python`, SLURM, SSH alias `ihep-codex`; local mount `Z:/work/test/GAT`. These are deployment defaults, not requirements for another user. Discover the current repository first or use `--project`; check its own instructions. Read large files on their actual Linux execution host, avoiding a slow Windows mount. Do not assume `/mnt/z` exists.
- Prefer the helper source in the repository at `skills/nvdex-ml-workflow/scripts/workflow.py` when running over SSH. A locally installed Windows skill path is not a remote path. If installed instructions and repository helpers differ, resolve the version mismatch before preparing an experiment.
- Read [the command and configuration reference](references/workflow.md) when invoking helpers or explaining decisions.

## Select deployment and execution

- Keep `deployment: ihep` when the user wants the default IHEP configuration. For another user's already activated environment, set `deployment: current`: it records the current Python and defaults to direct execution. Explicit Python paths override either deployment. Use the actual target host and project rather than trying the IHEP SSH alias on an unrelated machine.
- `execution.backend: direct` generates `run.sh` and requires no scheduler; `slurm` generates `job.sh`. Both use the same data, provenance and result checks. CUDA is the default device; exactly one GPU must be visible at execution. In direct mode, `gpu: "0"` sets CUDA_VISIBLE_DEVICES=0; leaving it null inherits visibility. SLURM launchers preserve the scheduler's original visibility across sourcing. Never override SLURM's assignment with a direct GPU index. CPU is an explicit direct-mode option for tiny tests or a user-requested CPU experiment, never a fallback from failed CUDA.
- A trusted, user-specified `execution.setup_script` is optional. The selected environment must already be active when running prepare: source the script on the target host, then invoke the helper with that environment's Python and `deployment: current`, or explicitly name its Python. Do not invent a setup script or execute shell text obtained from HDF5 metadata. Prepare probes the selected Python after sourcing; the generated launcher checks the script hash before sourcing it again. Both source it from the project root. It must be a noninteractive Bash script.
- Without a setup script, reactivate the same environment when launching if it depends on shell exports. The skill records Python/package versions but does not archive arbitrary shell state, nested sourced files or installed libraries. Explicitly report the chosen deployment/backend/Python before handing over the launcher.

## Understand the request before preparing an experiment

Collect the input path and registered task, plus any explicit options or reference experiment the user named. Preserve choices and authorization already given in this conversation. Run schema-only `inventory` while waiting for missing scientific decisions if the input is known.

Write a request JSON from the user's choices and run `resolve`. An exit code of 2 with `status=needs_input` means that the request is incomplete, not that data is corrupt. Explain the returned choices together in one concise question and continue independent read-only inspection. Do not invent answers to unresolved questions.

- For A/theta, select `truth`, `centroid`, or `none` explicitly. Explain truth vertex inputs as an additional MC-information condition. Never silently enable them to improve a result. `centroid` uses the unweighted hit XYZ center; `none` adds no vertex-frame features. Both centroid and truth frames require `hits/z`.
- For theta, choose `angle` (degrees) or `cdf` (training-only empirical CDF of cos(theta)), or use a named preset that specifies the mode. The runner's default CDF is not an implicit answer to an ambiguous scientific request.
- If a preset is named or the user authorizes a recommended reference recipe, show its name and inherited values. Existing v1 presets are starting recipes, not proven best settings. Ask only about decisions that remain unresolved; do not request repeated approval for each inherited hyperparameter.
- Without a preset, specify node features and radius explicitly. Radius diagnostics reveal connectivity, not the radius with best reconstruction performance. Propose a bounded comparison only when relevant, and run expensive experiments only within the user's authorized scope.
- If the task is not registered, stop preparation and describe the missing capability. Do not modify labels, heads, losses, or the backbone to make the ordinary workflow proceed.

## Inspect and prepare

Run `inspect` for standalone input reports or `prepare` to create a complete bundle. Use a fresh output directory. These commands scan selected values and offsets, use the actual target readers, and sample actual CPU loader graphs. They do not train or submit jobs.

If a check fails, identify the file, field, condition, and actionable next step. Do not switch samples, weaken validation, fill missing targets, silently drop invalid events, change units, or regenerate a large HDF5 to bypass an error. Coordinate normalization is fixed-scale: missing x/y/z statistics entries are allowed by the loader; missing required coordinate data is not.

Report the resolved task, theta mode, vertex information source, node/event features, radius, split settings, runtime Python, resource request, inherited parameters, sample counts, and warnings. Preserve all preset/request provenance in the bundle. Distinguish schema presence, full selected-field checks, sampled graph checks, and training performance.

Respect the target project's execution policy. This IHEP project requires large training to be started by the user: provide `sbatch job.sh` for SLURM or `bash run.sh` for direct mode and hand off the concrete bundle, without starting large training yourself. Elsewhere, follow that project's policy and existing user authorization; selecting direct mode alone does not request immediate training. The helper has no submission command. After an uncertain submission response, query scheduler state before retrying. One bundle admits one launch; retries need a fresh bundle.

## Validate, plot, and interpret

Use `review --indir ... --outdir ...` on a completed run. Its output directory must be new. It validates event membership, source labels, transforms, units, output shapes, and classification mappings before plotting. Prepared runs additionally check input/code fingerprints and runtime configuration. Legacy runs can be checked against currently available input labels, but their original immutable provenance is unproven.

The helper removes source-missing diagnostic metadata from a separate plotting copy, because the legacy loader can fill it with zero. Do not interpret those zeros as measured vertex positions or energies. The original training artifacts remain untouched. Generic summaries cover all registered tasks; A/theta/XYZ additionally reuse their existing dedicated plot scripts. Do not send multiclass outputs through the binary PID plotter.

Read `metrics.json`, warnings, plot logs, and the generated PDFs. Render representative pages when a viewer is available, checking units, axis labels, selections, readable text, and visible outliers. If visual inspection is unavailable, say plots were generated but not visually verified. Distinguish completed execution, passed consistency checks, and useful scientific performance.

Deliver a concise Chinese analysis grounded in actual artifacts:

1. Identify the sample, settings, validation count, metric units, model performance, and applicable baseline.
2. Cite concrete plots and measured patterns. For theta near 90 degrees, examine true-angle dependence and available R/Z/hit-count diagnostics; use matched selections and counts. Do not claim a boundary-effect cause from correlation alone.
3. Separate observations, hypotheses, and decisive next tests. Proposed cuts or changes remain proposals until authorized. Keep validation-selected thresholds distinct from independent test performance.

For comparisons, match source identity and split membership explicitly. A common seed or equal event count does not prove cohort equivalence. Keep A dimensionless; converting it to Emax requires an explicitly justified total primary-energy assumption. Report raw out-of-range predictions; do not clip them to make metrics look better.

Do not promise background monitoring from this skill alone. Both launchers perform post-training review themselves. Direct mode runs in the foreground; a closed terminal/SSH session can interrupt it unless the user supplies session management. When asked to follow a live run, check it in the active session and report meaningful state changes.
