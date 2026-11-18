# Commands and experiment contract

Run on the Linux host that will execute training. The default deployment is defined in `assets/ihep-deployment.json`: IHEP project path, tested Python `/hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251/bin/python`, SLURM and site resources. `--project` overrides project discovery; otherwise the current repository is preferred and the IHEP root is a fallback. `deployment: current` instead captures the helper's Python and defaults to direct execution. An explicit top-level `python` overrides both. Prepare must run with the selected interpreter.

```bash
cd /lustrefs/juno26/users/liuz3/work/test/GAT
PY=/hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251/bin/python
WF=skills/nvdex-ml-workflow/scripts/workflow.py
"$PY" "$WF" catalog
"$PY" "$WF" inventory --infile input/new_sample.h5
"$PY" "$WF" resolve --request check/theta_request.json
"$PY" "$WF" inspect --request check/theta_request.json --outdir check/theta_input_01
"$PY" "$WF" prepare --request check/theta_request.json --outdir jobs/prepared/theta_01
```

Example request; the assistant replaces the input with the user's actual sample, not an inferred default:

```json
{
  "schema_version": 1,
  "preset": "theta-angle-v1",
  "infile": "input/new_sample.h5",
  "vertex_source": "truth",
  "parameters": {"radius_mm": 4.0}
}
```

All request keys are checked. `parameters` accepts existing runner options only: epochs, batch_size, learning_rate, weight_decay, features, feature_norm_mode, radius_mm, xy_abs_max, z_abs_max, target_abs_max, event_features, keep_empty_events, hidden_channels, num_layers, heads, head_hidden_channels, dropout, activation, num_workers, grad_clip, seed, val_fraction, log_interval. `task` and `target_mode` are top-level keys. `vertex_source` replaces the two mutually exclusive runner flags.

## Direct execution in another user's environment

Start from `assets/direct-request.example.json`. No IHEP account, SSH alias or scheduler is required. Replace the sample path. A request with a setup script looks like:

```json
{
  "deployment": "current",
  "preset": "theta-angle-v1",
  "infile": "input/new_sample.h5",
  "vertex_source": "truth",
  "execution": {
    "backend": "direct",
    "device": "cuda",
    "gpu": "0",
    "setup_script": "/home/alice/setup_ml.sh"
  },
  "parameters": {"radius_mm": 4.0}
}
```

```bash
cd /home/alice/GAT
source /home/alice/setup_ml.sh
python skills/nvdex-ml-workflow/scripts/workflow.py --project /home/alice/GAT prepare \
  --request check/theta_request.json --outdir check/prepared/theta_01
bash check/prepared/theta_01/run.sh
```

Preparation runs a noninteractive setup probe, with a 60-second timeout, but never starts training. Run the helper from the desired environment first; it does not guess which Python an arbitrary script would activate. `setup_script` can be omitted/null if the environment is already active. Scripts are sourced from the project root; relative file references must be meaningful there. Launchers verify their recorded SHA-256 before sourcing, then check input/code/Python/package identity. Only the entry setup script is hashed; nested sources and external libraries are not archived. Direct launch uses absolute paths and must run on the prepared host/filesystem. It writes `results/direct/`; it does not require or manufacture SLURM_JOB_ID.

Direct `device: cuda` requires exactly one visible GPU. `gpu` is a single numeric string controlling CUDA_VISIBLE_DEVICES; null preserves visibility (also useful in an interactive allocation). Direct `device: cpu` explicitly hides CUDA for tests or requested CPU work. Never silently fall back to CPU. SLURM allows only CUDA and a null GPU selector so the scheduler's assignment is preserved. Direct requests reject SLURM `resources` instead of silently ignoring them. Both modes currently target Linux/Bash; native Windows execution is not supported.

Presets: `theta-angle-v1`, `theta-cdf-v1`, `A-v1`, `xyz-v1`, `z-v1`, `binary-v1`, `multiclass-v1`. They specify task/mode and a reference recipe: radius 2 mm, 30 epochs, batch 64, learning rate 3e-4, weight decay 5e-5, standard feature normalization, 128 channels, 3 layers, 4 heads, seed 42, validation fraction 0.2. A/theta/XYZ use x/y/z/energy/log_energy; Z/classification use x/y/energy/log_energy. The common recipe derives from current detsim A/theta/XYZ jobs; Z/classification recipes are explicit starting choices, not a claim to reproduce every historical job. No A/theta preset chooses a vertex source.

Without a preset, `task`, input, features and radius must be supplied; theta additionally needs its mode and A/theta need a vertex choice. Other common training options use visible workflow defaults. `resolve` returns exit code 2 with a structured `questions` list until these decisions are available. Invalid input returns exit code 1.

Optional resources:

```json
{"resources": {"partition": "gpu", "account": "junogpu", "qos": "normal", "cpus": 5, "memory": "64G", "time": "16:00:00"}}
```

SLURM supports one node and one GPU. CPU count must cover the main process and loader workers. Resource defaults come from current detsim jobs and remain site-specific. No training is started or job submitted by prepare. A prepared bundle contains `request_resolved.json`, `inspection.json`, `input_checks.pdf`, `plan.json`, `job.sh` (SLURM) or `run.sh` (direct), and `PLAN.md`. IHEP submission remains manual:

```bash
sbatch /lustrefs/juno26/users/liuz3/work/test/GAT/jobs/prepared/theta_01/job.sh
```

Both modes execute the prepared manifest after provenance/runtime checks. SLURM writes `results/<SLURM_JOB_ID>/`, direct writes `results/direct/`. A launch lock prevents using the same bundle twice. Input/code/setup changes cause a clear failure; prepare a fresh bundle after resolving the change. This is a drift check, not a complete archive. Direct mode is foreground execution, not a persistent daemon; terminal/session lifetime is the user's responsibility.

Result review:

```bash
"$PY" "$WF" review --indir jobs/prepared/theta_01/results/123456 --outdir check/theta_review_01
"$PY" "$WF" review --indir results/older_run --outdir check/older_review_01 --no-plots
```

Review needs access to the source HDF5 named in the run config; run it on the server. It writes metrics, a report skeleton, generic plots, and task-specific plots for A/theta/XYZ in a new directory. It never overwrites original artifacts. Dedicated plots run on sanitized result copies under `validated_artifacts/`; missing source metadata is removed there. Z and PID use the generic plots, not the historical plotters' implicit energy selections or binary-only assumptions.

`--sample-events` (default 24) and `--max-graph-hits` (default 2048) bound sampled graph construction in inspect/prepare. A full selected-field and offset scan still occurs. Very large graph samples are skipped and reported; if none can be checked, preparation fails. Field scans can be expensive on very large files; prepare the command for the user when it exceeds authorized interactive work. This bound does not cap training graph size or guarantee GPU memory sufficiency.

Scientific limits: imported HDF5 normalization stats are not asserted to come only from training data; original-source grouping is not available to the existing split interface; file identity is path/size/mtime rather than a content hash. A passing input check is not a training or physics-performance guarantee. Source code and dependencies must remain available for reproducibility.
