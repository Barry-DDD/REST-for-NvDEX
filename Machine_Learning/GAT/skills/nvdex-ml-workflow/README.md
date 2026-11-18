# NvDEx ML Workflow

`nvdex-ml-workflow` is a Codex skill for the NvDEx TPC GAT project. It uses the
training code, task definitions, HDF5 loader, and plotting scripts in this
repository. Keep the skill and the GAT code from the same checkout.

The workflow supports Z and XYZ vertex reconstruction, electron energy
asymmetry A, electron opening angle θ, and binary and multiclass particle
identification. It checks the HDF5 input, prepares a training launcher, and
validates saved results against the source data before generating metrics and
plots. See the [GAT repository README](../../README.md) for the data format and
Python environment.

## Set up

For Codex use, copy this skill directory to `.agents/skills/nvdex-ml-workflow`
in your GAT checkout, following the [installation instructions](../../docs/nvdex-ml-workflow-user.md).
You can also run the Python helper directly from the checkout.

Use Linux with Bash and a compatible GAT Python environment on the machine that
can access your input HDF5 and will run training. Select the execution mode that
matches your machine.

| Environment | Request configuration | Start training |
| --- | --- | --- |
| IHEP cluster | `deployment: "ihep"` (default). Use the Python and SLURM settings in the [IHEP deployment file](assets/ihep-deployment.json). | `sbatch <bundle>/job.sh` |
| Your own Linux GPU machine | Activate your Python environment and use `deployment: "current"`. This defaults to direct execution. Optionally specify `execution.gpu` and a trusted `execution.setup_script`. | `bash <bundle>/run.sh` |

CUDA runs require one visible GPU. Direct mode also supports an explicitly
selected CPU run. For a custom setup script, activate the environment before
preparing the bundle. The launcher will source the recorded script again when
you start training.

## Quick start

In Codex, invoke the skill from your GAT project. Replace the input path with
the path to your HDF5 on the execution host.

```text
$nvdex-ml-workflow

Check /path/to/sample.h5 and prepare a direct theta reconstruction run.
Use my active Python environment (deployment=current) and GPU 0.
Use theta-angle-v1, the MC truth vertex, and radius_mm=4.
Show me the resolved configuration and the launch command.
```

This example uses the MC truth vertex as a model input. Choose `centroid` for
hit-based vertex-relative features or `none` to omit the vertex reference.

For the same experiment from a terminal, use the
[direct request example](assets/direct-request.example.json) from the GAT
root. Set `infile` to your sample and adjust the GPU selection if needed.

```bash
cp skills/nvdex-ml-workflow/assets/direct-request.example.json check/theta_request.json
# Edit check/theta_request.json before preparing the run.
python skills/nvdex-ml-workflow/scripts/workflow.py prepare \
  --request check/theta_request.json --outdir check/prepared/theta_01
```

Review `PLAN.md`, `request_resolved.json`, and `input_checks.pdf` in the prepared
directory, then use the launch command in `PLAN.md`. The launcher validates
and plots the results after training. Look for `workflow_review/` inside
`results/direct/` for direct execution or `results/<job_id>/` for SLURM, both
under the prepared directory.

For all CLI options and examples, see the [workflow reference](references/workflow.md).
For more detail, see the [user guide](../../docs/nvdex-ml-workflow-user.md) and
[developer guide](../../docs/nvdex-ml-workflow-developer.md), both in Chinese.
The [skill instructions](SKILL.md) describe how Codex uses the workflow.
