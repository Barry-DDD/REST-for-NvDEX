"""CLI for registered NvDEx ML experiments; never submits scheduler jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from workflow_config import (IHEP, PRESETS, canonical_hash, load_json, project_root, resolve,
                             source_hashes, runtime_versions, training_arguments, write_json, interpreter_path)
from workflow_execution import (check_setup_fingerprint, execution_environment,
                                shell_environment_lines, verify_setup_environment)


def resolved_request(path, root):
    config = resolve(load_json(path), root)
    if config["status"] != "resolved":
        print(__import__("json").dumps(config, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    return config


def prepare(config, outdir, root, sample_events=24, max_graph_hits=2048):
    """Inspect an experiment and create a fresh direct or SLURM launch bundle."""
    from workflow_data import inspect, plot_inspection

    if os.name != "posix":
        raise ValueError("Prepare on the Linux execution host so paths and Python identify the runtime environment.")
    if interpreter_path(config["python"]) != interpreter_path(sys.executable):
        raise ValueError("Run prepare using the same Python executable selected in the request.")
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise ValueError("The preparation directory already exists; choose a fresh experiment directory.")
    backend = config["execution"]["backend"]
    if any(char in str(outdir) for char in (('\n', '\r', '"', '\\') if backend == "slurm" else ('\n', '\r', '\x00'))):
        raise ValueError("The job bundle path contains unsupported SLURM directive characters.")
    setup = verify_setup_environment(config, root)
    report, values = inspect(config, sample_events, max_graph_hits)
    outdir.mkdir(parents=True, exist_ok=False)
    plot_inspection(report, values, outdir / "input_checks.pdf")
    write_json(outdir / "request_resolved.json", config)
    write_json(outdir / "inspection.json", report)
    manifest = dict(schema_version=1, config=config, config_hash=canonical_hash(config),
                    inspection=report, sources=source_hashes(root), project=str(root),
                    workflow_entry=str(Path(__file__).resolve()), environment=runtime_versions(),
                    setup_script=setup)
    write_json(outdir / "plan.json", manifest)
    resources = config["resources"]
    log_path = str(outdir / "slurm-%j.out")
    lines = ["#!/bin/bash"]
    if backend == "slurm":
        lines += ["#SBATCH --job-name=nvdex-ml", f"#SBATCH --partition={resources['partition']}",
                  f"#SBATCH --account={resources['account']}", f"#SBATCH --qos={resources['qos']}",
                  "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --gres=gpu:1",
                  f"#SBATCH --cpus-per-task={resources['cpus']}", f"#SBATCH --mem={resources['memory']}",
                  f"#SBATCH --time={resources['time']}", f'#SBATCH --output="{log_path}"']
    lines += ["", *shell_environment_lines(config, setup, root),
              "exec " + shlex.join([config["python"], str(Path(__file__).resolve()), "--project", str(root),
                                     "execute", "--plan-dir", str(outdir)]), ""]
    launcher = outdir / ("job.sh" if backend == "slurm" else "run.sh")
    launcher.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    command = shlex.join(["sbatch" if backend == "slurm" else "bash", str(launcher)])
    result_location = "results/<SLURM_JOB_ID>/" if backend == "slurm" else "results/direct/"
    summary = ["# Prepared NvDEx experiment", "", "Status: prepared; no training has started.", "",
               f"Execution: `{backend}`; device: `{config['execution']['device']}`; Python: `{config['python']}`.",
               f"Task: `{config['task']}`; mode: `{config['target_mode']}`; vertex: `{config['vertex_source']}`.",
               f"Input: `{config['infile']}`; radius: {config['parameters']['radius_mm']} mm.", "",
               "Review request_resolved.json, inspection.json and input_checks.pdf before submission.", "",
               "Run on the selected host:", "", "```bash", command, "```", "",
               "One launch is allowed per bundle. A failed or cancelled launch requires a fresh preparation directory.",
               f"Successful training writes {result_location} and runs validated result review there.", ""]
    (outdir / "PLAN.md").write_text("\n".join(summary), encoding="utf-8", newline="\n")
    result = dict(status="prepared", backend=backend, directory=str(outdir), launch_command=command,
                  warnings=report["warnings"])
    if backend == "slurm":
        result["submit_command"] = command
    return result


def execute(plan_dir, root):
    """Run the prepared direct or allocated-SLURM experiment after provenance checks."""
    from workflow_data import fingerprint
    from workflow_results import review

    plan_dir = Path(plan_dir).resolve()
    plan = load_json(plan_dir / "plan.json")
    config = plan["config"]
    backend = config["execution"]["backend"]
    job_id = os.environ.get("SLURM_JOB_ID", "") if backend == "slurm" else None
    if backend == "slurm" and not job_id.isdigit():
        raise ValueError("SLURM execution requires an allocated job; submit the prepared job.sh manually.")
    run_id = job_id if backend == "slurm" else "direct"
    if plan["project"] != str(root) or canonical_hash(config) != plan["config_hash"]:
        raise ValueError("Prepared configuration/project identity mismatch.")
    if source_hashes(root) != plan["sources"]:
        raise ValueError("Sources changed after preparation; prepare a fresh bundle.")
    if runtime_versions() != plan["environment"]:
        raise ValueError("Python/package versions changed after preparation; prepare a fresh bundle.")
    if fingerprint(config["infile"]) != plan["inspection"]["input"]:
        raise ValueError("Input changed after inspection; prepare a fresh bundle.")
    check_setup_fingerprint(plan)
    if plan.get("setup_script") is not None and os.environ.get("NVDEX_WORKFLOW_SETUP_HASH") != plan["setup_script"]["sha256"]:
        raise ValueError("The setup script has not been loaded by this launcher; use the generated run.sh/job.sh.")
    env = execution_environment(config)
    if "CUDA_VISIBLE_DEVICES" in env:
        os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    import torch
    if config["execution"]["device"] == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("CUDA execution requires exactly one visible GPU; select execution.gpu for direct mode. No CPU fallback.")
    if interpreter_path(config["python"]) != interpreter_path(sys.executable):
        raise ValueError("The runtime Python differs from the prepared interpreter.")
    with (plan_dir / "launch.lock").open("x", encoding="utf-8") as handle:
        handle.write(run_id + "\n")
    outdir = plan_dir / "results" / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / "workflow_manifest.json", plan)
    status_path = outdir / "workflow_status.json"
    identity = dict(job_id=job_id, run_id=run_id, backend=backend, device=config["execution"]["device"])
    write_json(status_path, dict(state="training", **identity))
    try:
        command = [config["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                   str(root / "tpc_GAT/run.py"), *training_arguments(config, outdir)]
        subprocess.run(command, cwd=root, env=env, check=True)
        write_json(status_path, dict(state="reviewing", **identity))
        review(outdir, outdir / "workflow_review", root)
        write_json(status_path, dict(state="complete", **identity,
                                    note="Execution and artifact checks passed; physical performance still needs interpretation."))
    except Exception as error:
        write_json(status_path, dict(state="failed", **identity, error=str(error)))
        raise
    return dict(status="complete", directory=str(outdir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="GAT root; discover from the current directory, then try the IHEP default.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="List registered tasks and reference presets.")
    inventory_parser = commands.add_parser("inventory", help="List HDF5 fields without claiming readiness.")
    inventory_parser.add_argument("--infile", required=True)
    for name in ("resolve", "inspect", "prepare"):
        sub = commands.add_parser(name)
        sub.add_argument("--request", required=True)
        if name != "resolve":
            sub.add_argument("--outdir", required=True)
            sub.add_argument("--sample-events", type=int, default=24)
            sub.add_argument("--max-graph-hits", type=int, default=2048)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--plan-dir", required=True)
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--indir", required=True)
    review_parser.add_argument("--outdir", required=True)
    review_parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = project_root(args.project)
    if args.command == "catalog":
        from task_spec import SUPPORTED_TASKS, get_task_spec
        result = dict(tasks={name: vars(get_task_spec(name)) for name in SUPPORTED_TASKS}, presets=PRESETS,
                      default_deployment=IHEP, other_deployment="current: current Python and direct execution",
                      note="Reference presets are starting recipes, not proven optimal configurations.")
    elif args.command == "inventory":
        from workflow_data import inventory
        path = Path(args.infile)
        result = inventory(path if path.is_absolute() else root / path)
    elif args.command in {"resolve", "inspect", "prepare"}:
        config = resolved_request(args.request, root)
        if args.command == "resolve":
            result = config
        elif args.command == "prepare":
            result = prepare(config, args.outdir, root, args.sample_events, args.max_graph_hits)
        else:
            from workflow_data import inspect, plot_inspection
            report, values = inspect(config, args.sample_events, args.max_graph_hits)
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=False)
            write_json(outdir / "inspection.json", report)
            plot_inspection(report, values, outdir / "input_checks.pdf")
            result = report
    elif args.command == "execute":
        result = execute(args.plan_dir, root)
    else:
        from workflow_results import review
        result = review(args.indir, args.outdir, root, not args.no_plots)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError, subprocess.CalledProcessError) as error:
        print(__import__("json").dumps(dict(status="failed", error=str(error)), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
