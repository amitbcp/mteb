"""Creates slurm jobs for running models on all tasks"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import mteb

# SHOULD BE UPDATED
slurm_prefix = """#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --nodes=1
#SBATCH --partition=genai-tao
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 72:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/mnt/shared/aamita/project/mteb/scripts/oci/jobs/%x-%j.out           # output file name
#SBATCH --error=/mnt/shared/aamita/project/mteb/scripts/oci/jobs/%x-%j.log
#SBATCH --exclusive
# >>> Conda setup <<<
source /mnt/shared/aamita/miniconda3/etc/profile.d/conda.sh
conda activate image_retrieval
"""


def create_slurm_job_file(
    model_name: str,
    task_name: str,
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
    batch_size: int,
    device: str
) -> Path:
    """Create slurm job file for running a model on a task"""
    slurm_job = f"{slurm_prefix}\n"
    # slurm_job += f"mteb run -m {model_name} -t {task_name} --output_folder {results_folder.resolve()} --co2_tracker true"
    slurm_job += f"python benchmark_task.py --model_name  {model_name} --task_name  {task_name} --batch_size {batch_size} --device {device}"

    model_path_name = model_name.replace("/", "__")

    slurm_job_file = slurm_jobs_folder / f"{model_path_name}_{task_name}.sh"
    with open(slurm_job_file, "w") as f:
        f.write(slurm_job)
    return slurm_job_file


def create_slurm_job_files(
    model_names: list[str],
    tasks: Iterable[mteb.AbsTask],
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
    batch_size: int,
    device: str
) -> list[Path]:
    """Create slurm job files for running models on all tasks"""
    slurm_job_files = []
    for model_name in model_names:
        for task in tasks:
            slurm_job_file = create_slurm_job_file(
                model_name,
                task.metadata.name,
                results_folder,
                slurm_prefix,
                slurm_jobs_folder,
                batch_size,
                device
            )
            slurm_job_files.append(slurm_job_file)
    return slurm_job_files


def run_slurm_jobs(files: list[Path]) -> None:
    """Run slurm jobs based on the files provided"""
    for file in files:
        # subprocess.run(["sbatch", file])
        print(f"File Name : {file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark within OCI for Multimodal Retrieval.")
    parser.add_argument("--model_names", type=str,nargs='+',
                        default=["google/siglip-large-patch16-384"], help="Base path to the dataset directory.")
    parser.add_argument("--task_type", type=str, nargs='+',
                        default=["Any2AnyRetrieval"],
                        help="Task to process")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()



    # project_root = Path(__file__).parent / ".." / ".." / ".."
    results_folder = Path(__file__) / "results"
    # results_folder = Path("/data/niklas/results")
    slurm_jobs_folder = Path(__file__) / "slurm_jobs"

    tasks = mteb.get_tasks(languages=["eng"],modalities=["text", "image"],
                       task_types=args.task_type)
    task_names = [ task.metadata.name for task in tasks]

    # tasks = [t for t in tasks if t.metadata.name not in retrieval_to_be_downsampled]

    slurm_jobs_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)
    files = create_slurm_job_files(
        args.model_names, task_names, results_folder, slurm_prefix, slurm_jobs_folder, args.batch_size, args.device
    )
    run_slurm_jobs(files)
