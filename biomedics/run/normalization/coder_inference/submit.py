import os
from argparse import ArgumentParser
from pathlib import Path

import submitit


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to the output directory")
    parser.add_argument("--config", type=Path, required=True, help="Path to the config file")
    return parser.parse_args()

def main(args):

    os.system(f"python coder_inference/run.py {args.model_name} {args.input_dir} {args.output_dir} {args.config_path}")

if __name__ == "__main__":
    args = parse_args()
    executor = submitit.AutoExecutor(folder=args.output_dir)
    executor.update_parameters(
        slurm_job_name="coder_inference",
        gpus_per_node=1,
        slurm_partition="gpuV100",
        slurm_gres="gpu:v100:1",
        slurm_cpus_per_task=2,
        mem_gb=20,  # Total memory in GB
        slurm_time=48 * 60 * 60,  # 48 hours in seconds
        slurm_additional_parameters={
            "container_image": "/scratch/images/sparkhadoop.sqsh",
            "container_mounts": "/export/home/$USER:/export/home/$USER,/data/scratch/$USER:/data/scratch/$USER",
            "container_mount_home": True,
            "container_writable": True,
        }
    )
    job = executor.submit(main)
