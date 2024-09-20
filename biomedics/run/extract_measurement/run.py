from typing import Any, Dict, List

import typer
from omegaconf import OmegaConf

from biomedics.extract_measurement.main import main as bio_post_processing


def run(config_path: str):
    """
    Run the bio post-processing with configuration from a .yaml file.

    Args:
        config (str): Path to the .yaml configuration file.
    """
    config = OmegaConf.load(config_path)

    script_config: Dict[str, Any] = config["script"] # type: ignore
    brat_dirs: List[str] = script_config["data_paths"]
    output_dirs: List[str] = script_config["output_paths"]

    for brat_dir, output_dir in zip(brat_dirs, output_dirs):
        bio_post_processing(script_config, brat_dir, output_dir)


if __name__ == "__main__":
    typer.run(run)
