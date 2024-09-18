import configparser
from typing import Any, Dict, List

import typer

from biomedics.extract_measurement.main import main as bio_post_processing


def run(config_path: str):
    """
    Run the bio post-processing with configuration from a .cfg file.

    Args:
        config (str): Path to the .cfg configuration file.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    script_config: Dict[str, Any] = config["script"] # type: ignore
    brat_dirs: List[str] = script_config["brat_dirs"]
    output_dirs: List[str] = script_config["output_dirs"]

    for brat_dir, output_dir in zip(brat_dirs, output_dirs):
        bio_post_processing(script_config, brat_dir, output_dir)


if __name__ == "__main__":
    typer.run(run)
