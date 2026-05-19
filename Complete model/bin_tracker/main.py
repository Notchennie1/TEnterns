"""
Bin Tracker — Entry Point
=========================
Run this to start the full plug-and-play bin tracking system.

Usage:
    python -m bin_tracker.main                   # uses config.yaml in cwd
    python -m bin_tracker.main --config my.yaml  # custom config path
"""

import argparse
import sys

from bin_tracker.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Plug-and-Play Bin Hand Tracker")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    pipeline = Pipeline(config_path=args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
