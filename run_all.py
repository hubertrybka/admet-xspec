"""Run all processing configurations present in a directory."""

import argparse
import pathlib
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    args = parser.parse_args()

    # Get all .gin files in the specified directory
    config_dir = pathlib.Path(args.dir)
    config_files = list(config_dir.glob("*.gin"))

    # Run each configuration file using the process_data.py script
    for config_file in config_files:
        command = f"python process_data.py {config_file}"
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
    print("All configurations have been processed.")