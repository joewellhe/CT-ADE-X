"""Create readable MedDRA column names for one g4 dataset scope.

Read <data-dir>/<scope>/{soc,hlgt,hlt,pt}/ and write sibling *_friendly/
directories. The default data-dir is ./data/ct_ade, matching g4's default.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from src.meddra_graph import MedDRA


DATASET_SCOPES = ("single-only", "multi-only", "all")


def rename_columns_by_prefix(input_file, output_file, string_start, rename_dict):
    """
    Renames columns in a CSV file that start with a specific prefix and uses a dictionary to map suffixes.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the updated CSV file.
        string_start (str): The prefix to identify columns that need renaming.
        rename_dict (dict): Mapping of suffixes (after the prefix) to their new names.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Identify columns to rename
        columns_to_rename = [col for col in df.columns if col.startswith(string_start)]

        # Create a new mapping for renaming based on the suffix and the dictionary
        new_column_names = {
            col: string_start + rename_dict[col[len(string_start):]]
            for col in columns_to_rename
            if col[len(string_start):] in rename_dict
        }

        # Rename columns in the DataFrame
        df.rename(columns=new_column_names, inplace=True)

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Columns renamed successfully. Updated file saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main(scope="all", data_dir="./data/ct_ade"):
    if scope not in DATASET_SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; choose from {DATASET_SCOPES}")

    dataset_root = Path(data_dir) / scope
    meddra = MedDRA()
    meddra.load_data("./data/MedDRA_25_0_English/MedAscii")

    for level in ("SOC", "HLGT", "HLT", "PT"):
        rename_dict = {
            key[1]: value.term
            for key, value in meddra.nodes.items()
            if key[0] == level
        }
        input_dir = dataset_root / level.lower()
        output_dir = dataset_root / f"{level.lower()}_friendly"
        for split in ("train", "val", "test"):
            rename_columns_by_prefix(
                input_dir / f"{split}.csv",
                output_dir / f"{split}.csv",
                "label_",
                rename_dict,
            )
            rename_columns_by_prefix(
                input_dir / f"{split}_frequencies.csv",
                output_dir / f"{split}_frequencies.csv",
                "frequency_",
                rename_dict,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=DATASET_SCOPES, default="all",
        help="Dataset scope exported by g4 (default: all)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data/ct_ade"),
        help="Base directory containing scope subdirectories; use g4's --output-dir value "
             "(default: ./data/ct_ade)",
    )
    args = parser.parse_args()
    main(scope=args.scope, data_dir=args.data_dir)
