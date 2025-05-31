import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import datetime
import re

# Import the MOCAP parsing function and metadata.
from mocap_parser import read_mocap_file, FileMetadata as MocapFileMetadata

# Import the EEG processing function and metadata.
from eeg_parser import process_eeg_file, FileMetadata as EegFileMetadata


def extract_trial(filepath: Path) -> Optional[str]:
    """
    Extract the trial identifier (e.g., 'T1') from the file name.
    Searches in the filename (without the extension) for a pattern like 'T' followed by one or more digits.
    Returns None if no trial identifier is found.
    """
    # Use file stem (the filename without extension) for matching.
    match = re.search(r"\bT\d+\b", filepath.stem)
    if match:
        return match.group(0)
    return None


def find_matching_files_by_trial(mocap_dir: Path, eeg_dir: Path) -> Dict[Path, Path]:
    """
    Find matching MOCAP and EEG files by comparing their file names based on a trial identifier (e.g., 'T1').

    Only files whose names contain the same trial identifier are paired.

    Example:
        MOCAP file: "Saket Sarkar 12-7-24 T1 RM.csv"
        EEG file:   "Insight MOCAP 12-7-2024 T1 Right Arm RM.csv"
    """
    mocap_trial_to_file: Dict[str, Path] = {}
    eeg_trial_to_file: Dict[str, Path] = {}
    matches: Dict[Path, Path] = {}

    # Process all MOCAP files.
    for filepath in mocap_dir.glob("*.csv"):
        trial = extract_trial(filepath)
        if trial:
            mocap_trial_to_file[trial] = filepath
        else:
            print(f"No trial identifier found in MOCAP file: {filepath}")

    # Process all EEG files.
    for filepath in eeg_dir.glob("*.csv"):
        trial = extract_trial(filepath)
        if trial:
            eeg_trial_to_file[trial] = filepath
        else:
            print(f"No trial identifier found in EEG file: {filepath}")

    # Pair up files that share the same trial identifier.
    for trial, mocap_file in mocap_trial_to_file.items():
        if trial in eeg_trial_to_file:
            matches[mocap_file] = eeg_trial_to_file[trial]
        else:
            print(
                f"No matching EEG file found for trial {trial} in MOCAP file {mocap_file}"
            )

    return matches


def align_data(
    mocap_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
    mocap_meta: MocapFileMetadata,
    eeg_meta: EegFileMetadata,
) -> Tuple[pd.DataFrame, dict]:
    """
    Synchronize MOCAP and EEG data based on their absolute start times.

    This function:
      - Computes an absolute time for each sample using the file’s start_time and sampling_rate.
      - Determines the common overlapping window using:
            common_start = max(mocap_meta.start_time, eeg_meta.start_time)
            common_end   = min(mocap_meta.end_time,   eeg_meta.end_time)
      - Trims both dataframes to the overlapping window.
      - Creates a new 'time' column for the merged dataframe (starting at 0).
      - Merges the two dataframes on the new time axis.

    Returns:
        merged_df: A dataframe containing data from both modalities synchronized on time.
        sync_meta: A dict with synchronization metadata:
            - "common_start": the absolute time when both recordings overlap
            - "common_end": the absolute end time of the overlap
            - "duration": duration of the overlapping window in seconds
            - "sampling_rate": the common sampling rate (120 Hz)
    """
    # Work on copies to avoid modifying the originals.
    mocap_df = mocap_df.copy()
    eeg_df = eeg_df.copy()

    # Compute absolute time for each sample.
    # (We use start_time from the metadata; note that for mocap, the provided 'timestamp'
    # is actually the end time, so we use start_time instead.)
    mocap_df["abs_time"] = (
        mocap_meta.start_time + np.arange(len(mocap_df)) / mocap_meta.sampling_rate
    )
    eeg_df["abs_time"] = (
        eeg_meta.start_time + np.arange(len(eeg_df)) / eeg_meta.sampling_rate
    )

    # Determine the common overlapping window.
    common_start = max(mocap_meta.start_time, eeg_meta.start_time)
    common_end = min(mocap_meta.end_time, eeg_meta.end_time)

    if common_start >= common_end:
        raise ValueError("No overlapping time window between MOCAP and EEG data.")

    # Trim both dataframes to the overlapping window.
    mocap_aligned = mocap_df[
        (mocap_df["abs_time"] >= common_start) & (mocap_df["abs_time"] <= common_end)
    ].copy()
    eeg_aligned = eeg_df[
        (eeg_df["abs_time"] >= common_start) & (eeg_df["abs_time"] <= common_end)
    ].copy()

    # Create a new time column starting at 0 (relative time).
    mocap_aligned["time"] = mocap_aligned["abs_time"] - common_start
    eeg_aligned["time"] = eeg_aligned["abs_time"] - common_start

    # Merge the two dataframes on the new time column.
    # (Since both dataframes are at 120 Hz and have been trimmed to the same window,
    # the 'time' columns should line up. In case of minor numerical differences, you could use
    # pd.merge_asof with an appropriate tolerance.)
    merged_df = pd.merge_asof(
        mocap_aligned, eeg_aligned, on="time", suffixes=("_mocap", "_eeg")
    )

    time_values = np.arange(1, int(len(merged_df)) + 1) / 120
    merged_df.insert(0, "Global Time", time_values)

    # Construct synchronization metadata.
    sync_meta = {
        "common_start": common_start,
        "common_end": common_end,
        "duration": common_end - common_start,
        "sampling_rate": mocap_meta.sampling_rate,  # 120 Hz
    }

    return merged_df, sync_meta


def main(mocap_dir: str, eeg_dir: str, output_dir: str):
    mocap_path = Path(mocap_dir)
    eeg_path = Path(eeg_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find matching file pairs.
    matches = find_matching_files_by_trial(mocap_path, eeg_path)

    for mocap_file, eeg_file in matches.items():
        try:
            # Read the raw MOCAP file.
            mocap_df, mocap_meta = read_mocap_file(mocap_file)
            # Process the matching EEG file (without saving the processed output).
            eeg_df, eeg_meta = process_eeg_file(
                str(eeg_file),
                fs_new=120,
                lpf_cutoff=30,
                out_dir=None,
                save_output=False,
            )

            # Align the two datasets using the updated function.
            merged_df, sync_meta = align_data(mocap_df, eeg_df, mocap_meta, eeg_meta)

            # Save the merged aligned data.
            base_name = mocap_file.stem
            merged_df.to_csv(out_path / f"{base_name}_merged_aligned.csv", index=False)

            print(
                f"Successfully processed and aligned {mocap_file.name} with {eeg_file.name}"
            )

            # Print human-readable common start and end times.
            common_start_str = datetime.datetime.fromtimestamp(
                sync_meta["common_start"]
            ).strftime("%Y-%m-%d %H:%M:%S")
            common_end_str = datetime.datetime.fromtimestamp(
                sync_meta["common_end"]
            ).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Common start time: {common_start_str}")
            print(f"Common end time: {common_end_str}")

            # Print overlap duration and sampling rate.
            print(f"Overlap duration: {sync_meta['duration']:.3f} seconds")
            print(f"Sampling rate: {sync_meta['sampling_rate']} Hz")
            print(f"Output file saved in {out_path}/")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {mocap_file}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synchronize MOCAP and EEG data")
    parser.add_argument(
        "--mocap_dir", required=True, help="Directory containing MOCAP files"
    )
    parser.add_argument(
        "--eeg_dir", required=True, help="Directory containing EEG files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for aligned files"
    )
    args = parser.parse_args()
    main(args.mocap_dir, args.eeg_dir, args.output_dir)
