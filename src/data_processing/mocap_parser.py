import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import re
from typing import Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")


@dataclass
class FileMetadata:
    timestamp: float  # Header timestamp, which for MOCAP is the recording end time (seconds since epoch)
    sampling_rate: int  # Sampling rate (from header or default)
    filepath: Path  # Original file path
    original_samples: int  # Number of data rows in the original file
    start_time: float  # Absolute start time of recording (computed from header end time minus duration)
    end_time: float  # Absolute end time of recording (equal to header timestamp)


def read_mocap_file(
    filepath: Path, save_output: bool = False, out_dir: Path = None
) -> Tuple[pd.DataFrame, FileMetadata]:
    """
    Read a MOCAP CSV file and extract both the data and its metadata.

    Expected file structure:
      - Lines 1-10: header (e.g. title, date, Time:, etc.)
      - Lines 11-13: header rows for the data (columns, units, etc.)
      - Line 14+: actual numeric data

    This function:
      - Extracts the recording Date and Time from the header.
      - Extracts the sampling rate (from the line following "Model Outputs").
      - Finds the data header row (containing "Frame" and "Time") and reads the data starting two lines below.
      - Computes the number of samples and the absolute recording start and end times.
        Note: The header time is actually the recording end time, so:
                rec_start = header timestamp - (last value in "Time")
      - Optionally saves the processed DataFrame as CSV if save_output is True.

    Returns:
      tuple: (DataFrame, FileMetadata)
    """
    try:
        with open(filepath, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        # Initialize metadata variables
        date_str = None
        time_str_header = None
        sampling_rate = 120  # default value if not found

        header_data_index = None
        # Process header lines to extract Date, Time, and Sampling Rate
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            # Split on any whitespace
            parts = line.split(",")
            if parts[0].startswith("Date:"):
                if len(parts) >= 2:
                    date_str = parts[1].split()[0]
            elif parts[0].startswith("Time:"):
                if len(parts) >= 2:
                    time_str_header = parts[1]
            # Look for the "Model Outputs" line; the next non-empty line should contain the sampling rate.
            if "Model Outputs" in parts:
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        try:
                            sampling_rate = int(lines[j].strip(","))
                        except Exception:
                            pass
                        break
            # If we encounter the data header row (with "Frame" and "Time"), record its index.
            if "Frame" in line and "Time" in line:
                header_data_index = i
                break

        if header_data_index is None:
            raise ValueError("Data header row not found.")

        # Compute the header timestamp.
        # Note: For MOCAP, the header time is the end time.
        if date_str and time_str_header:
            dt = datetime.datetime.strptime(
                f"{date_str} {time_str_header}", "%Y-%m-%d %H:%M:%S"
            )
            timestamp = dt.timestamp()
        else:
            timestamp = 0

        # The data header row is at header_data_index; skip the next row (units) and start data at header_data_index + 2.
        data_start_index = header_data_index + 2

        # Construct final column headers
        headers = ["Frame", "Time"]
        angle_types = ["RShoulderAngles", "RElbowAngles", "RWristAngles"]
        components = ["X", "Y", "Z"]
        for angle in angle_types:
            for comp in components:
                headers.append(f"{angle}_{comp}")

        # Read data rows
        data = []
        for line in lines[data_start_index:]:
            line = line.strip()
            if not line:
                continue
            values = line.split(",")
            if len(values) < len(headers):
                continue  # skip incomplete rows
            data.append(values[: len(headers)])

        df = pd.DataFrame(data, columns=headers)
        # Convert numeric columns (all except "Frame") to numeric types
        numeric_cols = df.columns.difference(["Frame"])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        original_samples = len(df)

        # Compute recording start and end times.
        # Here, the header timestamp is the recording end time.
        # We assume the "Time" column gives the elapsed time since recording start.
        if "Time" in df.columns and not df["Time"].empty:
            rec_end = timestamp  # header timestamp is the end time.
            rec_start = rec_end - (len(df) / 120)
        else:
            rec_start = timestamp
            rec_end = timestamp

        # Optionally save the processed DataFrame as CSV
        if save_output and out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_fname = filepath.name.replace(".csv", "_processed.csv")
            output_path = out_dir / out_fname
            df.to_csv(output_path, index=False)

        file_meta = FileMetadata(
            timestamp=timestamp,
            sampling_rate=sampling_rate,
            filepath=filepath,
            original_samples=original_samples,
            start_time=rec_start,
            end_time=rec_end,
        )

        return df, file_meta

    except Exception as e:
        raise ValueError(f"Error parsing MOCAP file {filepath}: {str(e)}")


# Top-level worker function (for multiprocessing)
def worker(arg):
    filepath, out_dir = arg
    try:
        df, meta = read_mocap_file(
            Path(filepath), save_output=True, out_dir=Path(out_dir)
        )
        return {"filename": filepath, "status": "success", "meta": meta}
    except Exception as e:
        return {"filename": filepath, "status": "error", "error": str(e)}


def process_mocap_directory(mocap_dir, out_dir=None, n_workers=None):
    """
    Process all MOCAP CSV files in a directory in parallel.

    Parameters:
      mocap_dir: string or Path - directory containing MOCAP CSV files
      out_dir: string or Path - directory to save processed files (optional)
      n_workers: int - number of parallel workers (default: CPU count - 1)

    Returns:
      list: A list of dictionaries, each containing:
            - filename, status ("success" or "error"), and meta (FileMetadata) if success.
    """
    mocap_dir = Path(mocap_dir)
    if not mocap_dir.exists():
        raise ValueError(f"Directory not found: {mocap_dir}")

    if out_dir is None:
        out_dir = mocap_dir / "processed"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(mocap_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {mocap_dir}")

    args = [(str(f), str(out_dir)) for f in csv_files]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    with Pool(n_workers) as pool:
        results = pool.map(worker, args)

    return results


# Standalone usage: run this module directly to process a directory and print a summary.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel MOCAP file parser")
    parser.add_argument(
        "--mocap_dir", type=str, help="Directory containing MOCAP CSV files"
    )
    parser.add_argument(
        "--out_dir", type=str, help="Directory to save processed CSV files"
    )
    parser.add_argument(
        "--n_workers", type=int, default=None, help="Number of parallel workers"
    )
    args = parser.parse_args()

    results = process_mocap_directory(
        args.mocap_dir, out_dir=args.out_dir, n_workers=args.n_workers
    )

    print("\nProcessing Summary:")
    print("-" * 50)
    for r in results:
        filename = Path(r.get("filename", "Unknown")).name
        if r["status"] == "success":
            meta = r["meta"]
            header_str = datetime.datetime.fromtimestamp(meta.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            start_str = datetime.datetime.fromtimestamp(meta.start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_str = datetime.datetime.fromtimestamp(meta.end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"\nMOCAP File: {filename}")
            print(f"Header Timestamp (Recording End): {header_str}")
            print(f"Sampling rate: {meta.sampling_rate} Hz")
            print(f"Original samples: {meta.original_samples}")
            print("Recording info:")
            print(f"\tstart time: {start_str}\n\tend time: {end_str}")
        else:
            print(f"\nError processing {filename}:")
            print(f"Error: {r.get('error')}")
        print("-" * 50)
