import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal as sp
from multiprocessing import Pool, cpu_count
import warnings
import datetime
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class FileMetadata:
    timestamp: float  # Seconds since epoch (or use datetime if preferred)
    sampling_rate: int  # New sampling rate (fs_new)
    filepath: Path
    original_fs: int  # Original sampling rate (from header)
    new_fs: int  # New sampling rate (should equal sampling_rate)
    original_samples: int  # Number of samples in the original file
    new_samples: int  # Number of samples in the processed file
    channels_processed: list
    start_time: float
    end_time: float


def parse_eeg_timestamp(time_str: str) -> datetime.datetime:
    """Parse EEG timestamp from format HH.MM.SS or HH:MM:SS."""
    time_str = time_str.replace(".", ":")
    return datetime.datetime.strptime(time_str, "%d:%m:%y %H:%M:%S")


def filt_lowpass(x, fs, fc, filt_type="FIR"):
    """Low pass filter implementation with adaptive filter length

    Parameters
    ----------
    x: ndarray
        input signal
    fs: integer
        sampling frequency
    fc: float
        low-pass cutoff frequency
    filt_type: string
        filter type, 'FIR' only in this implementation
    Returns
    -------
    y : ndarray
        The filtered signal.
    """
    # Ensure x is a float array
    x = np.asarray(x, dtype=float)
    
    if filt_type.lower() == "fir":
        l_filt = min(501, len(x) // 4)
        l_filt = l_filt + 1 if l_filt % 2 == 0 else l_filt
        b = sp.firwin(l_filt, fc, window="blackmanharris", pass_zero="lowpass", fs=fs)
        xmean = np.nanmean(x)
        pad_len = min(len(x) - 1, l_filt * 3)
        y = sp.filtfilt(b, 1, x - xmean, padlen=pad_len)
        y += xmean
        return y
    else:
        raise ValueError("Only FIR filter type supported")

def convertEPOC_PLUS(value_1, value_2):   
    edk_value = "%.8f" % (((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) -128) * 32.82051289))
    return edk_value


def process_eeg_file(
    fname: str, fs_new=120, lpf_cutoff=55, out_dir=None, save_output=False
):
    """
    Process a single EEG file and return a tuple: (processed DataFrame, FileMetadata).

    The processing includes:
      - Reading header information and parsing metadata.
      - Reading EEG data (skipping the header).
      - Filtering and resampling selected EEG channels.
      - Interpolating quality indicators and recomputing timing columns.
      - Optionally saving the processed file.

    Returns:
      tuple: (df_resampled, FileMetadata) on success.
    """
    try:
        with open(fname, "r") as f:
            header = f.readline().strip()

        header_parts = header.split(",")
        header_dict = {}
        for part in header_parts:
            if part and ":" in part:
                key, value = part.split(":", 1)
                header_dict[key.strip().replace(" ", "_")] = value.strip()

        fs_original = int(header_dict.get("sampling", 128))
        # labels = header_dict.get("labels", "").split(" ")
        # if not any(labels):
        #     labels = None
        # labels = ["1", "2", "AF3", "3", "T7", "4", "Pz", "5", "T8", "6", "AF4"]
        # df = pd.read_csv(fname, skiprows=1, names=labels, usecols=range(len(labels)))

        use_cols = [4, 5, 8, 9, 12, 13, 22, 23, 26, 27]
        column_names = ["AF3", "AF3+1", "T7", "T7+1", "Pz", "Pz+1", "T8", "T8+1", "AF4", "AF4+1"]
        df = pd.read_csv(fname, skiprows=1, usecols=use_cols, names=column_names)
        # Mapping: tuple (base, adjacent)
        channel_mapping = {
            "AF3": ("AF3", "AF3+1"),
            "T7":  ("T7", "T7+1"),
            "Pz":  ("Pz", "Pz+1"),
            "T8":  ("T8", "T8+1"),
            "AF4": ("AF4", "AF4+1")
        }

        # Loop through each pair in channel_mapping and apply conversion in place.
        for base, pair in channel_mapping.values():
            df[base] = df.apply(
                lambda row: convertEPOC_PLUS(
                    str(int(float(row[base]))),  # first parameter from the adjacent column
                    str(int(float(row[pair])))   # second parameter from the base column
                ),
                axis=1
            )
        print(df.head())

        timestamp = 0
        for k, v in header_dict.items():
            if "recorded" in k.lower():
                try:
                    timestamp = parse_eeg_timestamp(v).timestamp()
                except Exception:
                    timestamp = 0
                break

        eeg_channels = ["AF3", "T7", "Pz", "T8", "AF4"]
        metadata_cols = ["COUNTER", "TIME_STAMP_s", "TIME_STAMP_ms"]
        # quality_cols = [col for col in df.columns if col.startswith("CQ_")]

        resampled_data = {}
        n_samples_new = int(len(df) * fs_new / fs_original)
        # COUNTER from 1 - 120
        resampled_data["Frame"] = (np.arange(n_samples_new) % fs_new) + 1

        for ch in eeg_channels:
            if ch in df.columns:
                x_filt = filt_lowpass(df[ch].values, fs_original, lpf_cutoff, "FIR")
                resampled = sp.resample_poly(x_filt, fs_new, fs_original)
                if len(resampled) > n_samples_new:
                    resampled_data[ch] = resampled[:n_samples_new]
                else:
                    resampled_data[ch] = np.pad(
                        resampled, (0, n_samples_new - len(resampled)), "edge"
                    )
            else:
                resampled_data[ch] = np.full(n_samples_new, np.nan)

        # for col in quality_cols:
        #     orig_times = np.arange(len(df))
        #     new_times = np.linspace(0, len(df) - 1, n_samples_new)
        #     resampled_data[col] = np.interp(new_times, orig_times, df[col].values)

        if "TIME_STAMP_s" in df.columns and "TIME_STAMP_ms" in df.columns:
            orig_time = df["TIME_STAMP_s"] + df["TIME_STAMP_ms"] / 1000
            new_times = np.linspace(
                orig_time.iloc[0], orig_time.iloc[-1], n_samples_new
            )
            resampled_data["TIME_STAMP_s"] = np.floor(new_times)
            resampled_data["TIME_STAMP_ms"] = (new_times % 1) * 1000

        df_resampled = pd.DataFrame(resampled_data)
        for col in metadata_cols:
            if col in df_resampled.columns:
                df_resampled[col] = df_resampled[col].round().astype(int)

        if "Time" not in df_resampled.columns:
            time_values = np.arange(1, n_samples_new + 1) / fs_new
            df_resampled.insert(1, "Time", time_values)

        if save_output and out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            input_path = Path(fname)
            out_fname = input_path.name.replace(".csv", f"_{fs_new}Hz.csv")
            output_path = out_dir / out_fname
            header_dict["sampling"] = str(fs_new)
            new_header_parts = [
                f"{k.replace('_', ' ')}:{v}"
                for k, v in header_dict.items()
                if k != "labels"
            ]
            new_header = ",".join(new_header_parts)
            with open(output_path, "w", newline="") as f:
                f.write(new_header + "\n")
                df_resampled.to_csv(f, sep=",", index=False, header=True)

        file_meta = FileMetadata(
            timestamp=timestamp,
            sampling_rate=fs_new,
            filepath=Path(fname),
            original_fs=fs_original,
            new_fs=fs_new,
            original_samples=len(df),
            new_samples=len(df_resampled),
            channels_processed=eeg_channels,
            start_time=timestamp,
            end_time=timestamp + (len(df_resampled) / fs_new),
        )

        return df_resampled, file_meta

    except Exception as e:
        raise ValueError(f"Error processing EEG file {fname}: {str(e)}")


# Move the worker function to the top-level so it can be pickled.
def worker(arg):
    fname, fs_new, lpf_cutoff, out_dir = arg
    try:
        _, meta = process_eeg_file(fname, fs_new, lpf_cutoff, out_dir, save_output=True)
        return {"filename": fname, "status": "success", "meta": meta}
    except Exception as e:
        return {"filename": fname, "status": "error", "error": str(e)}


def process_eeg_directory(
    eeg_dir, out_dir=None, fs_new=120, lpf_cutoff=55, n_workers=None
):
    """Process all EEG files in directory in parallel

    Parameters
    ----------
    eeg_dir: string or Path
        directory containing EEG CSV files
    fs_new: int
        new sampling frequency (default 120)
    lpf_cutoff: float
        low-pass filter cutoff frequency (default 55)
    n_workers: int
        number of parallel workers (default: CPU count - 1)
    """
    eeg_dir = Path(eeg_dir)
    if not eeg_dir.exists():
        raise ValueError(f"Directory not found: {eeg_dir}")

    if out_dir is None:
        out_dir = eeg_dir / f"processed_{fs_new}Hz"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(eeg_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {eeg_dir}")

    args = [(str(f), fs_new, lpf_cutoff, out_dir) for f in csv_files]

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    with Pool(n_workers) as pool:
        results = pool.map(worker, args)

    return results


# If running as a script.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel EEG file processor")
    parser.add_argument(
        "--eeg_dir", type=str, required=True, help="Directory containing EEG CSV files"
    )
    parser.add_argument(
        "--out_dir", type=str, help="Directory to save processed EEG CSV files"
    )
    parser.add_argument(
        "--fs_new", type=int, default=120, help="New sampling frequency (Hz)"
    )
    parser.add_argument(
        "--lpf_cutoff",
        type=float,
        default=30,
        help="Low-pass filter cutoff frequency (Hz)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None, help="Number of parallel workers"
    )
    args = parser.parse_args()
    results = process_eeg_directory(
        eeg_dir=args.eeg_dir,
        out_dir=args.out_dir,
        fs_new=args.fs_new,
        lpf_cutoff=args.lpf_cutoff,
        n_workers=args.n_workers,
    )
    print("\nProcessing Summary:")
    print("-" * 50)
    for r in results:
        filename = Path(r.get("filename", "Unknown")).name

        if r["status"] == "success":
            meta = r["meta"]
            # Precompute formatted timestamps
            start_str = datetime.datetime.fromtimestamp(meta.start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_str = datetime.datetime.fromtimestamp(meta.end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            print(f"\nEEG File: {filename}")
            print(f"Original sampling rate: {meta.original_fs} Hz")
            print(f"New sampling rate: {meta.new_fs} Hz")
            print(f"Samples: {meta.original_samples} → {meta.new_samples}")
            print(f"Channels processed: {', '.join(meta.channels_processed)}")
            print("Recording info:")
            print(f"\tstart time: {start_str}\n\tend time: {end_str}")
        else:
            print(f"\nError processing {filename}:")
            print(f"Error: {r.get('error')}")
        print("-" * 50)
