# EEG_GPT_Model

This repository contains all data and code used for the paper titled: "ChatGPT-Based Model for Controlling Active Assistive Devices Using Non-Invasive EEG Signals"

## Project Structure

```
EEG_GPT_Model/
├── data/
│   ├── raw/           # Raw EEG and MoCap data
│   └── processed/     # Processed and synchronized datasets
├── results/           # Model evaluation results and visualizations
└── src/
    ├── data_processing/  # Data parsing and synchronization
    └── models/          # Model training and prediction
```

## Data Setup

Before running any code, you need to extract the data files:

1. Extract the raw data:

   ```bash
   # Extract EEG data
   unzip "data/raw/eeg/RAW EEG DATA FINAL.zip" -d data/raw/eeg/

   # Extract MoCap data
   unzip "data/raw/mocap/RAW MOCAP DATA FINAL.zip" -d data/raw/mocap/
   ```

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

For data processing:

```bash
pip install -r src/data_processing/requirements.txt
```

For model training and prediction:

```bash
pip install -r src/models/requirements.txt
```

## Usage

1. Data Processing:

   - Synchronize datasets: `python src/data_processing/synchronize.py --mocap_dir data/raw/mocap --eeg_dir data/raw/eeg --output_dir data/processed`

2. Model Training:

   - Train the model: `python src/models/train.py`

3. Model Prediction:
   - Run predictions: `python src/models/predict.py`

## Data

- Raw EEG data is stored in `data/raw/eeg/`
- Raw MoCap data is stored in `data/raw/mocap/`
- Processed and synchronized datasets are in `data/processed/`

## Results

The `results/` directory contains all model evaluation outputs:

- Training and validation plots
- Direction classifier visualizations
- Joint angles analysis
- Model performance metrics
- Evaluation results from different training runs

## Dependencies

### Data Processing Dependencies

- numpy>=1.19.2
- pandas>=1.2.0
- scipy>=1.6.0
- scikit-learn>=0.24.0
- matplotlib>=3.3.0

### Model Dependencies

- All data processing dependencies plus:
- tensorflow>=2.4.0
- joblib>=1.0.0
