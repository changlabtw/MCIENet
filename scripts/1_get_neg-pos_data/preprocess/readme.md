# Data Preprocessing

## Step 1: Data Preparation and Preprocessing

This directory contains scripts adapted from the [CHINN (Chromatin Interaction Neural Network) repository](https://github.com/mjflab/chinn) for processing raw ChIA-PET data into training-ready format.

### Input Requirements
- Raw interaction files in BEDPE format
- DNase/open chromatin regions in BED format
- Transcription factor peaks in BED format

### Processing Steps
1. Filter and clean raw interaction data
2. Generate positive and negative sample pairs
3. Process DNase and TF peak data
4. Generate training-ready datasets

### Usage
Refer to the main pipeline script `pipe.sh` for processing your data:

```bash
./pipe.sh <interactions.bedpe> <dnase.bed> <tf_peaks.bed> <sample_name> <output_dir>
```

### Note
These scripts are adapted from the original [CHINN repository](https://github.com/mjflab/chinn) with modifications for this project.