# MCIENet

MCIENet: Multi-scale CNN-based Information Extraction from DNA Sequences for 3D Chromatin Interactions Prediction

![MCIENet Workflow](figures/fig1-a_Workflow.png)

## Table of Contents
- [Getting Started](#getting-started)
  - [Step 0: Clone the Repository](#step-0-clone-the-repository)
  - [Step 1: Setup Environment](#step-1-setup-environment)
  - [Step 2: Prepare Dataset](#step-2-prepare-dataset)
    - [2.1 Generate Positive-Negative Pairs](#21-generate-pos-neg-pairs)
    - [2.2 Generate Training Data](#22-generate-training-data)
  - [Step 3: Model Training](#step-3-train)
    - [3.1 BaseCNN](#31-basecnn)
    - [3.2 MCIENet](#32-mcienet)
  - [Step 4: Explainable AI (XAI) Analysis](#step-4-xai)
- [References](#references)
- [Citation](#citation)

## Project Structure

```
MCIENet/
├── MCIENet/
│   ├── model/                 # Model implementation
│   │   ├── classifier.py      # Classifier implementation
│   │   ├── data_extractor.py  # Data extractor
│   │   ├── layers.py          # Custom neural network layers
│   │   └── utils.py           # Model related utility functions
│   │
│   ├── utils/                 # Utility functions
│   │
│   ├── dataset.py             # Dataset handling
│   ├── loop_model.py          # Model training loop
│   └── trainer.py             # Trainer implementation
│
├── conf/                      # Configuration files
├── data/                      # Dataset directory
├── docker/                    # Docker-related files
├── figures/                   # Figures
├── notebook/                  # Jupyter Notebooks
├── output/                    # Training outputs and logs
├── scripts/                   # Utility scripts organized by workflow stages
│   ├── 1_get_neg-pos_data/    # Scripts for generating pos/neg pairs
│   ├── 2_generate_traindata/  # Scripts for preparing training data
│   ├── 3_train/               # Scripts for model training
│   ├── 4_XAI/                 # Scripts for explainable AI analysis
│   ├── helper_scripts/        # Helper utilities
│   └── set_env/               # Environment setup scripts
│
├── data_helper.py            # Data processing utilities
├── get_attr.py               # Attribute access utilities
└── train.py                  # Training entry point
```

## Getting Started
### Step 0: Clone the Repository
```shell
git clone https://github.com/aaron-ho/MCIENet.git
```

### Step 1: Setup Environment
#### Option 1: Using Docker (Recommended)
> *you may need to Install  `docker` and `docker-compose` first*


Create and enter the container
```shell
# Build and start the container (in background)
docker-compose -f docker/docker-compose.yml up -d

# Enter the container
docker-compose -f docker/docker-compose.yml exec mcienet /bin/bash

# You can now use MCIENet in the command line ...
```
exit and remove the container

```shell
# To exit the container
exit

# To stop and remove the container
docker-compose -f docker/docker-compose.yml down
```

> **note**: image we use will cost about 16.4GB disk space. If you don't have enough disk space, you can use option 2.

#### Option 2: Manual Setup with Scripts
Scripts is under `scripts/set_env`, you can use it to setup the environment.
- `set-env_conda`: set up conda environment for MCIENet
- `set-env_venv`: set up venv environment for MCIENet
- `set-env_dnabert`: set up conda environment for dnabert

> **note**: these scripts just for reference, you need to customize your environment path in the script.  

### Step 2: Prepare Dataset
**First, you need to make sure you are in the docker container or activated the environment**

> **note**: this project includes pre-processed data for two example datasets `gm12878 ctcf` and `helas3 ctcf` located in the `data/proc/` directory. The pre-processing steps have already been completed for these example datasets. If you plan to use these datasets, you can skip the [2.1 Generate Pos-Neg Pairs](#21-generate-pos-neg-pairs) step and proceed directly to [2.2 Generate Training Data](#22-generate-training-data).

#### Data Structure
- Raw data: `data/raw/` - Contains the original input files (e.g., BED, BAM, FASTA files)
  - **Important**: You need to download the hg19 reference genome (hg19.fa) from [UCSC](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/) and place it in this directory before running the scripts.
- Processed data: `data/proc/` - Contains pre-processed data ready for training
- Training data: `data/train/` - Will contain the final training data generated from processed data

#### 2.1 Generate Pos-Neg Pairs

The `scripts/1_get_neg-pos_data/` directory contains the data processing pipeline that transforms raw interaction data into training-ready format. This step is crucial for preparing both positive and negative samples for model training.

##### Key Components:
- `preprocess/`: Contains scripts for initial data processing
  - `pipe.sh`: Main pipeline script that orchestrates the preprocessing steps
  - `process_pos.sh`: Processes positive interaction samples
  - `generate_*.py`: Python scripts for generating and processing sample pairs
- `gm12878_ctcf.sh` and `helas3_ctcf.sh`: Example scripts demonstrating how to run the pipeline

##### Requirements:
- **BEDTools** (includes `mergeBed` and `pairToBed`):
  - Ubuntu/Debian: `sudo apt-get install bedtools`
  - For other systems, see [BEDTools documentation](https://bedtools.readthedocs.io/en/latest/content/installation.html)

##### Processing Your Own Data
1. Place your raw data in the `data/raw/` directory:
   - Interaction files in BEDPE format
   - DNase/open chromatin regions in BED format
   - Transcription factor peaks in BED format

2. Run the preprocessing pipeline:
   ```bash
   # Example command structure
   ./scripts/1_get_neg-pos_data/preprocess/pipe.sh \
       <interactions.bedpe> \
       <dnase.bed> \
       <tf_peaks.bed> \
       <sample_name> \
       <output_directory>
   ```

> **Note:** The preprocessing scripts in `scripts/1_get_neg-pos_data/preprocess/` are adapted from the [chinn](https://github.com/mjflab/chinn) repository.

#### 2.2 Generate Training Data

For this tutorial, we'll use the pre-processed data in `data/proc/gm12878_ctcf` to generate the training data in `data/train/gm12878_ctcf`.


```shell
scripts/2_generate_traindata/gm12878_ctcf/Linux/1000bp.onehot.sh
```

after the script is done, you will find the train data in `data/train/gm12878_ctcf/1000bp.50ms.onehot/data.h5`.

> **note**: this proccess need at least 4GB memory. here is example scripts for 1000bp, when we use 2000 or 3000 bp as anchor size, we need more memory.

### Step 3: Model Training

#### BaseCNN Model

Train the BaseCNN model:
```shell
scripts/3_train/example/BaseCNN.sh
```

#### MCIENet Model

Train the MCIENet model:
```shell
scripts/3_train/example/MCIENet.sh
```

### Step 4: Explainable AI (XAI) Analysis

Perform model interpretation using various XAI methods. Here we demonstrate using DeepLIFT:
we can use the model we already trained under `output\best` as the model path.

Here we use DeepLift as the XAI method, you can also use other methods like LIME, SHAP, etc. Details arguments can be found in `get_attr.py`.

#### BaseCNN
```shell
python get_attr.py \
    --model_folder "output/best/BaseCNN-gm12878.ctcf-1kb" \
    --output_folder "output/XAI/BaseCNN-gm12878.ctcf-1kb" \
    --data_folder "data/train/gm12878_ctcf/1000bp.50ms.onehot" \
    --phases train val test \
    --batch_size 500 \
    --method "DeepLift" \
    --crop_center 500 \
    --crop_size 1000 \
    --use_cuda True
``` 

#### MCIENet
```shell
python get_attr.py \
    --model_folder "output/best/MCIENet-gm12878.ctcf-1kb" \
    --output_folder "output/XAI/MCIENet-gm12878.ctcf-1kb" \
    --data_folder "data/train/gm12878_ctcf/1000bp.50ms.onehot" \
    --phases train val test \
    --batch_size 500 \
    --method "DeepLift" \
    --crop_center 500 \
    --crop_size 1000 \
    --use_cuda True
``` 

> **note**: more example scripts can be found in `scripts\4_XAI\Linux`.

## References
- _Cao, Fan, et al. "Chromatin interaction neural network (ChINN): a machine learning-based method for predicting chromatin interactions from DNA sequences." Genome biology 22 (2021): 1-25. https://doi.org/10.1186/s13059-021-02453-5._
  - Github: https://github.com/mjflab/chinn
- _Zhou, Zhihan, et al. "Dnabert-2: Efficient foundation model and benchmark for multi-species genome." arXiv preprint arXiv:2306.15006 (2023). https://doi.org/10.48550/arXiv.2306.15006._
  - Github: https://github.com/MAGICS-LAB/DNABERT_2
  - Pretrain model: https://huggingface.co/zhihan1996/DNABERT-2-117M

## Citation

This version of implementation is only for learning purpose. For research, please refer to  and  cite from the following paper:
```
@inproceedings{ MCIENet,
  author = "Yen-Nan Ho and Jia-Ming Chang"
  title = "MCIENet: Multi-scale CNN-based Information Extraction from DNA Sequences for 3D chromatin interactions Prediction",
  booktitle = "",
  pages = "",
  year = "2025",
}
```
