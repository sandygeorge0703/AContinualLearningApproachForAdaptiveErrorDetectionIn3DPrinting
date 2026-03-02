# A Continual Learning Approach for Adaptive Error Detection in 3D Printing

This repository contains the dissertation and code from my MEng Aerospace Engineering Individual Project at the University of Nottingham. The project received a mark of 83% for the final report and 82% overall for the module.
## Abstract

The increasing popularity of 3D printing in additive manufacturing is currently hindered by its susceptibility to errors, 
limiting its widespread adoption for end-use applications. Traditional real-time in-process control and monitoring 
methods, such as acoustic sensors and camera-based systems, predominantly rely on extensive and diverse datasets, 
resulting in significant computational demands. Although machine learning techniques using Convolutional Neural 
Networks (CNNs) have been proposed to address these challenges, they are similarly constrained by continuous reliance 
on large datasets and complete model retraining when encountering evolving data distributions or operational 
conditions. This paper introduces a novel continual learning approach based on Elastic Weight Consolidation (EWC), 
enabling a CNN model to incrementally adapt to new data distributions while retaining previously acquired 3D printing 
knowledge. This approach mitigates the necessity for repeated retraining, making it ideally suited for a “first-time-right” 
manufacturing approach in dynamic 3D printing environments. A baseline CNN model was first developed using the full 
set of hot-end temperature classes from a curated subset of the CAXTON 3D printing dataset, establishing the upper 
bound for classification performance. To evaluate continual learning, a benchmark was constructed with two sequential 
tasks, and the model was initially trained using a Naïve continual learning approach. This resulted in complete 
catastrophic forgetting of the first task, highlighting the limitations of sequential learning without memory retention. In 
contrast, the integration of the EWC strategy led to noticeable improvements. With fine-tuned regularisation, the model 
retained an accuracy of 43.71% on the initial task, demonstrating EWC’s ability to mitigate catastrophic forgetting. 
These findings indicate that, with carefully tuned EWC parameters, a continual learning approach can successfully 
mitigate catastrophic forgetting, adapt to evolving data, and substantially improve upon traditional methods. 
Consequently, this method holds significant potential for enhancing product quality, reducing waste material, lowering 
operational costs, and minimising environmental impact, thereby supporting a more efficient, robust, and sustainable 
future in additive manufacturing. 

## Key Links

- 📄 **Full Dissertation Document**  
  [View Dissertation (PDF)](https://drive.google.com/file/d/15OJgNdXYkY8z8Ed3q7vxx4whlX5tfeLF/view?usp=sharing)

- 📊 **Dataset Examples**  
  Example dataset demonstrating the type of data used for training and testing:  
  https://www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8

## Overview

This repository contains experiments built with the [Avalanche](https://avalanche.continualai.org/) continual learning library (PyTorch) for image-based classification. It includes training scripts/notebooks, experiment runners, datasets, saved runs, and plotting utilities.

## Repository layout

> Note: the repository contains some large/generated folders (for example `my_new_env/`) that are not required to run the code.

Top-level items you will commonly interact with:

- **Training / experiment entry points**
  - `train_new.py` / `train_new.ipynb` — main training workflow (data loading, filtering, training, evaluation, plots).
  - `EWC.py` / `EWC.ipynb` — Elastic Weight Consolidation (EWC) continual learning experiment.
  - `NAIVE.ipynb` — baseline/naive continual learning approach (notebook).
  - `balanced_naive_model_class_incremental*.{py,ipynb}` — class-incremental experiments on a balanced dataset.

- **Experiment runners**
  - `run_experiments.sh` — runs a training script multiple times and writes logs per run.
  - `run_benchmark_experiments.sh` — runs benchmark experiments multiple times and writes logs per run.

- **Models**
  - `models/` — model definitions (for example `models/cnn_models.py` is imported by several scripts).

- **Data / datasets**
  - `data/`, `dataset/`, `increased_data/` — local datasets and dataset-related artifacts used by experiments.
  - `balanced_dataset_filtered.csv` — example CSV used by some balanced/class-incremental experiments.

- **Experiment outputs**
  - `run_*/`, `tb_data/`, `loss_plots/`, `benchmark_model_data/`, `benchmark_experiment/`, `ewc_experiment/`, `naive_experiment*/` — output folders produced by runs (logs, tensorboard events, plots, intermediate artifacts).
  - `process_logs.py` — utility to parse per-run logs and generate a combined loss plot.
  - `log.txt`, `acc_plot_exp1.png` — example outputs/checkpoints from prior runs.

- **Misc**
  - `Code for Giovanna/` and `Experiment 1/`, `Experiment 2/` — project-specific working directories and experiment snapshots.

## Setup

### 1) Prerequisites

- Python 3.9+ recommended
- PyTorch (CPU or CUDA)
- (Optional) Jupyter if you want to run notebooks

### 2) Clone

```bash
git clone https://github.com/sandygeorge0703/avalanche.git
cd avalanche
```

### 3) Create a virtual environment

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

You will also need Avalanche itself:

```bash
pip install avalanche-lib
```

If you plan to run notebooks:

```bash
pip install jupyter
```

### 5) Data configuration

Some scripts currently contain **absolute, machine-specific paths** (for example `C:\Users\...` or `/gpfs01/...`) for:

- CSV metadata file path (e.g. `data/dataset.csv`)
- Root image folder path (e.g. `caxton_dataset/print24`)

To run the code on a new machine, update those constants near the top of the script you are running (search for `CSV_FILE_PATH`, `ROOT_DIR_PATH`, `csv_file`, `root_dir`).

A recommended approach is to standardize on a relative layout like:

```text
avalanche/
  data/
    dataset.csv
  caxton_dataset/
    print24/
      image-0001.jpg
      ...
```

…and then change scripts to use relative paths (or environment variables).

### 6) Run an experiment

Run a Python script directly:

```bash
python train_new.py
```

Or run a notebook:

```bash
jupyter notebook
```

Or run the batch runner scripts (paths inside the scripts may need editing for your environment):

```bash
bash run_experiments.sh
bash run_benchmark_experiments.sh
```

## Notes / gotchas

- `my_new_env/` appears to be a committed virtual environment (generated files). It is usually best to avoid committing virtualenvs and instead recreate them locally.
- Runner scripts reference specific HPC paths (e.g. `/gpfs01/...`). Update those before running locally.

## Contributing

If you create new experiments, consider adding them under a dedicated folder (e.g. `experiments/`) and updating this README with the new entry points.
