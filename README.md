# avalanche

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
