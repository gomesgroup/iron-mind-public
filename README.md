# Iron Mind: Manuscript Figure Generation

<p align="left">
  <img src="logo.png" alt="Iron Mind Logo" width="600">
</p>

This repository contains the code to reproduce all figures from the Iron Mind manuscript. The repository provides Python scripts to generate publication-quality figures from benchmark optimization data across six chemical reaction datasets.
The preprint can be found on arXiv: https://arxiv.org/abs/2509.00103

## Website for testing optimizers and engaging in human-driven optimization campaigns

This work comes with a website for users to test out both the LLM and BO optimization strategies on the benchmark datasets.
Additionally, we are excited to offer humans the opportunity to conduct optimization campaigns on the datasets.
You can access the website [here](https://gomes.andrew.cmu.edu/iron-mind).

## Repository Structure

- `computed_descriptors/` - Descriptors used for Bayesian optimization methods
- `descriptors/` - Code to reproduce descriptors
- `figures/` - Python scripts for generating manuscript figures
- `histograms/` - Histogram plots showing objective distributions for each dataset
- `schematics/` - Chemical reaction schematics for each dataset

## Datasets

The repository works with six chemical reaction optimization datasets:
- **Buchwald-Hartwig** - C-N coupling reactions (yield optimization)
- **Suzuki-Miyaura A** - Cross-coupling reactions (yield optimization) 
- **Suzuki-Miyaura B** - Cross-coupling reactions (conversion optimization)
- **Reductive Amination** - Amine synthesis (conversion optimization)
- **N-Alkylation/Deprotection** - Two-step synthesis (yield optimization)
- **Chan-Lam Coupling** - C-N coupling reactions (multi-objective: desired vs undesired yield)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/gomesgroup/iron-mind-public.git
cd iron-mind-public
```

2. Create a conda environment with required dependencies:
```bash
conda create -n iron-mind-figures python=3.10
conda activate iron-mind-figures
```

3. Install the required packages:
```bash
pip install git+https://github.com/gomesgroup/olympus.git
pip install pandas numpy matplotlib seaborn scikit-learn plotly scipy
```

4. Setup to save plotly figures:
```bash
pip install kaleido
plotly_get_chrome
```

## Data Access

The benchmark optimization data used to generate these figures is available on [Hugging Face](https://huggingface.co/datasets/gomesgroup/iron-mind-data):

```bash
pip install huggingface-hub
hf auth login
hf download gomesgroup/iron-mind-data runs.zip --repo-type dataset --local-dir .
unzip runs.zip
```

This will produce the `runs/` directory in your current working directory. Use this path when generating figures.

## Figure Reproduction

Each figure script in the `figures/` directory can be run independently:

```bash
cd figures/
python figure_2.py
python figure_3.py
python figure_5_S12.py
...
```
Some figure scripts require the path to the `runs/` directory, be sure to provide the absolute path, opposed to the relative path.

To generate all figures:
```bash
bash generate_all_figures.sh <path_to_runs>
```
The `path_to_runs` must be an absolute path.

Generated figures are saved to `figures/pngs/` directory.

## Descriptors

The descriptors used for Bayesian optimization can be found in `computed_descriptors`.

## Citation

If you use this code or data, please cite our manuscript:

```bibtex
@article{macknight2025iron,
  title={Pre-trained knowledge elevates large language models beyond traditional chemical reaction optimizers},
  author={MacKnight, Robert and Regio, Jose Emilio and Ethier, Jeffrey G. and Baldwin, Luke A. and Gomes, Gabe},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```
