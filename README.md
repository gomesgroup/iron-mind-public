# Iron Mind: Manuscript Figure Generation

This repository contains the code to reproduce all figures from the Iron Mind manuscript. The repository provides Python scripts to generate publication-quality figures from benchmark optimization data across six chemical reaction datasets.

## Repository Structure

- `figures/` - Python scripts for generating manuscript figures
- `histograms/` - Histogram plots showing objective distributions for each dataset
- `schematics/` - Chemical reaction schematics for each dataset

## Datasets

The repository works with six chemical reaction optimization datasets:
- **Buchwald-Hartwig** - C-N coupling reactions (yield optimization)
- **Suzuki-Miyaura A** - Cross-coupling reactions (yield optimization) 
- **Suzuki-Miyaura B** - Cross-coupling reactions (conversion optimization)
- **Reductive Amination** - Amine synthesis (percent conversion optimization)
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

The benchmark optimization data used to generate these figures is available on Hugging Face:

```bash
pip install huggingface-hub
hf auth login
hf download gomesgroup/iron-mind-data runs.zip --repo-type dataset --local-dir .
unzip runs.zip
```

This will produce the `runs/` directory in your current working directory. Use this path when generating figures.

## Usage

Each figure script in the `figures/` directory can be run independently:

```bash
cd figures/
python figure_2.py    # Dataset objective histograms
python figure_3.py    # Optimization complexity analysis
python figure_5.py    # LLM optimization performance
python figure_6.py    # Duplicate suggestion analysis  
python figure_7.py    # Entropy analysis
python figure_SI_convergence.py  # Convergence analysis
```

To generate all figures:
```bash
bash generate_all_figures.sh <path_to_runs>
```

Generated figures are saved to `figures/pngs/` directory.

## Figure Descriptions

- **Figure 2**: Histograms of objective values across all datasets
- **Figure 3**: Radar charts showing optimization complexity metrics
- **Figure 5**: Boxplots of LLM optimization performance by provider
- **Figure 6**: Analysis of duplicate suggestions in LLM methods
- **Figure 7**: Entropy analysis
- **Figure SI Convergence Analysis**: Convergence analysis

## Citation

If you use this code or data, please cite our manuscript:

```bibtex
@article{macknight2025iron,
  title={The Iron Mind Project: Benchmarking Artificial Intelligence on the Optimization of Scientific Experimental Campaigns},
  author={MacKnight, Robert and Regio, Jose Emilio and Ethier, Jeffrey G. and Baldwin, Luke A. and Gomes, Gabe},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```
