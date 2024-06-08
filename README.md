# On the Robustness of Global Feature Effect Explanations

This repository is a supplement to the following paper:

> Hubert Baniecki, Giuseppe Casalicchio, Bernd Bischl, Przemyslaw Biecek. *On the Robustness of Global Feature Effect Explanations*. **ECML PKDD 2024**

```bibtex
@inproceedings{baniecki2024robustness,
    title     = {On the Robustness of Global Feature Effect Explanations},
    author    = {Hubert Baniecki and 
                 Giuseppe Casalicchio and 
                 Bernd Bischl and 
                 Przemyslaw Biecek},
    booktitle = {ECML PKDD},
    year      = {2024}
}
```

### Install the environment

1. `mamba env create -f env.yml`
2. install [OpenXAI](https://github.com/AI4LIFE-GROUP/OpenXAI):
    - download `https://github.com/AI4LIFE-GROUP/OpenXAI`
    - remove version of `torch`
    - `mamba activate robustfe`
    - `pip install .`

### Run the experiments

- `experiment1.ipynb` uses the algorithm [(Baniecki et al., 2022)](https://doi.org/10.1007/978-3-031-26409-2_8) implemented in `src` to perform experiments reported in Section 5.1
- `experiment2.ipynb`, `experiment2_plot.ipynb` perform experiments reported in Section 5.2
- `results` directory contains metadata of results from running `experiment1.ipynb` and `experiment2.ipynb`


### Check out also

Adebayo et al. **[Sanity Checks for Saliency Maps](https://doi.org/10.48550/arXiv.1810.03292)**. NeurIPS 2018

Baniecki et al. **[Fooling Partial Dependence via Data Poisoning](https://doi.org/10.1007/978-3-031-26409-2_8)**. ECML PKDD 2022

Gkolemis et al. **[RHALE: Robust and Heterogeneity-aware Accumulated Local Effects](https://doi.org/10.48550/arXiv.2309.11193)**. ECAI 2023

Lin et al. **[On the Robustness of Removal-Based Feature Attributions](https://doi.org/10.48550/arXiv.2306.07462)**. NeurIPS 2023