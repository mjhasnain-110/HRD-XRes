# HRD-XRes

**Transcriptomic inference of homologous recombination deficiency across bulk, single-cell and spatial resolution in high-grade serous ovarian cancer**

HRD-XRes is a pathway-constrained, domain-adapted, multi-task neural network that reads genomic homologous recombination deficiency (HRD) from gene expression. It is trained on TCGA-OV bulk RNA-seq with genomic scar labels, aligned without labels to single-cell and Visium spatial data, and applied frozen to new datasets.

This repository contains the full analysis pipeline, the model code, a prediction script and the scripts that regenerate every figure and supplementary table of the paper.

> \*\*Paper:\*\* \[Authors]. Transcriptomic inference of homologous recombination deficiency reveals a clonal tumour state and a spatial immune niche in ovarian cancer. \*\[Journal]\*, \[year]. doi: \[to be added]
> \*\*Trained models:\*\* Zenodo, doi: \[10.5281/zenodo.XXXXXXX] \[ to be added]

\---

## Main findings

* HRD is encoded robustly in tumour expression. HRD-XRes, an elastic net and the independent IdentifiHR classifier agree closely (AUROC 0.92–0.94 against WGS-defined HRD in malignant cells; inter-model ρ = 0.87). A widely cited 230-gene signature does not (AUROC 0.57).
* Inferred HRD follows WGS mutational subtypes, is consistent across 126 metastatic sites (ICC 0.92) and is unchanged by chemotherapy, consistent with a clonal, patient-level state.
* Higher inferred HRD predicts longer overall survival across 16 external cohorts (n = 1,982; HR 0.83 per SD).
* In Visium sections, HRD-high tumour regions co-localise with dendritic and mast cells (19 and 20 of 20 sections). The association is not explained by transcript admixture or by overall immune abundance.
* Platinum resistance is not predictable from expression by any model tested.

\---

## Model

|Component|Setting|
|-|-|
|Input|8,702 genes (log CPM, z-scored per cohort)|
|Pathway layer|masked linear map to 1,509 Reactome pathways|
|Encoder|LayerNorm, dropout 0.3, 256 → 64 units (GELU)|
|Task heads|HRD score, HRD status, BRCA1/2, platinum, overall survival (Cox), purity|
|Domain alignment|gradient reversal to a discriminator over 6 data domains|
|Training|nested CV (5 outer × 3 inner folds); final model = 10-seed ensemble|

\---

## Repository layout

```
HRD-XRes/
├── config/config.yaml        all paths, label rules, model and training settings
├── hrdx/                     Python package (model, trainer, data, metrics, baselines)
├── scripts/                  pipeline steps 00–20, predict.py, R/ helpers
├── tests/                    synthetic-data generator for end-to-end testing
├── envs/environment.yml      conda environment
├── run\_pipeline.sh           runs steps 0–13 (or a range)
└── submit\_slurm.sh           optional cluster submission
```

The pipeline creates `Inputs/`, `Processed/`, `Models/`, `Results/` and `Logs/`. These are not tracked in git.

\---

## Installation

```bash
git clone https://github.com/\[user]/HRD-XRes.git
cd HRD-XRes
conda env create -f envs/environment.yml
conda activate hrdx
```

A GPU is optional; the full pipeline runs on CPU.

R is needed for two steps:

* **Step 0:** curatedOvarianData (Bioconductor).
* **Step 15:** IdentifiHR. Install it with:

```bash
  Rscript -e 'BiocManager::install(c("edgeR","SummarizedExperiment")); remotes::install\_github("DavidsonGroup/IdentifiHR", dependencies=FALSE)'
  ```

\---

## Data

All data are public:

|Dataset|Source|
|-|-|
|TCGA-OV expression|UCSC Xena|
|TCGA-OV clinical|TCGA Pan-Cancer Clinical Data Resource|
|TCGA-OV HRD, DDR and copy number|Knijnenburg et al. 2018; ABSOLUTE|
|MSK SPECTRUM|CELLxGENE and the original supplementary tables|
|Single-cell, pre/post chemotherapy|GSE165897|
|Visium|GSE211956, GSE189843|
|Bulk microarray|curatedOvarianData; GSE140082, GSE63885, GSE51373|
|Pathways|Reactome GMT|

`download\_hgsoc\_hrd\_data.sh` downloads them into `Inputs/hgsoc\_hrd\_data/`. Paths are set in `config/config.yaml`.

\---

## Running the pipeline

Steps 0–13 run in sequence:

```bash
bash run\_pipeline.sh            # steps 0–13
bash run\_pipeline.sh 7 13       # a range of steps
```

The remaining steps run one command at a time, for example:

```bash
nohup python scripts/14\_robustness.py > Logs/14.out 2>\&1 \&
```

|Step|Script|Purpose|
|-|-|-|
|0|`R/export\_curatedOvarianData.R`|export serous cohorts with survival (TCGA excluded)|
|0b|`00b\_update\_peng\_symbols.py`|update the Peng 230-gene signature to current gene symbols|
|1a|`01a\_compute\_hrd\_scar.py`|recompute HRD-LOH, TAI and LST from ABSOLUTE segments|
|1|`01\_prepare\_tcga.py`|TCGA-OV expression and labels|
|2|`02\_prepare\_spectrum.py`|SPECTRUM pseudobulk, metacells, reference and WGS labels (streams the 11 GB atlas)|
|3|`03\_prepare\_spatial.py`|Denisenko and Stur Visium data and response labels|
|4|`04\_prepare\_bulk.py`|external GEO and curatedOvarianData cohorts|
|5|`05\_prepare\_gse165897.py`|paired pre/post-chemotherapy metacells|
|6|`06\_build\_features.py`|shared gene space, Reactome mask, model inputs|
|7|`07\_cross\_validation.py`|nested CV of HRD-XRes, ablations and baselines|
|7b|`07b\_peng\_baseline.py`|Peng signature in three orientations|
|8|`08\_train\_final.py`|10-seed final ensembles|
|9|`09\_validate\_bulk.py`|external survival, platinum, meta-analysis|
|10|`10\_validate\_singlecell.py`|WGS validation, subtypes, site ICC, allelic imbalance, pre/post|
|11|`11\_validate\_spatial.py`|spot maps, deconvolution, response, neighbourhoods|
|12|`12\_interpret.py`|integrated-gradient attributions, domain alignment|
|13, 13b|`13\_make\_figures.py`, `13b\_peng\_figures.py`|main and Peng figures|
|14|`14\_robustness.py`|gene loss, calibration, P(HRD) survival meta-analysis|
|15|`15\_identifihr.py`|IdentifiHR head-to-head; adjusted spatial neighbourhoods|
|16|`16\_composition\_check.py`|WGS-labelled cell-type composition (model-free)|
|17|`17\_mixing\_test.py`|in-silico admixture test|
|18|`18\_new\_figures.py`|figures for steps 14–17|
|19|`19\_purity\_check.py`|purity-adjusted spatial response tests|
|20|`20\_fig1\_schematic.py`|Figure 1|

Training resumes from saved checkpoints if interrupted. A full CPU run of step 7 takes several hours.

\---

## Trained models

Model checkpoints are too large for GitHub and are archived on Zenodo \[doi]:

* **`HRD-XRes\_final\_models.tar.gz`:** the deployed 10-seed ensemble (`Models/final/full/`, plus `no\_da/`) and `Processed/features/` (gene list, pathway mask, cohort statistics). This is all that `predict.py` needs.
* **`HRD-XRes\_cv\_models.tar.gz`:** all cross-validation and inner-loop networks, for full reproducibility.

```bash
wget https://zenodo.org/records/XXXXXXX/files/HRD-XRes\_final\_models.tar.gz
tar xzf HRD-XRes\_final\_models.tar.gz
```

\---

## Predicting HRD in new data

```bash
python scripts/predict.py --input cohort.tsv --value-type log2 --output predictions.csv
```

The input is a gene × sample matrix with gene symbols. `--value-type` accepts `counts` or `log2`.

The output holds, per sample:

|Column|Meaning|
|-|-|
|`pred\_hrd`|predicted HRD score|
|`p\_hr`|P(HRD)|
|`p\_brca`|P(BRCA1/2 alteration)|
|`p\_plat`|P(platinum resistance)|
|`risk`|survival risk score|
|`pred\_purity`|predicted tumour purity|

For single-cell data, score malignant-cell pseudobulk or metacells. Absolute P(HRD) shifts with non-malignant admixture.

\---

## Reproducing the paper

|Paper item|Generated by|Output|
|-|-|-|
|Figure 1|step 20|`Results/figures/Fig1\_study\_design`|
|Figures 2–5|steps 13, 13b, 18|`Results/figures/`|
|Supplementary Tables S1–S13|steps 1a, 7, 7b, 9–12, 14–19|CSV files in `Results/` subfolders|

The figure and table key in the manuscript maps each panel and table to its source file.

\---

## Testing

```bash
python tests/make\_synthetic\_inputs.py /tmp/hrdx\_test
```

This builds a small synthetic copy of the input folder with identical file formats, so the pipeline can be tested end to end without downloading data.

\---

## Citation

If you use HRD-XRes, please cite the paper above. Please also cite IdentifiHR (Weir et al., *Commun Med* 2026) if you use the head-to-head comparison.

## License

\[MIT]

## Contact

\[Hasnain Mirza Jawad Ul, mirza.jawad@vu.edu.p# HRD-XRes



\*\*Transcriptomic inference of homologous recombination deficiency across bulk, single-cell and spatial resolution in high-grade serous ovarian cancer\*\*



HRD-XRes is a pathway-constrained, domain-adapted, multi-task neural network that reads genomic homologous recombination deficiency (HRD) from gene expression. It is trained on TCGA-OV bulk RNA-seq with genomic scar labels, aligned without labels to single-cell and Visium spatial data, and applied frozen to new datasets.



This repository contains the full analysis pipeline, the model code, a prediction script and the scripts that regenerate every figure and supplementary table of the paper.



> \*\*Paper:\*\* \[Authors]. Transcriptomic inference of homologous recombination deficiency reveals a clonal tumour state and a spatial immune niche in ovarian cancer. \*\[Journal]\*, \[year]. doi: \[to be added]

> \*\*Trained models:\*\* Zenodo, doi: \[10.5281/zenodo.XXXXXXX]



\---



\## Main findings



\- HRD is encoded robustly in tumour expression. HRD-XRes, an elastic net and the independent IdentifiHR classifier agree closely (AUROC 0.92–0.94 against WGS-defined HRD in malignant cells; inter-model ρ = 0.87). A widely cited 230-gene signature does not (AUROC 0.57).

\- Inferred HRD follows WGS mutational subtypes, is consistent across 126 metastatic sites (ICC 0.92) and is unchanged by chemotherapy, consistent with a clonal, patient-level state.

\- Higher inferred HRD predicts longer overall survival across 16 external cohorts (n = 1,982; HR 0.83 per SD).

\- In Visium sections, HRD-high tumour regions co-localise with dendritic and mast cells (19 and 20 of 20 sections). The association is not explained by transcript admixture or by overall immune abundance.

\- Platinum resistance is not predictable from expression by any model tested.



\---



\## Model



| Component | Setting |

|---|---|

| Input | 8,702 genes (log CPM, z-scored per cohort) |

| Pathway layer | masked linear map to 1,509 Reactome pathways |

| Encoder | LayerNorm, dropout 0.3, 256 → 64 units (GELU) |

| Task heads | HRD score, HRD status, BRCA1/2, platinum, overall survival (Cox), purity |

| Domain alignment | gradient reversal to a discriminator over 6 data domains |

| Training | nested CV (5 outer × 3 inner folds); final model = 10-seed ensemble |



\---



\## Repository layout



```

HRD-XRes/

├── config/config.yaml        all paths, label rules, model and training settings

├── hrdx/                     Python package (model, trainer, data, metrics, baselines)

├── scripts/                  pipeline steps 00–20, predict.py, R/ helpers

├── tests/                    synthetic-data generator for end-to-end testing

├── envs/environment.yml      conda environment

├── download\_hgsoc\_hrd\_data.sh  downloads all public input data

├── run\_pipeline.sh           runs steps 0–13 (or a range)

└── submit\_slurm.sh           optional cluster submission

```



The pipeline creates `Inputs/`, `Processed/`, `Models/`, `Results/` and `Logs/`. These are not tracked in git.



\---



\## Installation



```bash

git clone https://github.com/\[user]/HRD-XRes.git

cd HRD-XRes

conda env create -f envs/environment.yml

conda activate hrdx

```



A GPU is optional; the full pipeline runs on CPU.



R is needed for two steps:

\- \*\*Step 0:\*\* curatedOvarianData (Bioconductor).

\- \*\*Step 15:\*\* IdentifiHR. Install it with:

&#x20; ```bash

&#x20; Rscript -e 'BiocManager::install(c("edgeR","SummarizedExperiment")); remotes::install\_github("DavidsonGroup/IdentifiHR", dependencies=FALSE)'

&#x20; ```



\---



\## Data



All data are public:



| Dataset | Source |

|---|---|

| TCGA-OV expression | UCSC Xena |

| TCGA-OV clinical | TCGA Pan-Cancer Clinical Data Resource |

| TCGA-OV HRD, DDR and copy number | Knijnenburg et al. 2018; ABSOLUTE |

| MSK SPECTRUM | CELLxGENE and the original supplementary tables |

| Single-cell, pre/post chemotherapy | GSE165897 |

| Visium | GSE211956, GSE189843 |

| Bulk microarray | curatedOvarianData; GSE140082, GSE63885, GSE51373 |

| Pathways | Reactome GMT |



`download\_hgsoc\_hrd\_data.sh` (repository root) downloads them into `Inputs/hgsoc\_hrd\_data/`. Paths are set in `config/config.yaml`.



\---



\## Running the pipeline



Steps 0–13 run in sequence:



```bash

bash run\_pipeline.sh            # steps 0–13

bash run\_pipeline.sh 7 13       # a range of steps

```



The remaining steps run one command at a time, for example:



```bash

nohup python scripts/14\_robustness.py > Logs/14.out 2>\&1 \&

```



| Step | Script | Purpose |

|---|---|---|

| 0 | `R/export\_curatedOvarianData.R` | export serous cohorts with survival (TCGA excluded) |

| 0b | `00b\_update\_peng\_symbols.py` | update the Peng 230-gene signature to current gene symbols |

| 1a | `01a\_compute\_hrd\_scar.py` | recompute HRD-LOH, TAI and LST from ABSOLUTE segments |

| 1 | `01\_prepare\_tcga.py` | TCGA-OV expression and labels |

| 2 | `02\_prepare\_spectrum.py` | SPECTRUM pseudobulk, metacells, reference and WGS labels (streams the 11 GB atlas) |

| 3 | `03\_prepare\_spatial.py` | Denisenko and Stur Visium data and response labels |

| 4 | `04\_prepare\_bulk.py` | external GEO and curatedOvarianData cohorts |

| 5 | `05\_prepare\_gse165897.py` | paired pre/post-chemotherapy metacells |

| 6 | `06\_build\_features.py` | shared gene space, Reactome mask, model inputs |

| 7 | `07\_cross\_validation.py` | nested CV of HRD-XRes, ablations and baselines |

| 7b | `07b\_peng\_baseline.py` | Peng signature in three orientations |

| 8 | `08\_train\_final.py` | 10-seed final ensembles |

| 9 | `09\_validate\_bulk.py` | external survival, platinum, meta-analysis |

| 10 | `10\_validate\_singlecell.py` | WGS validation, subtypes, site ICC, allelic imbalance, pre/post |

| 11 | `11\_validate\_spatial.py` | spot maps, deconvolution, response, neighbourhoods |

| 12 | `12\_interpret.py` | integrated-gradient attributions, domain alignment |

| 13, 13b | `13\_make\_figures.py`, `13b\_peng\_figures.py` | main and Peng figures |

| 14 | `14\_robustness.py` | gene loss, calibration, P(HRD) survival meta-analysis |

| 15 | `15\_identifihr.py` | IdentifiHR head-to-head; adjusted spatial neighbourhoods |

| 16 | `16\_composition\_check.py` | WGS-labelled cell-type composition (model-free) |

| 17 | `17\_mixing\_test.py` | in-silico admixture test |

| 18 | `18\_new\_figures.py` | figures for steps 14–17 |

| 19 | `19\_purity\_check.py` | purity-adjusted spatial response tests |

| 20 | `20\_fig1\_schematic.py` | Figure 1 |



Training resumes from saved checkpoints if interrupted. A full CPU run of step 7 takes several hours.



\---



\## Trained models



Model checkpoints are too large for GitHub and are archived on Zenodo \[doi]:



\- \*\*`HRD-XRes\_final\_models.tar.gz`:\*\* the deployed 10-seed ensemble (`Models/final/full/`, plus `no\_da/`) and `Processed/features/` (gene list, pathway mask, cohort statistics). This is all that `predict.py` needs.

\- \*\*`HRD-XRes\_cv\_models.tar.gz`:\*\* all cross-validation and inner-loop networks, for full reproducibility.



```bash

wget https://zenodo.org/records/XXXXXXX/files/HRD-XRes\_final\_models.tar.gz \[ to be added]

tar xzf HRD-XRes\_final\_models.tar.gz

```



\---



\## Predicting HRD in new data



```bash

python scripts/predict.py --input cohort.tsv --value-type log2 --output predictions.csv

```



The input is either a gene × sample TSV (gene symbols in the first column) or an `.h5ad` file (observations = samples, cells or spots; variables = gene symbols). `--value-type` is `counts` (default), `log2` or `linear`. Add `--variant no\_da` to use the ensemble without domain adaptation.



The output holds, per sample:



| Column | Meaning |

|---|---|

| `pred\_hrd` | predicted HRD score |

| `p\_hr` | P(HRD) |

| `p\_brca` | P(BRCA1/2 alteration) |

| `p\_plat` | P(platinum resistance) |

| `risk` | survival risk score |

| `pred\_purity` | predicted tumour purity |



For single-cell data, score malignant-cell pseudobulk or metacells. Absolute P(HRD) shifts with non-malignant admixture.



\---



\## Reproducing the paper



| Paper item | Generated by | Output |

|---|---|---|

| Figure 1 | step 20 | `Results/figures/Fig1\_study\_design` |

| Figures 2–5 | steps 13, 13b, 18 | `Results/figures/` |

| Supplementary Tables S1–S13 | steps 1a, 7, 7b, 9–12, 14–19 | CSV files in `Results/` subfolders |



The figure and table key in the manuscript maps each panel and table to its source file.



\---



\## Testing



```bash

python tests/make\_synthetic\_inputs.py /tmp/hrdx\_test

```



This builds a small synthetic copy of the input folder with identical file formats, so the pipeline can be tested end to end without downloading data.



\---



\## Citation



If you use HRD-XRes, please cite the paper above. Please also cite IdentifiHR (Weir et al., \*Commun Med\* 2026) if you use the head-to-head comparison.



\## License



\[MIT]



\## Contact



\[Hasnain Mirza Jawad ul; mirza.jawad@vu.edu.pk; mjhasnain@gmail.com]

