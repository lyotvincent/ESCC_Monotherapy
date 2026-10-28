<p align="center">
  <img src="./abstract.jpg" width="600" alt="Abstract"/>
</p>

# Abstract
Neoadjuvant PD-1 monotherapy remains limited in unselected esophageal squamous cell carcinoma (ESCC), and biomarkers of benefit remain undefined. In a prospective PD-L1-selected cohort (combined positive score (CPS) ≥20; NCT06601309), 4 of 10 patients (40%) achieved a pathological complete response. We integrated longitudinal single-cell profiling, post-treatment spatial transcriptomics and multiplex imaging, and evaluated key features in an independent cohort (n = 43). Responders harbored CCL19+ fibroblast-centered niches containing CCR7+ lymphocytes within approximately 80 μm; CCR7 fluorescence intensity in lymphocytes was associated with response in the validation cohort. These niches comprised follicular and plasma cell-rich tertiary lymphoid structures and were enriched for clonal-expansion programs in tissue-resident exhausted CD8+ T cells and stress-activated B cells. Non-responders showed sparse CCL19+ stromal organization, collagen-rich immune–stromal interfaces and a spatially segregated NOTCH3-associated vascular mural program. These findings identify fibroblast-organized lymphoid niches and vascular-stromal barriers as spatial features associated with pathological response to PD-1 monotherapy in ESCC.

# Project Description

This repository contains the analysis scripts and annotation files required to reproduce the figures presented in our study.

## 📂 Repository Structure

The repository is organized as follows:

- **`annotation/`**: Contains cell type annotation files.
  > **Note on Data Availability:** The raw expression data is not included in this public repository due to patient privacy concerns. 

- **`Fig1/` – `Fig6/`**: Directories containing scripts to generate **Main Figures 1 through 6**.

- **`Supplementary Fig/`**: Directory containing scripts to generate **Supplementary Figures**.

- **`table/`**: Contains tabular data (e.g., statistical results, gene lists) generated automatically during the execution of the figure-generation scripts.

- **`FJTLS_Code/`**: Code for multiplex immunofluorescence analysis in Fig. 2g
## 📊 Data Access & Annotation Usage

To reproduce the analysis with the original patient data:
1. Please refer to the **"Data Availability"** section in our published paper for instructions on how to request access to the raw dataset.
2. Use the files in the `annotation/` folder to map cell type identities to your local dataset using the observation names (`obs_names`) as the key.

