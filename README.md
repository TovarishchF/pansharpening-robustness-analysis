# Statistical Analysis of Pansharpening Method Robustness for Landscape Classification
🌐 **Languages:** [English](README.md) | [Русский](README_ru.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

*A course project at the Department of Cartography and Geoinformatics  
Saint Petersburg State University (SPbU).*

This repository contains the source code and methodology for the research project **"Statistical Analysis of Pansharpening Method Robustness for Landscape Classification"**.

The study performs a comprehensive comparative analysis of various pansharpening algorithms, evaluating their spectral fidelity, spatial quality, and robustness across different biomes (urban, forest, agricultural) using a statistical framework. The ultimate goal is to provide data-driven recommendations for selecting optimal pansharpening methods to improve the accuracy of satellite image classification.

## 📖 Overview

Medium-resolution satellites like Landsat 8 provide valuable data but are limited by their spatial resolution for detailed landscape analysis. Pansharpening techniques aim to enhance this resolution by fusing multispectral (MS) bands with a high-resolution panchromatic (PAN) band. However, the performance and stability of these methods can vary significantly depending on the landscape type.

This research addresses this challenge by:
*   Implementing a wide range of classical and modern pansharpening algorithms.
*   Evaluating their performance using no-reference quality metrics.
*   Applying a robust statistical methodology (Consensus Ranking, Bootstrap analysis, Agreement analysis) to assess method stability and reliability.
*   Analyzing the impact of pansharpening on the accuracy of supervised classification.

## 🚀 Features

*   **End-to-End Pipeline:** A complete, automated pipeline from data preprocessing to statistical analysis and visualization.
*   **Multiple Pansharpening Algorithms:** Implementation of 9 different pansharpening methods from Component Substitution, Multi-Resolution Analysis, and Hybrid categories.
*   **No-Reference Quality Assessment:** Evaluation using metrics like QNR, Gradient Similarity, Entropy, and Spatial Correlation.
*   **Robust Statistical Framework:** Utilizes Consensus Ranking, Bootstrap analysis, and Agreement analysis based on Kendall's W for reliable conclusions.
*   **Biome-Specific Analysis:** Evaluation of method performance and stability across urban, forest, and agricultural landscapes.
*   **Supervised Classification:** Integration of classification algorithms (Max Likelihood, Random Forest, XGBoost) to measure the practical impact of pansharpening.

## 🛠️ Implemented Methods

### Pansharpening Algorithms
*   **Component Substitution (CS):**
    *   BT (Brovey Transform)
    *   BT-H (Brovey Transform with Histogram Matching)
    *   PCA (Principal Component Analysis)
*   **Multi-Resolution Analysis (MRA):**
    *   ATWT (À Trous Wavelet Transform)
    *   HPF (High-Pass Filtering)
    *   SFIM (Smoothing Filter-based Intensity Modulation)
*   **Hybrid/Model-Based:**
    *   GS (Gram-Schmidt)
    *   GS2 (Gram-Schmidt Adaptive)
    *   PRACS (Partial Replacement Adaptive Component Substitution)

### Classification Algorithms
*   Maximum Likelihood
*   Random Forest
*   XGBoost

## 📋 Requirements

The project is implemented in Python. Key dependencies include:
*   Python 3.8+
*   Core scientific Python stack:
    * NumPy
    * SciPy
    * pandas
    * matplotlib
    * seaborn
    * scikit-learn
    * scikit-image
    * PyYAML

A detailed list of requirements can be found in `requirements.txt`.

## ⚙️ Installation

To install and run the project locally, follow the steps below.

### 1. Clone the repository

```bash
git clone https://github.com/TovarishchF/pansharpening-robustness-analysis.git
cd pansharpening-robustness-analysis
```
### 2. (Optional but recommended) Create a virtual environment
```bash
python -m venv .venv
```
Activate the environment:

* Git Bash / Linux / macOS
```bash
source .venv/bin/activate
```
* Windows (PowerShell / CMD)
```bat
.venv\Scripts\activate
```
### 3. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🔁 Reproducibility

All experiments in this project are fully reproducible using the provided source code,
configuration files, and fixed random seeds specified in `config.yaml`.

Due to data volume constraints, the original satellite imagery (Landsat 8 scenes)
is not included in the repository.

The exact input data can be independently retrieved from the **USGS EarthExplorer**
platform using the provided metadata file  
`LC08_L1TP_185018_20250718_20250726_02_T1_MTL.txt`
and the following scene information:

- **Landsat Product ID:** LC08_L1TP_185018_20250718_20250726_02_T1  
- **Acquisition date:** 2025-07-18  
- **Satellite / Sensor:** Landsat 8 OLI/TIRS  
- **WRS Path / Row:** 185 / 18  

All Landsat data used in this study are publicly available and distributed by the
U.S. Geological Survey (USGS) without usage restrictions.

## 📁 Repository Structure
pansharpening-robustness-analysis/\
│\
├── data/                    # Directory for input data (Landsat 8 scenes, test polygons)\
│ ├── raw/                   # Original Landsat 8 scenes\
│ ├── intermediate/          # Preprocessed and cropped scenes\
│ ├── processed/             # Classification and pansharpening results\
│ └── polygons/              # Polygons used for classification and cropping\
│\
├── src/                     # Source code for the pipeline\
│ ├── preprocessing/         # Module for data loading, cropping, DOS1 correction\
│ ├── pansharpening/         # Implementations of all pansharpening algorithms and metrics\
│ │ ├── cs/                  # Component Substitution methods\
│ │ ├── model_based/         # Hybrid/Model-based methods\
│ │ └── mra/                 # Multi-Resolution Analysis methods\
│ ├── classification/        # Implementations of classification algorithms and metrics & train/test split\
│ ├── statistical_analysis/  # Descriptive analysis, consensus ranking, bootstrap, agreement analysis (Kendall’s W)\
│ ├── utils/                 # Helper functions and logging\
│ ├── visualisation/         # Visualisation for easier analysis\
│\
├── results/                 # Generated outputs (tables, plots, final ratings)\
├── logs/                 # Automatic logs with daily rotation\
│\
├── main_pipeline.py         # Main pipeline script\
├── config.yaml             # Main config file\
├── requirements.txt         # Python environment dependencies\
├── LICENSE.txt              # MIT LICENSE\
└── README.md                # This file

## 📌 Project Information

**Author:** Egor D. Fanin  
**Supervisor:** Natalia A. Pozdnyakova  

This project was developed as part of a course research at the Department of Cartography and Geoinformatics, Saint Petersburg State University.

**License:** MIT  
(see `LICENSE` for details)