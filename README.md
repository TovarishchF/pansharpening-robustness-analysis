# Statistical Analysis of Pansharpening Method Robustness for Landscape Classification
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
*   Applying a robust statistical methodology (Consensus Ranking, Bootstrap analysis, ANOVA) to assess method stability and reliability.
*   Analyzing the impact of pansharpening on the accuracy of supervised classification.

## 🚀 Features

*   **End-to-End Pipeline:** A complete, automated pipeline from data preprocessing to statistical analysis and visualization.
*   **Multiple Pansharpening Algorithms:** Implementation of 10 different pansharpening methods from Component Substitution, Multi-Resolution Analysis, and Hybrid categories.
*   **No-Reference Quality Assessment:** Evaluation using metrics like QNR, SAM, ERGAS, SSIM, and Spatial Correlation.
*   **Robust Statistical Framework:** Utilizes Consensus Ranking, Bootstrap analysis, and ANOVA for reliable conclusions.
*   **Biome-Specific Analysis:** Evaluation of method performance and stability across urban, forest, and agricultural landscapes.
*   **Supervised Classification:** Integration of classification algorithms (Max Likelihood, Random Forest, SVM) to measure the practical impact of pansharpening.

## 🛠️ Implemented Methods

### Pansharpening Algorithms
*   **Component Substitution (CS):**
    *   IHS (Intensity-Hue-Saturation)
    *   Brovey Transform
    *   BT-H (Brovey with Histogram Matching)
    *   PCA (Principal Component Analysis)
*   **Multi-Resolution Analysis (MRA):**
    *   Wavelet-based
    *   HPF (High-Pass Filtering)
    *   SFIM (Smoothing Filter-based Intensity Modulation)
*   **Hybrid/Model-Based:**
    *   Gram-Schmidt (GS)
    *   GS2 (Gram-Schmidt Adaptive)
    *   PRACS (Partial Replacement Adaptive Component Substitution)

### Classification Algorithms
*   Maximum Likelihood
*   Random Forest
*   Support Vector Machines (SVM)

## 📋 Requirements

The project is implemented in Python. Key dependencies include:
*   Python 3.8+
*   W I P

A detailed list of requirements can be found in `requirements.txt`.

## 📁 Repository Structure
pansharpening-robustness-analysis/\
│\
├── data/                    # Directory for input data (Landsat 8 scenes, test polygons)\
│ ├── raw/                   # Original Landsat 8 scenes\
│ ├── processed/             # Preprocessed and cropped scenes\
│ └── polygons/              # Polygons used for classification and cropping\
│\
├── src/                     # Source code for the pipeline\
│ ├── preprocessing/         # Module for data loading, cropping, DOS1 correction\
│ ├── pansharpening/         # Implementations of all pansharpening algorithms\
│ ├── quality_metrics/       # Calculations for no-reference metrics (QNR, SAM, etc.)\
│ ├── classification/        # Implementations of classification algorithms\
│ ├── statistical_analysis/  # Consensus ranking, bootstrap, ANOVA scripts\
│ ├── utils/                 # Helper functions and logging\
│ ├── visualisation/         # Visualisation for easier analysis\
│ └── main_pipeline.py       # Main pipeline script\
│\
├── config.yaml/             # Main config file\
├── results/                 # Generated outputs (tables, plots, final ratings)\
├── requirements.txt         # Python environment dependencies\
├── LICENSE.txt              # MIT LICENSE\
└── README.md                # This file

# 👥 Authors
Egor D. Fanin - Primary Developer & Researcher\
Natalia A. Pozdnyakova - Scientific Supervisor

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

# 📧 Contact
For inquiries, please email faninhd@yandex.ru or faninhd@gmail.com
