\# SPDC with BBO Crystal: Quantum Entanglement \& Machine Learning



\[!\[Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\## Overview



Complete quantum optics project for Spontaneous Parametric Down-Conversion (SPDC) using a BBO crystal. Integrates theory, simulation, experimental data analysis, machine learning, and sparsity-driven entanglement detection.



\## Key Results



| Metric | Value | Significance |

|--------|-------|--------------|

| EPR Parameter | 0.1712 | ENTANGLED |

| Bell Parameter S | 2.6872 | VIOLATION |

| State Fidelity | 0.9622 | 96 percent match |

| Concurrence | 0.9274 | Strong entanglement |

| Power Prediction R2 | 0.9294 | Random Forest |

| LSTM Forecast R2 | 0.8269 | Time series |

| Raw SNR | 3.2849 | Baseline |

| Lasso SNR | 3.2881 | Improved |



\## Project Structure



SPDC\_Project/

|

|--- 01\_Thesis\_Figures/

|    |--- 17 publication-ready figures

|

|--- 02\_Python\_Scripts/

|    |--- Main/ (core quantum optics)

|    |--- Analysis/ (data exploration)

|    |--- Pipeline/ (end-to-end pipelines)

|    |--- Visualization/ (plotting utilities)

|

|--- 03\_Data/

|    |--- ECMBI\_Tomography/

|         |--- FINAL\_RESULTS/ (MAIN RESULTS)

|         |--- 12 raw CSV files

|

|--- 05\_Results/

|    |--- CSV\_Data/ (15 files)

|    |--- Graphs/ (12 files)

|    |--- PDF\_Reports/ (7 files)

|

|--- 06a\_SNLO\_Graphs/

&#x20;    |--- SNLO simulation outputs



\## Quick Start



Clone the repository:



git clone https://github.com/YOUR\_USERNAME/SPDC\_Project.git

cd SPDC\_Project



Install dependencies:



pip install -r requirements.txt



Run the complete pipeline:



cd 02\_Python\_Scripts

python spdc\_pipeline.py



View the results:



start ../03\_Data/ECMBI\_Tomography/FINAL\_RESULTS/



\## Dependencies



\- numpy

\- pandas

\- matplotlib

\- scikit-learn

\- tensorflow

\- qutip

\- scipy

\- seaborn



\## Paper Implementation



This project implements sparsity-driven entanglement detection from arXiv:2511.12546.



\- Equation 1: Sample covariance matrix

\- Equation 2: L1-regularization with LassoCV

\- Equation 3: SNR calculation

\- EPR Criterion: Less than 1 equals entangled



\## Results Location



All final results are in: 03\_Data/ECMBI\_Tomography/FINAL\_RESULTS/



Files:

\- epr\_results.csv (EPR = 0.1712)

\- snr\_results.csv (SNR improvement)

\- heatmap\_comparison.png

\- comparison\_plot.png

\- summary\_report.txt



\## What You Can Learn



1\. Quantum Optics: SPDC, BBO crystals, phase matching

2\. Simulation: SNLO software for nonlinear optics

3\. Data Analysis: Quantum tomography, Bell inequalities

4\. Machine Learning: Random Forest, LSTM, SVM

5\. Signal Processing: Sparsity denoising



\## Limitations



\- Small dataset: 12 samples only

\- SNR improvement limited by sample size

\- Downloaded data (not own experiment)



\## Acknowledgments



\- ECMBI Dataset (Zenodo 10927445)

\- arXiv:2511.12546

\- SNLO Software

\- QuTiP



\## License



MIT License



\## Author



GitHub: @Athleity



Project Link: https://github.com/Athleity/SPDC\_Project

