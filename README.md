# SPDC with BBO Crystal  

Quantum Entanglement Detection and Machine Learning  

Complete pipeline for Spontaneous Parametric Down-Conversion (SPDC) analysis using a BBO crystal, featuring quantum state tomography, Bell inequality violation, sparsity-driven entanglement detection, and machine learning prediction.  

Paper implementation: arXiv:2511.12546 - Sparsity-driven entanglement detection  

---

## Key Results  

| Metric | Value | Significance |  
|--------|-------|--------------|  
| EPR Parameter | 0.1712 | ENTANGLED |  
| Bell Parameter S | 2.6872 | VIOLATION |  
| State Fidelity | 0.9622 | 96 percent match |  
| Concurrence | 0.9274 | Strong entanglement |  
| Random Forest R2 | 0.9294 | Power prediction |  
| LSTM R2 | 0.8269 | Time series forecast |  

---

## Project Structure  

SPDC_Project/  
|  
|-- 01_Thesis_Figures/  
|   |-- 17 publication-ready figures  
|  
|-- 02_Python_Scripts/  
|   |-- Main/                 (quantum tomography, Bell inequality)  
|   |-- Analysis/             (data exploration scripts)  
|   |-- Pipeline/             (end-to-end pipelines)  
|   |-- Visualization/        (plotting utilities)  
|  
|-- 03_Data/  
|   |-- ECMBI_Tomography/  
|       |-- FINAL_RESULTS/    (EPR=0.1712, SNR results)  
|       |-- 12 raw CSV files  
|  
|-- 05_Results/  
|   |-- CSV_Data/             (15 files)  
|   |-- Graphs/               (12 files)  
|   |-- PDF_Reports/          (7 files)  
|  
|-- 06a_SNLO_Graphs/  
    |-- SNLO simulation outputs  

---

## Quick Start  

Clone the repository:  
git clone https://github.com/Athleity/SPDC_Project.git  
cd SPDC_Project  

Install dependencies:  
pip install -r requirements.txt  

Run the pipeline:  
cd 02_Python_Scripts  
python spdc_pipeline.py  

View results:  
start ..\03_Data\ECMBI_Tomography\FINAL_RESULTS\  

---

## Dependencies  

- numpy >= 1.21.0  
- pandas >= 1.3.0  
- matplotlib >= 3.4.0  
- scikit-learn >= 1.0.0  
- tensorflow >= 2.10.0  
- qutip >= 4.7.0  
- scipy >= 1.7.0  
- seaborn >= 0.11.0  

---

## Results Location  

All final results are in: 03_Data/ECMBI_Tomography/FINAL_RESULTS/  

Files:  
- epr_results.csv (EPR = 0.1712)  
- snr_results.csv (SNR improvement +0.0083 dB)  
- heatmap_comparison.png  
- comparison_plot.png  
- summary_report.txt  

---

## Paper Implementation  

This project implements sparsity-driven entanglement detection from arXiv:2511.12546.  

- Equation 1: Sample covariance matrix  
- Equation 2: L1-regularization with LassoCV  
- Equation 3: SNR calculation  
- EPR Criterion: Less than 1 equals entangled  

---

## Machine Learning Results  

| Model | Task | Result |  
|-------|------|--------|  
| Random Forest | Predict pump power | R2 = 0.9294 |  
| LSTM | Forecast photon counts | R2 = 0.8269 |  
| SVM | Classify signal vs noise | Accuracy = 99.9% |  

---

## SNLO Simulations  

| Parameter | Value |  
|-----------|-------|  
| BBO angle for SHG | 33.2 degrees |  
| Compensator setting | 77.5 degrees |  
| Signal wavelength | 583 nm |  
| Idler wavelength | 900 nm |  
| Pump wavelength | 354 nm |  

---

## Limitations  

- Small dataset: 12 samples only  
- SNR improvement limited by sample size  
- Downloaded data (not own experiment)  

---

## Acknowledgments  

- ECMBI Dataset (Zenodo 10927445)  
- arXiv:2511.12546  
- SNLO Software  
- QuTiP  

---

## License  

MIT License  

## Author  

GitHub: @Athleity  

Project Link: https://github.com/Athleity/SPDC_Project  
