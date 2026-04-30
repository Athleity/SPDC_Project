@echo off
echo ========================================
echo ORGANIZING SPDC PROJECT FILES
echo ========================================

REM Create main folders if not exist
mkdir 01_Thesis_Figures 2>nul
mkdir 02_Python_Scripts 2>nul
mkdir 03_Data 2>nul
mkdir 04_Lab_References 2>nul
mkdir 05_Results 2>nul
mkdir 06_ML_Results 2>nul
mkdir 07_Simulation_Results 2>nul
mkdir 08_SNLO_Graphs 2>nul
mkdir 09_Final_Submission 2>nul

REM Create subfolders inside Results
mkdir 05_Results\Graphs 2>nul
mkdir 05_Results\CSV_Data 2>nul
mkdir 05_Results\PDF_Reports 2>nul

REM Create subfolders inside Python Scripts
mkdir 02_Python_Scripts\Main 2>nul
mkdir 02_Python_Scripts\Visualization 2>nul
mkdir 02_Python_Scripts\Analysis 2>nul
mkdir 02_Python_Scripts\Pipeline 2>nul
mkdir 02_Python_Scripts\ML_Models 2>nul
mkdir 02_Python_Scripts\Paper_Implementation 2>nul

echo.
echo Folders created successfully!
echo.
pause