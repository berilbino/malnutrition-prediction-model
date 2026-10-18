@echo off
title UNICEF Malnutrition & Survival Prediction Dashboard
echo ======================================================================
echo Launching Child Malnutrition, Survival & Disease Prediction System...
echo ======================================================================
cd /d "%~dp0"
python -m streamlit run app\streamlit_app.py
pause
