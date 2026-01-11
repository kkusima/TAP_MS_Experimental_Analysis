# TAP Reactor Experimental Analysis Tool

<a href="https://tap-exp-analysis-app.streamlit.app" target="_blank" rel="noopener noreferrer">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" width="250" height="150"/>
</a>

## Overview
This Python application uses Streamlit to process and analyze experimental data from Temporal Analysis of Products (TAP) reactors. It reads raw National Instruments `.tdms` files, converts mass spectrometry signals into molar flow rates, and performs moment-based kinetic analysis.

The tool replaces manual Excel-based workflows by automating data extraction, peak integration, and Knudsen diffusion validation.

## Features
- **Direct TDMS Parsing:** Reads binary `.tdms` files directly without requiring intermediate conversion to Excel.
- **Automated Pulse Processing:** Identifies and separates multiple pulses within a dataset.
- **Kinetic Analysis:** Calculates Zeroth Moment ($M_0$), Peak Time ($t_p$), Peak Height ($H_p$), and the Knudsen diffusion criterion.
- **Multi-Peak Deconvolution:** Supports defining multiple peaks per pulse with user-specified delay times.
- **Interactive Visualization:** Generates zoomable plots for pulse shapes using Plotly.
- **Data Export:** Provides downloads for summary statistics (CSV) and full processed time-series data (CSV and Excel).

## Dependencies
The application requires Python 3.8+ and the following libraries:

- `streamlit`
- `nptdms`
- `pandas`
- `numpy`
- `plotly`
- `openpyxl`

To install dependencies:
```bash
pip install -r requirements.txt
```

To run the app:
```bash
streamlit run TAP_Exp_Analysis.py
```
