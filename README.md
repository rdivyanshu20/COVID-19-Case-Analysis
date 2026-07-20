#  COVID-19 Global Case Analysis & Forecasting

##  Project Overview
This project is a comprehensive Data Science pipeline that fetches, processes, and visualizes real-world COVID-19 data. It answers a critical question: **What were the key patterns and timelines in the spread of COVID-19 globally?**

By connecting to live public health APIs, this script analyzes case growth, mortality rates, and recovery trends across multiple countries. It also includes a machine learning component to forecast future case trajectories based on recent data.

##  Key Features
- **Real-Time Data Fetching:** Utilizes the [disease.sh](https://disease.sh/) API to dynamically pull historical time-series data, bypassing the need for static CSV files.
- **Time-Series Analysis:** Computes 7-day rolling averages to smooth out reporting anomalies and weekend lags.
- **Metric Derivation:** Calculates daily new cases, Case Fatality Rates (CFR), and recovery percentages.
- **Forecasting Basics:** Implements a Scikit-Learn `LinearRegression` model to project cumulative cases 30 days into the future.
- **Multi-Country Dashboard:** Generates a stacked, professional Matplotlib visualization to compare different regional timelines (e.g., India, USA, Italy) simultaneously.

##  Concepts & Skills Demonstrated
- **Languages/Libraries:** Python, Pandas, NumPy, Matplotlib, Scikit-Learn, Requests
- **Data Engineering:** API integration, JSON parsing, handling missing or anomalous data (`.clip()`, `.fillna()`).
- **Analytics:** Public health data analytics, continuous time-series formatting.
- **Machine Learning:** Trend plotting and foundational forecasting models.

## How to Run
1. Ensure you have Python installed on your system.
2. Install the required dependencies using pip:
   ```bash
   pip install requests pandas numpy matplotlib scikit-learn
   ```
3. Execute the Python script:
   ```bash
   python covid_analysis.py
   ```
4. A 3-tier dashboard will render automatically, displaying the comparative analysis and forecasts.

## Dashboard Visualizations
1. **Case Growth Across Countries:** A comparative timeline of daily new cases.
2. **Mortality & Recovery Trends:** A detailed look at the evolution of survival and fatality rates over time for a target region.
3. **30-Day Trajectory Forecast:** A machine-learning-driven projection of cumulative cases based on recent trend data.

---
*Developed as a Data Science Internship Project showcasing end-to-end data processing and analytics.*
