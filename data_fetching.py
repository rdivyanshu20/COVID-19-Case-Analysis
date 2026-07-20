import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Phase 1: Data Fetching (API)
# Requirement Met: "Fetches COVID data via APIs"

print("Fetching real-time historical data from disease.sh API...")
countries = ["USA", "India", "Italy"]
url = f"https://disease.sh/v3/covid-19/historical/{','.join(countries)}?lastdays=all"
response = requests.get(url)

if response.status_code != 200:
    print("Failed to retrieve data.")
    exit()

data = response.json()
print("Data fetched successfully!\n")


# Phase 2: Preprocessing, Mortality, and Recovery Trends
# Requirement Met: "Analyzes case growth, mortality rates, and recovery trends"

country_dfs = {}

for country_data in data:
    name = country_data['country']
    timeline = country_data['timeline']
    
    # Extract all three metrics required by the project
    df = pd.DataFrame({
        'cases': pd.Series(timeline['cases']),
        'deaths': pd.Series(timeline['deaths']),
        'recovered': pd.Series(timeline['recovered'])
    })
    
    # Time-series analysis formatting
    df.index = pd.to_datetime(df.index)
    
    # 1. Case Growth: Calculate daily new cases (preventing negative data anomalies)
    df['daily_cases'] = df['cases'].diff().fillna(0).clip(lower=0)
    df['rolling_daily'] = df['daily_cases'].rolling(window=7).mean()
    
    # 2. Mortality & Recovery Rates: (Metric / Total Cases) * 100
    # .replace(0, np.nan) prevents division by zero errors on the first few days
    df['mortality_rate'] = (df['deaths'] / df['cases'].replace(0, np.nan)) * 100
    df['recovery_rate'] = (df['recovered'] / df['cases'].replace(0, np.nan)) * 100
    
    country_dfs[name] = df

# Phase 3: Forecasting Basics
# Requirement Met: "Trend plotting and forecasting basics"

# We will demonstrate forecasting on India's cumulative cases 
target_country = "India"
df_forecast = country_dfs[target_country].dropna().copy()

# Create a numeric time index for the Scikit-Learn model
df_forecast['time_idx'] = np.arange(len(df_forecast))

# Train the model on the last 100 days of data to capture the most recent trend
recent_data = df_forecast.tail(100)
X_train = recent_data[['time_idx']]
y_train = recent_data['cases']

model = LinearRegression()
model.fit(X_train, y_train)

# Generate future dates to forecast 30 days into the future
last_date = df_forecast.index[-1]
last_idx = df_forecast['time_idx'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
future_idx = pd.DataFrame({'time_idx': np.arange(last_idx + 1, last_idx + 31)})

# Run the forecast
future_predictions = model.predict(future_idx)


# Phase 4: Data Visualization (Dashboard)

# Create a dashboard with 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# --- Plot 1: Case Growth Across Countries ---
for name, df in country_dfs.items():
    axes[0].plot(df.index, df['rolling_daily'], label=name, linewidth=2)
axes[0].set_title('1. Case Growth Across Countries (7-Day Moving Avg)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Daily New Cases')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Mortality & Recovery Rates ---
axes[1].plot(df_forecast.index, df_forecast['mortality_rate'], color='red', label='Mortality Rate (%)', linewidth=2)
axes[1].plot(df_forecast.index, df_forecast['recovery_rate'], color='green', label='Recovery Rate (%)', linewidth=2)
axes[1].set_title(f'2. Mortality and Recovery Trends ({target_country})', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Rate (%)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- Plot 3: Forecasting Basics ---
axes[2].plot(recent_data.index, recent_data['cases'], label='Actual Cumulative Cases', color='blue', linewidth=2)
axes[2].plot(future_dates, future_predictions, label='30-Day Forecast', color='orange', linestyle='--', linewidth=2)
axes[2].set_title(f'3. Forecasting Basics: 30-Day Trajectory ({target_country})', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Total Cumulative Cases')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

# THE FIX: Adjusting the layout to prevent overlapping

# 1. Rotate the x-axis dates slightly on all subplots so they don't take up as much vertical space
for ax in axes:
    ax.tick_params(axis='x', rotation=45)

# 2. Apply tight_layout first, then explicitly force the vertical space (hspace) between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.5) 

# Display the layout
plt.show()
