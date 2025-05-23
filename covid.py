import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# 1. Data Loading
df = pd.read_csv('owid-covid-data.csv')

# 2. Data Exploration
print("Columns:", df.columns)
print(df.head())
print("Missing values:\n", df.isnull().sum())

# 3. Data Cleaning
countries = ['Kenya', 'United States', 'India']
df = df[df['location'].isin(countries)]
df = df.dropna(subset=['date', 'total_cases', 'total_deaths'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['location', 'date'])
num_cols = ['total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'total_vaccinations']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].interpolate().fillna(0)

# 4. Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
for country in countries:
    plt.plot(df[df['location'] == country]['date'],
             df[df['location'] == country]['total_cases'],
             label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for country in countries:
    plt.plot(df[df['location'] == country]['date'],
             df[df['location'] == country]['total_deaths'],
             label=country)
plt.title('Total COVID-19 Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for country in countries:
    plt.plot(df[df['location'] == country]['date'],
             df[df['location'] == country]['new_cases'].rolling(7).mean(),
             label=country)
plt.title('Daily New COVID-19 Cases (7-day avg)')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.tight_layout()
plt.show()

df['death_rate'] = df['total_deaths'] / df['total_cases']
plt.figure(figsize=(12, 6))
for country in countries:
    plt.plot(df[df['location'] == country]['date'],
             df[df['location'] == country]['death_rate'],
             label=country)
plt.title('COVID-19 Death Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Death Rate')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Vaccination Progress
if 'total_vaccinations' in df.columns:
    plt.figure(figsize=(12, 6))
    for country in countries:
        plt.plot(df[df['location'] == country]['date'],
                 df[df['location'] == country]['total_vaccinations'],
                 label=country)
    plt.title('Cumulative Vaccinations Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Vaccinations')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # % vaccinated (if population column exists)
    if 'population' in df.columns and 'people_vaccinated' in df.columns:
        latest = df.sort_values('date').groupby('location').tail(1)
        latest['pct_vaccinated'] = latest['people_vaccinated'] / latest['population'] * 100
        sns.barplot(x='location', y='pct_vaccinated', data=latest)
        plt.title('% Population Vaccinated (Latest)')
        plt.ylabel('% Vaccinated')
        plt.show()







