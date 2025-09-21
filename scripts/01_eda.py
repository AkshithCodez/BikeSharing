# scripts/01_eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda():
    # Load the dataset
    df = pd.read_csv('data/raw/train.csv')

    # Create directory for reports
    os.makedirs('reports/figures', exist_ok=True)

    # Print basic info
    print("DATA INFO:")
    df.info()
    print("\nDATA DESCRIPTION:")
    print(df.describe())

    # Plot distribution of the target variable 'count'
    plt.figure(figsize=(10, 6))
    sns.histplot(df['count'], bins=50, kde=True)
    plt.title('Distribution of Bike Rentals (count)')
    plt.savefig('reports/figures/count_distribution.png')
    plt.close()
    print("\nSaved count distribution plot.")

    # Plot count vs. season
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='season', y='count')
    plt.title('Bike Rentals by Season')
    plt.savefig('reports/figures/rentals_by_season.png')
    plt.close()
    print("Saved rentals by season plot.")
    
    # Plot count vs. hour
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='hour', y='count') # We will need to engineer 'hour' from 'datetime'
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    sns.boxplot(data=df, x='hour', y='count')
    plt.title('Bike Rentals by Hour')
    plt.savefig('reports/figures/rentals_by_hour.png')
    plt.close()
    print("Saved rentals by hour plot.")

    # Correlation heatmap
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    print("Saved correlation heatmap.")


if __name__ == "__main__":
    run_eda()