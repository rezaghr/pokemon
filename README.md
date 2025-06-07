# Project Documentation: Pokémon Database Scraper & Data Analysis with 4 Machine Learning Algorithms

## Project Overview
This project scrapes Pokémon table data from [Pokémon DB](https://pokemondb.net/pokedex/all) and analyzes it using four machine learning algorithms: **K-Means Clustering**, **Linear Regression**, **Decision Tree Regression**, and **Random Forest Regression**.  
It also generates a variety of charts for data distribution, clustering, and feature importance, providing a comprehensive overview of the Pokémon dataset and the impact of data cleaning on model performance.

---

## Project Workflow

1. **Data Collection:**  
   - Scrape the complete Pokémon table from the website.
   - Save the raw data as `pokemon_raw.csv`.

2. **Initial Preprocessing:**  
   - Load the raw CSV.
   - Remove rows with missing or invalid numeric values.
   - Define features (`X`: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed) and target (`y`: Total).

3. **Exploratory Data Analysis (EDA):**  
   - Plot the distribution of total base stats.
   - Generate a correlation matrix for numeric features.

4. **Modeling on Raw Data:**  
   - **K-Means Clustering:**  
     - Calculate Silhouette scores for k=2 to k=10.
     - Plot Silhouette Score vs. Number of Clusters.
     - Assign clusters and save cluster assignments.
   - **Linear Regression, Decision Tree, Random Forest:**  
     - Train/test split.
     - Calculate MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).
     - Plot Actual vs. Predicted values.
     - For Random Forest, plot feature importances.

5. **Data Cleaning:**  
   - Remove Pokémon with total stats < 100 or > 700 (outliers).
   - Remove rows where any stat is zero.
   - Save the cleaned data as `pokemon_clean.csv`.

6. **Modeling on Cleaned Data:**  
   - Repeat all modeling and charting steps on the cleaned dataset.

7. **Results Comparison:**  
   - Compare model metrics (Silhouette, MAE, RMSE) before and after data cleaning.
   - Summarize the impact of cleaning on clustering and regression performance.

---

## Main Functions Overview

- **scrape_pokemon_stats:**  
  Scrapes Pokémon data from the website and saves it as a CSV file.

- **initial_preprocessing:**  
  Loads the raw data, removes incomplete rows, and separates features and target.

- **plot_raw_data_insights:**  
  Plots the distribution of total stats and the correlation matrix.

- **model_on_raw_data / model_on_clean_data:**  
  Runs all machine learning models on raw and cleaned data, saving related charts and metrics.

- **clean_data:**  
  Cleans the data by removing outliers and invalid rows, then saves a new CSV.

- **show_cluster_examples:**  
  Displays and saves sample Pokémon from each cluster, including cluster statistics.

- **main:**  
  The main function that orchestrates all steps in sequence.

---

## Output Files and Directories

- **Raw data:** `pokemon_raw.csv`
- **Cleaned data:** `pokemon_clean.csv`
- **Charts and outputs:** All visualizations and cluster assignments are saved in the `charts/` directory.

---

## Example Output Charts

- Distribution of Total Base Stats
- Correlation Matrix of Stats
- Silhouette Score vs. Number of Clusters (Raw & Cleaned)
- Actual vs. Predicted (for each model, Raw & Cleaned)
- Feature Importances (Random Forest, Raw & Cleaned)
- Cluster Assignments (CSV)

---

## How to Run the Project

1. Make sure you have Python 3 and the required libraries installed:
   ```
   pip install requests beautifulsoup4 pandas numpy scikit-learn matplotlib seaborn
   ```
2. Run the main script:
   ```
   python3 pokemon.py
   ```
3. All steps will be performed automatically.  
   Outputs and charts will be available in the terminal and the `charts` directory.

---

## Requirements

- Python 3.x
- Libraries: `requests`, `beautifulsoup4`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Credits

**Prepared by:**  
Pokémon Project Development Team  
2025

---