"""
Project: Pokémon DB Scraper – Pokedex All and Data Analysis with 4 Machine Learning Algorithms
Added: Useful charts for data distribution analysis, clustering, and feature importance.

Steps:
1. Scrape data from Pokémon DB (Pokedex All)
2. Initial preprocessing of raw data
3. Modeling (K-Means, Linear Regression, Decision Tree, Random Forest) on raw data
4. Distribution chart of Total and correlation matrix (Raw Data)
5. Data cleaning (remove Outliers)
6. Re-modeling on cleaned data
7. Compare metrics (Silhouette Score, MAE, RMSE) before and after cleaning
8. Silhouette vs k chart (Raw/Cleaned)
9. Feature importance chart (Raw/Cleaned)
10. Actual vs Predicted chart for each model (Raw/Cleaned)
"""

# ----------------------------
# Import all libraries
# ----------------------------
import os
import requests                      # For sending HTTP requests
from bs4 import BeautifulSoup        # For parsing HTML
import pandas as pd                  # For working with DataFrame and CSV
import numpy as np                   # For numerical computations

from sklearn.cluster import KMeans   # K-Means clustering
from sklearn.linear_model import LinearRegression  # Linear Regression
from sklearn.tree import DecisionTreeRegressor      # Decision Tree
from sklearn.ensemble import RandomForestRegressor  # Random Forest

from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error  # Metrics: Silhouette, MAE, RMSE
from sklearn.model_selection import train_test_split   # Split data into Train/Test
from sklearn.preprocessing import StandardScaler        # For feature scaling

import matplotlib.pyplot as plt  # For plotting
import seaborn as sns            # For drawing correlation matrix


# --------------------------------------------
# Constants and main parameters
# --------------------------------------------
RAW_CSV = "pokemon_raw.csv"
CLEAN_CSV = "pokemon_clean.csv"
CHART_DIR = "charts"

# Create charts directory if it doesn't exist
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)


# --------------------------------------------
# 1. Scrape Pokémon table function
# --------------------------------------------
def scrape_pokemon_stats(output_csv=RAW_CSV):
    """
    This function scrapes the Pokedex All page, converts the table to a DataFrame, and saves it.
    The current structure of the Pokémon DB table is as described on Medium (10 columns per row).
    """
    url = "https://pokemondb.net/pokedex/all"
    print("⏳ Sending request to Pokémon DB to fetch Pokédex data ...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Find main table with id="pokedex"
    table = soup.find("table", {"id": "pokedex"})
    if table is None:
        raise Exception("⚠️ Pokedex table not found!")

    # Extract table rows (each <tr> has 10 <td>)
    rows = table.find("tbody").find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        # Ensure exactly 10 columns
        if len(cols) < 10:
            continue

        # Columns (index 0 to 9): #, Name, Type, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
        pokedex_no = cols[0].text.strip()                        # Pokédex number
        name = cols[1].text.strip()                              # Pokémon name
        types = [t.text.strip() for t in cols[2].find_all("a")]  # Type list
        total = int(cols[3].text.strip())                        # Total base stats
        hp = int(cols[4].text.strip())                           # HP
        attack = int(cols[5].text.strip())                       # Attack
        defense = int(cols[6].text.strip())                      # Defense
        sp_atk = int(cols[7].text.strip())                       # Special Attack
        sp_def = int(cols[8].text.strip())                       # Special Defense
        speed = int(cols[9].text.strip())                        # Speed

        data.append({
            "PokedexNo": pokedex_no,
            "Name": name,
            "Type1": types[0] if len(types) > 0 else None,
            "Type2": types[1] if len(types) > 1 else None,
            "Total": total,
            "HP": hp,
            "Attack": attack,
            "Defense": defense,
            "Sp. Atk": sp_atk,
            "Sp. Def": sp_def,
            "Speed": speed
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")     # Save raw CSV
    print(f"✅ Raw Pokémon data saved to '{output_csv}'. Number of records: {len(df)}")

    # Show first and last 5 rows for verification
    print("\nFirst 5 Pokémon samples:")
    print(df.head(5))
    print("\nLast 5 Pokémon samples:")
    print(df.tail(5))
    return df


# --------------------------------------------
# 2. Initial preprocessing of raw data
# --------------------------------------------
def initial_preprocessing(input_csv=RAW_CSV):
    """
    - Load raw CSV
    - Remove rows with missing numeric columns (NaN)
    - Define X and y (features and target)
    """
    print("\n⏳ Starting initial preprocessing of raw data ...")
    df = pd.read_csv(input_csv)  # Read raw CSV

    print("  • Columns:", df.columns.tolist())
    print("  • Table shape before dropna:", df.shape)
    print("  • First five rows:\n", df.head(), "\n")

    # Remove Pokémon with missing numeric columns
    df = df.dropna(subset=["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])
    print("  • Table shape after dropna:", df.shape, "\n")

    # Define X (features) and y (target)
    feature_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    X = df[feature_cols]
    y = df["Total"]

    print("Initial preprocessing completed.\n")
    return df, X, y


# --------------------------------------------
# 3. Distribution and correlation charts (Raw Data)
# --------------------------------------------
def plot_raw_data_insights(df):
    """
    Plot distribution of Total and correlation matrix of six numeric features,
    and save them in charts/
    """
    # 1. Distribution chart of Total
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Total"], bins=30, kde=True, color="skyblue")
    plt.title("Distribution of Total Base Stats (Raw Data)")
    plt.xlabel("Total Base Stats")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path_total_hist = os.path.join(CHART_DIR, "raw_total_distribution.png")
    plt.savefig(path_total_hist)
    plt.close()
    print(f"✅ Total distribution chart (Raw) saved to '{path_total_hist}'.")

    # 2. Correlation matrix of six numeric features
    corr = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Base Stats (Raw Data)")
    plt.tight_layout()
    path_corr = os.path.join(CHART_DIR, "raw_correlation_matrix.png")
    plt.savefig(path_corr)
    plt.close()
    print(f"✅ Correlation matrix (Raw) saved to '{path_corr}'.")


# --------------------------------------------
# 4. Modeling on raw data
# --------------------------------------------
def model_on_raw_data(df, X, y):
    """
    Runs four algorithms on raw data and also creates the following charts:
      1. K-Means → Calculate Silhouette Score and save Silhouette vs k (Raw)
      2. Linear Regression → Calculate MAE and RMSE and Actual vs Predicted chart
      3. Decision Tree Regressor → Calculate MAE and RMSE and Actual vs Predicted chart
      4. Random Forest Regressor → Calculate MAE and RMSE and Actual vs Predicted chart and Feature Importances (bar chart)
    """
    print("⏳ Running models on raw data ...")

    if X.shape[0] == 0:
        print("⚠️ Raw data has no records; modeling cannot continue.")
        return None

    results = {}

    # ---------- K-Means clustering and Silhouette vs k ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)

    # Save Silhouette vs k chart (Raw)
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), silhouette_scores, marker="o", linestyle="--", color="olive")
    plt.title("Silhouette Score vs k (Raw Data)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(K_range))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path_sil_raw = os.path.join(CHART_DIR, "raw_silhouette_vs_k.png")
    plt.savefig(path_sil_raw)
    plt.close()
    print(f"✅ Silhouette vs k (Raw) saved to '{path_sil_raw}'.")

    # For k=4 (default)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"  • Silhouette Score (Raw Data, k=4): {sil_score:.4f}")
    results["silhouette_raw"] = sil_score

    # ---------- Split into Train/Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- Linear Regression ----------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"  • Linear Regression (Raw): MAE = {mae_lr:.4f}, RMSE = {rmse_lr:.4f}")
    results["lr_mae_raw"] = mae_lr
    results["lr_rmse_raw"] = rmse_lr

    # Actual vs Predicted chart (Linear Regression, Raw)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.6, color="navy", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Actual vs Predicted (Linear Regression, Raw)")
    plt.tight_layout()
    path_lr_raw = os.path.join(CHART_DIR, "raw_actual_vs_predicted_lr.png")
    plt.savefig(path_lr_raw)
    plt.close()
    print(f"✅ Actual vs Predicted chart (LR, Raw) saved to '{path_lr_raw}'.")

    # ---------- Decision Tree Regressor ----------
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    print(f"  • Decision Tree (Raw): MAE = {mae_dt:.4f}, RMSE = {rmse_dt:.4f}")
    results["dt_mae_raw"] = mae_dt
    results["dt_rmse_raw"] = rmse_dt

    # Actual vs Predicted chart (Decision Tree, Raw)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_dt, alpha=0.6, color="darkorange", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Actual vs Predicted (Decision Tree, Raw)")
    plt.tight_layout()
    path_dt_raw = os.path.join(CHART_DIR, "raw_actual_vs_predicted_dt.png")
    plt.savefig(path_dt_raw)
    plt.close()
    print(f"✅ Actual vs Predicted chart (DT, Raw) saved to '{path_dt_raw}'.")

    # ---------- Random Forest Regressor ----------
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"  • Random Forest (Raw): MAE = {mae_rf:.4f}, RMSE = {rmse_rf:.4f}")
    results["rf_mae_raw"] = mae_rf
    results["rf_rmse_raw"] = rmse_rf

    # Actual vs Predicted chart (Random Forest, Raw)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.6, color="forestgreen", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Actual vs Predicted (Random Forest, Raw)")
    plt.tight_layout()
    path_rf_raw = os.path.join(CHART_DIR, "raw_actual_vs_predicted_rf.png")
    plt.savefig(path_rf_raw)
    plt.close()
    print(f"✅ Actual vs Predicted chart (RF, Raw) saved to '{path_rf_raw}'.")

    # Bar Chart of Feature Importances (Random Forest, Raw)
    importances = rf.feature_importances_
    feature_names = X.columns.tolist()
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names, vals = zip(*fi)

    plt.figure(figsize=(8, 5))
    plt.bar(names, vals, color="teal", edgecolor="black")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances (Random Forest, Raw)")
    plt.tight_layout()
    path_fi_raw = os.path.join(CHART_DIR, "raw_feature_importances.png")
    plt.savefig(path_fi_raw)
    plt.close()
    print(f"✅ Feature Importances chart (Raw) saved to '{path_fi_raw}'.")

    return results


# --------------------------------------------
# 5. Data cleaning
# --------------------------------------------
def clean_data(df, output_csv=CLEAN_CSV):
    """
    - Remove Pokémon with Total < 100 or Total > 700 (remove outliers)
    - Remove rows where any of the six stats is zero
    - Save cleaned CSV and return X_clean, y_clean
    """
    print("⏳ Starting data cleaning ...")
    df_clean = df.copy()

    # Remove outliers based on Total (less than 100 or more than 700)
    df_clean = df_clean[(df_clean["Total"] >= 100) & (df_clean["Total"] <= 700)]

    # Remove rows where any stat is zero
    df_clean = df_clean[(df_clean["HP"] > 0) &
                        (df_clean["Attack"] > 0) &
                        (df_clean["Defense"] > 0) &
                        (df_clean["Sp. Atk"] > 0) &
                        (df_clean["Sp. Def"] > 0) &
                        (df_clean["Speed"] > 0)]

    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")  # Save cleaned CSV
    print(f"  • Cleaned data ({len(df_clean)} records) saved to '{output_csv}'.\n")

    X_clean = df_clean[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    y_clean = df_clean["Total"]
    return df_clean, X_clean, y_clean


# --------------------------------------------
# 6. Modeling on cleaned data
# --------------------------------------------
def model_on_clean_data(df_clean, X_clean, y_clean):
    """
    Runs the same four algorithms on cleaned data and also creates the following charts:
      1. K-Means → Silhouette Score and Silhouette vs k (Cleaned)
      2. Linear Regression → MAE, RMSE and Actual vs Predicted chart
      3. Decision Tree Regressor → MAE, RMSE and Actual vs Predicted chart
      4. Random Forest Regressor → MAE, RMSE and Actual vs Predicted chart and Feature Importances (bar chart)
    """
    print("⏳ Running models on cleaned data ...")

    if X_clean.shape[0] == 0:
        print("⚠️ Cleaned data has no records; clustering cannot continue.")
        return None

    results = {}

    # ---------- K-Means clustering and Silhouette vs k (Cleaned) ----------
    scaler_c = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(X_clean)

    silhouette_scores_c = []
    K_range = range(2, 11)
    for k in K_range:
        km_c = KMeans(n_clusters=k, random_state=42)
        labels_c = km_c.fit_predict(Xc_scaled)
        sil_c = silhouette_score(Xc_scaled, labels_c)
        silhouette_scores_c.append(sil_c)

    # Save Silhouette vs k chart (Cleaned)
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), silhouette_scores_c, marker="o", linestyle="--", color="crimson")
    plt.title("Silhouette Score vs k (Cleaned Data)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(K_range))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path_sil_clean = os.path.join(CHART_DIR, "cleaned_silhouette_vs_k.png")
    plt.savefig(path_sil_clean)
    plt.close()
    print(f"✅ Silhouette vs k (Cleaned) saved to '{path_sil_clean}'.")

    # For k=4 (default)
    kmeans_c = KMeans(n_clusters=4, random_state=42)
    clusters_c = kmeans_c.fit_predict(Xc_scaled)
    sil_score_c = silhouette_score(Xc_scaled, clusters_c)
    print(f"  • Silhouette Score (Cleaned Data, k=4): {sil_score_c:.4f}")
    results["silhouette_clean"] = sil_score_c

    # ---------- Split into Train/Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    # ---------- Linear Regression ----------
    lr_c = LinearRegression()
    lr_c.fit(X_train, y_train)
    y_pred_lr_c = lr_c.predict(X_test)
    mae_lr_c = mean_absolute_error(y_test, y_pred_lr_c)
    rmse_lr_c = np.sqrt(mean_squared_error(y_test, y_pred_lr_c))
    print(f"  • Linear Regression (Cleaned): MAE = {mae_lr_c:.4f}, RMSE = {rmse_lr_c:.4f}")
    results["lr_mae_clean"] = mae_lr_c
    results["lr_rmse_clean"] = rmse_lr_c

    # Actual vs Predicted chart (Linear Regression, Cleaned)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_lr_c, alpha=0.6, color="dodgerblue", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Actual vs Predicted (Linear Regression, Cleaned)")
    plt.tight_layout()
    path_lr_clean = os.path.join(CHART_DIR, "cleaned_actual_vs_predicted_lr.png")
    plt.savefig(path_lr_clean)
    plt.close()
    print(f"✅ Actual vs Predicted chart (LR, Cleaned) saved to '{path_lr_clean}'.")

    # ---------- Decision Tree Regressor ----------
    dt_c = DecisionTreeRegressor(random_state=42)
    dt_c.fit(X_train, y_train)
    y_pred_dt_c = dt_c.predict(X_test)
    mae_dt_c = mean_absolute_error(y_test, y_pred_dt_c)
    rmse_dt_c = np.sqrt(mean_squared_error(y_test, y_pred_dt_c))
    print(f"  • Decision Tree (Cleaned): MAE = {mae_dt_c:.4f}, RMSE = {rmse_dt_c:.4f}")
    results["dt_mae_clean"] = mae_dt_c
    results["dt_rmse_clean"] = rmse_dt_c

    # Actual vs Predicted chart (Decision Tree, Cleaned)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_dt_c, alpha=0.6, color="darkorange", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Actual vs Predicted (Decision Tree, Cleaned)")
    plt.tight_layout()
    path_dt_clean = os.path.join(CHART_DIR, "cleaned_actual_vs_predicted_dt.png")
    plt.savefig(path_dt_clean)
    plt.close()
    print(f"✅ Actual vs Predicted chart (DT, Cleaned) saved to '{path_dt_clean}'.")

    # ---------- Random Forest Regressor ----------
    rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_c.fit(X_train, y_train)
    y_pred_rf_c = rf_c.predict(X_test)
    mae_rf_c = mean_absolute_error(y_test, y_pred_rf_c)
    rmse_rf_c = np.sqrt(mean_squared_error(y_test, y_pred_rf_c))
    print(f"  • Random Forest (Cleaned): MAE = {mae_rf_c:.4f}, RMSE = {rmse_rf_c:.4f}")
    results["rf_mae_clean"] = mae_rf_c
    results["rf_rmse_clean"] = rmse_rf_c

    # Actual vs Predicted chart (Random Forest, Cleaned)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_rf_c, alpha=0.6, color="forestgreen", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Total")
    plt.ylabel("Predicted Total")
    plt.title("Actual vs Predicted (Random Forest, Cleaned)")
    plt.tight_layout()
    path_rf_clean = os.path.join(CHART_DIR, "cleaned_actual_vs_predicted_rf.png")
    plt.savefig(path_rf_clean)
    plt.close()
    print(f"✅ Actual vs Predicted chart (RF, Cleaned) saved to '{path_rf_clean}'.")

    # Bar chart of Feature Importances (Random Forest, Cleaned)
    importances_c = rf_c.feature_importances_
    feature_names_c = X_clean.columns.tolist()
    fi_c = sorted(zip(feature_names_c, importances_c), key=lambda x: x[1], reverse=True)
    names_c, vals_c = zip(*fi_c)

    plt.figure(figsize=(8, 5))
    plt.bar(names_c, vals_c, color="crimson", edgecolor="black")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances (Random Forest, Cleaned)")
    plt.tight_layout()
    path_fi_clean = os.path.join(CHART_DIR, "cleaned_feature_importances.png")
    plt.savefig(path_fi_clean)
    plt.close()
    print(f"✅ Feature Importances chart (Cleaned) saved to '{path_fi_clean}'.")

    return results


# --------------------------------------------
# 7. Main function to run all steps
# --------------------------------------------
def main():
    # 1. Scrape data and save raw CSV
    df_raw = scrape_pokemon_stats(output_csv=RAW_CSV)

    # 2. Initial preprocessing of raw data
    df_raw, X_raw, y_raw = initial_preprocessing(input_csv=RAW_CSV)

    # 3. Plot distribution and correlation charts (Raw Data)
    plot_raw_data_insights(df_raw)

    # 4. Modeling on raw data and related charts
    results_raw = model_on_raw_data(df_raw, X_raw, y_raw)

    # 5. Data cleaning
    df_clean, X_clean, y_clean = clean_data(df_raw, output_csv=CLEAN_CSV)

    # 6. Modeling on cleaned data and related charts
    results_clean = model_on_clean_data(df_clean, X_clean, y_clean)

    # 7. Show and compare results
    print("\n--- Comparison of results before and after data cleaning ---")
    if results_raw:
        print("Initial results (Raw Data):")
        for k, v in results_raw.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Initial results (Raw Data): NA")

    if results_clean:
        print("\nResults after cleaning (Cleaned Data):")
        for k, v in results_clean.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nResults after cleaning (Cleaned Data): NA")

    print("\n✅ Project execution finished.")

if __name__ == "__main__":
    main()
