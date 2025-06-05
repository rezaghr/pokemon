"""
Project: Pokémon DB Scraper – Pokedex All and Data Analysis with 4 Machine Learning Algorithms
Steps:
1. Scrape data from Pokémon DB (Pokedex All)
2. Initial preprocessing of raw data
3. Modeling (K-Means, Linear Regression, Decision Tree, Random Forest) on raw data
4. Data cleaning (remove outliers)
5. Re-modeling on cleaned data
6. Compare metrics (Silhouette Score, MAE, RMSE) before and after cleaning
"""

# ----------------------------
# Import all libraries
# ----------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# --------------------------------------------
# Constants and main parameters
# --------------------------------------------
RAW_CSV = "pokemon_raw.csv"
CLEAN_CSV = "pokemon_clean.csv"


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
        if len(cols) < 10:
            continue

        # Columns (index 0 to 9): #, Name, Type, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
        pokedex_no = cols[0].text.strip()
        name = cols[1].text.strip()
        types = [t.text.strip() for t in cols[2].find_all("a")]
        total = int(cols[3].text.strip())
        hp = int(cols[4].text.strip())
        attack = int(cols[5].text.strip())
        defense = int(cols[6].text.strip())
        sp_atk = int(cols[7].text.strip())
        sp_def = int(cols[8].text.strip())
        speed = int(cols[9].text.strip())

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
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
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
    - Set X and y (features and target)
    """
    print("\n⏳ Starting initial preprocessing of raw data ...")
    df = pd.read_csv(input_csv)

    print("  • Columns:", df.columns.tolist())
    print("  • Table shape before dropna:", df.shape)
    print("  • First five rows:\n", df.head(), "\n")

    # Remove Pokémon with missing numeric columns
    df = df.dropna(subset=["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])
    print("  • Table shape after dropna:", df.shape, "\n")

    # Set X (features) and y (target)
    feature_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    X = df[feature_cols]
    y = df["Total"]

    print("Initial preprocessing completed.\n")
    return df, X, y


# --------------------------------------------
# 3. Modeling on raw data
# --------------------------------------------
def model_on_raw_data(df, X, y):
    """
    Runs four algorithms on raw data:
      1. K-Means → Silhouette Score
      2. Linear Regression → MAE and RMSE
      3. Decision Tree Regressor → MAE and RMSE
      4. Random Forest Regressor → MAE, RMSE, and Feature Importances
    """
    print("⏳ Running models on raw data ...")

    if X.shape[0] == 0:
        print("⚠️ Raw data has no records; modeling cannot continue.")
        return None

    # 1. K-Means clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"  • Silhouette Score (Raw Data): {sil_score:.4f}")

    # 2. Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"  • Linear Regression (Raw): MAE = {mae_lr:.4f}, RMSE = {rmse_lr:.4f}")

    # 4. Decision Tree Regressor
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    print(f"  • Decision Tree (Raw): MAE = {mae_dt:.4f}, RMSE = {rmse_dt:.4f}")

    # 5. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"  • Random Forest (Raw): MAE = {mae_rf:.4f}, RMSE = {rmse_rf:.4f}")

    # Show feature importances
    importances = rf.feature_importances_
    feature_names = X.columns.tolist()
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("  • Feature Importances (Raw Data):")
    for name, imp in fi:
        print(f"      - {name}: {imp:.4f}")
    print()

    return {
        "silhouette_raw": sil_score,
        "lr_mae_raw": mae_lr,
        "lr_rmse_raw": rmse_lr,
        "dt_mae_raw": mae_dt,
        "dt_rmse_raw": rmse_dt,
        "rf_mae_raw": mae_rf,
        "rf_rmse_raw": rmse_rf
    }


# --------------------------------------------
# 4. Data cleaning
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

    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"  • Cleaned data ({len(df_clean)} records) saved to '{output_csv}'.\n")

    X_clean = df_clean[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    y_clean = df_clean["Total"]
    return df_clean, X_clean, y_clean


# --------------------------------------------
# 5. Modeling on cleaned data
# --------------------------------------------
def model_on_clean_data(df_clean, X_clean, y_clean):
    """
    Runs the same four algorithms on cleaned data:
      1. K-Means → Silhouette Score
      2. Linear Regression → MAE, RMSE
      3. Decision Tree Regressor → MAE, RMSE
      4. Random Forest Regressor → MAE, RMSE + Feature Importances
    """
    print("⏳ Running models on cleaned data ...")

    if X_clean.shape[0] == 0:
        print("⚠️ Cleaned data has no records; clustering cannot continue.")
        return None

    # K-Means clustering
    scaler_c = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(X_clean)
    kmeans_c = KMeans(n_clusters=4, random_state=42)
    clusters_c = kmeans_c.fit_predict(Xc_scaled)
    sil_score_c = silhouette_score(Xc_scaled, clusters_c)
    print(f"  • Silhouette Score (Cleaned Data): {sil_score_c:.4f}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr_c = LinearRegression()
    lr_c.fit(X_train, y_train)
    y_pred_lr_c = lr_c.predict(X_test)
    mae_lr_c = mean_absolute_error(y_test, y_pred_lr_c)
    rmse_lr_c = np.sqrt(mean_squared_error(y_test, y_pred_lr_c))
    print(f"  • Linear Regression (Cleaned): MAE = {mae_lr_c:.4f}, RMSE = {rmse_lr_c:.4f}")

    # Decision Tree Regressor
    dt_c = DecisionTreeRegressor(random_state=42)
    dt_c.fit(X_train, y_train)
    y_pred_dt_c = dt_c.predict(X_test)
    mae_dt_c = mean_absolute_error(y_test, y_pred_dt_c)
    rmse_dt_c = np.sqrt(mean_squared_error(y_test, y_pred_dt_c))
    print(f"  • Decision Tree (Cleaned): MAE = {mae_dt_c:.4f}, RMSE = {rmse_dt_c:.4f}")

    # Random Forest Regressor
    rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_c.fit(X_train, y_train)
    y_pred_rf_c = rf_c.predict(X_test)
    mae_rf_c = mean_absolute_error(y_test, y_pred_rf_c)
    rmse_rf_c = np.sqrt(mean_squared_error(y_test, y_pred_rf_c))
    print(f"  • Random Forest (Cleaned): MAE = {mae_rf_c:.4f}, RMSE = {rmse_rf_c:.4f}")

    # Feature importances
    importances_c = rf_c.feature_importances_
    feature_names_c = X_clean.columns.tolist()
    fi_c = sorted(zip(feature_names_c, importances_c), key=lambda x: x[1], reverse=True)
    print("  • Feature Importances (Cleaned Data):")
    for name, imp in fi_c:
        print(f"      - {name}: {imp:.4f}")
    print()

    return {
        "silhouette_clean": sil_score_c,
        "lr_mae_clean": mae_lr_c,
        "lr_rmse_clean": rmse_lr_c,
        "dt_mae_clean": mae_dt_c,
        "dt_rmse_clean": rmse_dt_c,
        "rf_mae_clean": mae_rf_c,
        "rf_rmse_clean": rmse_rf_c
    }


# --------------------------------------------
# 6. Main function to run all steps
# --------------------------------------------
def main():
    # 1. Scrape data and save raw CSV
    df_raw = scrape_pokemon_stats(output_csv=RAW_CSV)

    # 2. Initial preprocessing of raw data
    df_raw, X_raw, y_raw = initial_preprocessing(input_csv=RAW_CSV)

    # 3. Modeling on raw data
    results_raw = model_on_raw_data(df_raw, X_raw, y_raw)

    # 4. Data cleaning
    df_clean, X_clean, y_clean = clean_data(df_raw, output_csv=CLEAN_CSV)

    # 5. Modeling on cleaned data
    results_clean = model_on_clean_data(df_clean, X_clean, y_clean)

    # 6. Show and compare results
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
