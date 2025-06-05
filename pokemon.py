"""
پروژه: اسکرپ Pokémon DB – Pokedex All و تحلیل داده‌ها با ۴ الگوریتم یادگیری ماشین
افزوده‌شده: نمودارهای کاربردی برای تحلیل توزیع داده، خوشه‌بندی و اهمیت ویژگی‌ها.

مراحل:
1. جمع‌آوری داده‌ها از Pokémon DB (Pokedex All)
2. پیش‌پردازش اولیهٔ دادهٔ خام
3. مدل‌سازی (K-Means, Linear Regression, Decision Tree, Random Forest) روی دادهٔ خام
4. نمودار توزیع Total و ماتریس همبستگی (Raw Data)
5. پاک‌سازی داده‌ها (حذف Outliers)
6. مدل‌سازی مجدد روی دادهٔ پاک‌شده
7. مقایسهٔ معیارها (Silhouette Score, MAE, RMSE) قبل و بعد از پاک‌سازی
8. نمودار Silhouette vs k (Raw/Cleaned)
9. نمودار اهمیت ویژگی‌ها (Raw/Cleaned)
10. نمودار Actual vs Predicted برای هر مدل (Raw/Cleaned)
"""

# ----------------------------
# وارد کردن تمام کتابخانه‌ها
# ----------------------------
import os
import requests                      # برای ارسال درخواست HTTP 
from bs4 import BeautifulSoup        # برای پارس HTML 
import pandas as pd                  # برای کار با DataFrame و CSV 
import numpy as np                   # برای محاسبات عددی 

from sklearn.cluster import KMeans   # خوشه‌بندی K-Means 
from sklearn.linear_model import LinearRegression  # رگرسیون خطی 
from sklearn.tree import DecisionTreeRegressor      # درخت تصمیم 
from sklearn.ensemble import RandomForestRegressor  # جنگل تصادفی 

from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error  # معیارها: Silhouette, MAE, RMSE 
from sklearn.model_selection import train_test_split   # تقسیم داده به Train/Test 
from sklearn.preprocessing import StandardScaler        # برای مقیاس‌بندی ویژگی‌ها 

import matplotlib.pyplot as plt  # برای نمایش نمودار 
import seaborn as sns            # برای رسم ماتریس همبستگی 


# --------------------------------------------
# ثابت‌ها و پارامترهای اصلی
# --------------------------------------------
RAW_CSV = "pokemon_raw.csv"
CLEAN_CSV = "pokemon_clean.csv"
CHART_DIR = "charts"

# ایجاد پوشهٔ charts در صورت نبودن
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)


# --------------------------------------------
# 1. تابع اسکرپ جدول Pokémon
# --------------------------------------------
def scrape_pokemon_stats(output_csv=RAW_CSV):
    """
    این تابع صفحهٔ Pokedex All را اسکرپ می‌کند و جدول را به DataFrame تبدیل و در خروجی ذخیره می‌کند.
    ساختار فعلی جدول Pokémon DB شبیه به Medium توضیح داده شده است (۱۰ ستون در هر سطر) .
    """
    url = "https://pokemondb.net/pokedex/all"
    print("⏳ ارسال درخواست به Pokémon DB برای دریافت داده‌های Pokédex ...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"خطا در دریافت داده‌ها: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # یافتن جدول اصلی با id="pokedex"
    table = soup.find("table", {"id": "pokedex"})
    if table is None:
        raise Exception("⚠️ جدول Pokedex یافت نشد!")

    # استخراج سطرهای جدول (هر <tr> شامل 10 <td> است) 
    rows = table.find("tbody").find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        # بررسی اینکه دقیقاً 10 ستون داریم
        if len(cols) < 10:
            continue

        # ستون‌ها (اندیس 0 تا 9): #, Name, Type, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
        pokedex_no = cols[0].text.strip()                        # شماره Pokédex 
        name = cols[1].text.strip()                               # نام Pokémon
        types = [t.text.strip() for t in cols[2].find_all("a")]  # لیست نوع 
        total = int(cols[3].text.strip())                         # مجموع امتیازات عددی
        hp = int(cols[4].text.strip())                            # HP
        attack = int(cols[5].text.strip())                        # Attack
        defense = int(cols[6].text.strip())                       # Defense
        sp_atk = int(cols[7].text.strip())                         # Special Attack
        sp_def = int(cols[8].text.strip())                         # Special Defense
        speed = int(cols[9].text.strip())                          # Speed

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
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")     # ذخیرهٔ CSV خام 
    print(f"✅ دادهٔ خام Pokémon در فایل '{output_csv}' ذخیره شد. تعداد رکوردها: {len(df)}")

    # نمایش ۵ سطر اول و آخر برای اطمینان
    print("\nنمونهٔ ۵ Pokémon اول:")
    print(df.head(5))
    print("\nنمونهٔ ۵ Pokémon آخر:")
    print(df.tail(5))
    return df


# --------------------------------------------
# 2. پیش‌پردازش اولیهٔ دادهٔ خام
# --------------------------------------------
def initial_preprocessing(input_csv=RAW_CSV):
    """
    - بارگذاری CSV خام
    - حذف ردیف‌هایی که یکی از ستون‌های عددی‌شان ناقص (NaN) است
    - تعیین X و y (ویژگی‌ها و هدف)
    """
    print("\n⏳ شروع پیش‌پردازش اولیهٔ دادهٔ خام ...")
    df = pd.read_csv(input_csv)  # خواندن CSV خام 

    print("  • ستون‌ها:", df.columns.tolist())
    print("  • شکل جدول قبل از dropna:", df.shape)
    print("  • پنج ردیف اول:\n", df.head(), "\n")

    # حذف Pokémonهایی که یکی از ستون‌های عددی‌شان ناقص است
    df = df.dropna(subset=["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])
    print("  • شکل جدول پس از dropna:", df.shape, "\n")

    # تعیین X (ویژگی‌ها) و y (هدف)
    feature_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    X = df[feature_cols]
    y = df["Total"]

    print("پیش‌پردازش اولیه انجام شد.\n")
    return df, X, y


# --------------------------------------------
# 3. نمودارهای توزیع و همبستگی (Raw Data)
# --------------------------------------------
def plot_raw_data_insights(df):
    """
    رسم نمودارهای توزیع Total و ماتریس همبستگی شش ویژگی عددی
    و ذخیرهٔ آن‌ها در پوشهٔ charts/
    """
    # ۱. نمودار توزیع Total
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Total"], bins=30, kde=True, color="skyblue")
    plt.title("Distribution of Total Base Stats (Raw Data)")
    plt.xlabel("Total Base Stats")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path_total_hist = os.path.join(CHART_DIR, "raw_total_distribution.png")
    plt.savefig(path_total_hist)
    plt.close()
    print(f"✅ نمودار توزیع Total (Raw) در '{path_total_hist}' ذخیره شد.")

    # ۲. ماتریس همبستگی شش ویژگی عددی
    corr = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Base Stats (Raw Data)")
    plt.tight_layout()
    path_corr = os.path.join(CHART_DIR, "raw_correlation_matrix.png")
    plt.savefig(path_corr)
    plt.close()
    print(f"✅ ماتریس همبستگی (Raw) در '{path_corr}' ذخیره شد.")


# --------------------------------------------
# 4. مدل‌سازی روی دادهٔ خام
# --------------------------------------------
def model_on_raw_data(df, X, y):
    """
    چهار الگوریتم را روی دادهٔ خام اجرا می‌کند و نمودارهای زیر را نیز می‌سازد:
      ۱. K-Means → محاسبۀ Silhouette Score و ذخیرهٔ Silhouette vs k (Raw)
      ۲. Linear Regression → محاسبۀ MAE و RMSE و نمودار Actual vs Predicted
      ۳. Decision Tree Regressor → محاسبۀ MAE و RMSE و نمودار Actual vs Predicted
      ۴. Random Forest Regressor → محاسبۀ MAE و RMSE و نمودار Actual vs Predicted و Feature Importances (نمودار میله‌ای)
    """
    print("⏳ اجرای مدل‌ها روی دادهٔ خام ...")

    if X.shape[0] == 0:
        print("⚠️ دادهٔ خام هیچ رکوردی ندارد؛ ادامهٔ مدلسازی ممکن نیست.")
        return None

    results = {}

    # ---------- خوشه‌بندی K-Means و Silhouette vs k ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)

    # ذخیرهٔ نمودار Silhouette vs k (Raw)
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
    print(f"✅ Silhouette vs k (Raw) در '{path_sil_raw}' ذخیره شد.")

    # برای k=4 (پیش‌فرض)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"  • Silhouette Score (Raw Data, k=4): {sil_score:.4f}")
    results["silhouette_raw"] = sil_score

    # ---------- تقسیم به Train/Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- رگرسیون خطی (Linear Regression) ----------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"  • Linear Regression (Raw): MAE = {mae_lr:.4f}, RMSE = {rmse_lr:.4f}")
    results["lr_mae_raw"] = mae_lr
    results["lr_rmse_raw"] = rmse_lr

    # نمودار Actual vs Predicted (Linear Regression, Raw)
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
    print(f"✅ نمودار Actual vs Predicted (LR, Raw) در '{path_lr_raw}' ذخیره شد.")

    # ---------- درخت تصمیم (Decision Tree Regressor) ----------
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    print(f"  • Decision Tree (Raw): MAE = {mae_dt:.4f}, RMSE = {rmse_dt:.4f}")
    results["dt_mae_raw"] = mae_dt
    results["dt_rmse_raw"] = rmse_dt

    # نمودار Actual vs Predicted (Decision Tree, Raw)
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
    print(f"✅ نمودار Actual vs Predicted (DT, Raw) در '{path_dt_raw}' ذخیره شد.")

    # ---------- جنگل تصادفی (Random Forest Regressor) ----------
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"  • Random Forest (Raw): MAE = {mae_rf:.4f}, RMSE = {rmse_rf:.4f}")
    results["rf_mae_raw"] = mae_rf
    results["rf_rmse_raw"] = rmse_rf

    # نمودار Actual vs Predicted (Random Forest, Raw)
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
    print(f"✅ نمودار Actual vs Predicted (RF, Raw) در '{path_rf_raw}' ذخیره شد.")

    # نمودار Bar Chart اهمیت ویژگی‌ها (Random Forest, Raw)
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
    print(f"✅ نمودار Feature Importances (Raw) در '{path_fi_raw}' ذخیره شد.")

    return results


# --------------------------------------------
# 5. پاک‌سازی داده‌ها (Data Cleaning)
# --------------------------------------------
def clean_data(df, output_csv=CLEAN_CSV):
    """
    - حذف Pokémonهایی که Total < 100 یا Total > 700 (حذف مقادیر پرت)
    - حذف ردیف‌هایی که یکی از شش ویژگی آماری‌شان صفر است
    - ذخیرهٔ CSV پاک‌شده و بازگشت X_clean, y_clean
    """
    print("⏳ شروع پاک‌سازی داده‌ها ...")
    df_clean = df.copy()

    # حذف Outliers بر اساس Total (کمتر از 100 یا بیشتر از 700) 
    df_clean = df_clean[(df_clean["Total"] >= 100) & (df_clean["Total"] <= 700)]

    # حذف ردیف‌هایی که یکی از ویژگی‌های آماری‌شان صفر است
    df_clean = df_clean[(df_clean["HP"] > 0) &
                        (df_clean["Attack"] > 0) &
                        (df_clean["Defense"] > 0) &
                        (df_clean["Sp. Atk"] > 0) &
                        (df_clean["Sp. Def"] > 0) &
                        (df_clean["Speed"] > 0)]

    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")  # ذخیرهٔ CSV پاک‌شده 
    print(f"  • دادهٔ پاک‌شده ({len(df_clean)} رکورد) در '{output_csv}' ذخیره شد.\n")

    X_clean = df_clean[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    y_clean = df_clean["Total"]
    return df_clean, X_clean, y_clean


# --------------------------------------------
# 6. مدل‌سازی روی دادهٔ پاک‌شده
# --------------------------------------------
def model_on_clean_data(df_clean, X_clean, y_clean):
    """
    همان چهار الگوریتم را روی دادهٔ پاک‌شده اجرا می‌کند و نمودارهای زیر را نیز می‌سازد:
      ۱. K-Means → Silhouette Score و Silhouette vs k (Cleaned)
      ۲. Linear Regression → MAE, RMSE و نمودار Actual vs Predicted
      ۳. Decision Tree Regressor → MAE, RMSE و نمودار Actual vs Predicted
      ۴. Random Forest Regressor → MAE, RMSE و نمودار Actual vs Predicted و اهمیت ویژگی‌ها (نمودار میله‌ای)
    """
    print("⏳ اجرای مدل‌ها روی دادهٔ پاک‌شده ...")

    if X_clean.shape[0] == 0:
        print("⚠️ دادهٔ پاک‌شده هیچ رکوردی ندارد؛ ادامهٔ خوشه‌بندی ممکن نیست.")
        return None

    results = {}

    # ---------- خوشه‌بندی K-Means و Silhouette vs k (Cleaned) ----------
    scaler_c = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(X_clean)

    silhouette_scores_c = []
    K_range = range(2, 11)
    for k in K_range:
        km_c = KMeans(n_clusters=k, random_state=42)
        labels_c = km_c.fit_predict(Xc_scaled)
        sil_c = silhouette_score(Xc_scaled, labels_c)
        silhouette_scores_c.append(sil_c)

    # ذخیرهٔ نمودار Silhouette vs k (Cleaned)
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
    print(f"✅ Silhouette vs k (Cleaned) در '{path_sil_clean}' ذخیره شد.")

    # برای k=4 (پیش‌فرض)
    kmeans_c = KMeans(n_clusters=4, random_state=42)
    clusters_c = kmeans_c.fit_predict(Xc_scaled)
    sil_score_c = silhouette_score(Xc_scaled, clusters_c)
    print(f"  • Silhouette Score (Cleaned Data, k=4): {sil_score_c:.4f}")
    results["silhouette_clean"] = sil_score_c

    # ---------- تقسیم به Train/Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    # ---------- رگرسیون خطی (Linear Regression) ----------
    lr_c = LinearRegression()
    lr_c.fit(X_train, y_train)
    y_pred_lr_c = lr_c.predict(X_test)
    mae_lr_c = mean_absolute_error(y_test, y_pred_lr_c)
    rmse_lr_c = np.sqrt(mean_squared_error(y_test, y_pred_lr_c))
    print(f"  • Linear Regression (Cleaned): MAE = {mae_lr_c:.4f}, RMSE = {rmse_lr_c:.4f}")
    results["lr_mae_clean"] = mae_lr_c
    results["lr_rmse_clean"] = rmse_lr_c

    # نمودار Actual vs Predicted (Linear Regression, Cleaned)
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
    print(f"✅ نمودار Actual vs Predicted (LR, Cleaned) در '{path_lr_clean}' ذخیره شد.")

    # ---------- درخت تصمیم (Decision Tree Regressor) ----------
    dt_c = DecisionTreeRegressor(random_state=42)
    dt_c.fit(X_train, y_train)
    y_pred_dt_c = dt_c.predict(X_test)
    mae_dt_c = mean_absolute_error(y_test, y_pred_dt_c)
    rmse_dt_c = np.sqrt(mean_squared_error(y_test, y_pred_dt_c))
    print(f"  • Decision Tree (Cleaned): MAE = {mae_dt_c:.4f}, RMSE = {rmse_dt_c:.4f}")
    results["dt_mae_clean"] = mae_dt_c
    results["dt_rmse_clean"] = rmse_dt_c

    # نمودار Actual vs Predicted (Decision Tree, Cleaned)
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
    print(f"✅ نمودار Actual vs Predicted (DT, Cleaned) در '{path_dt_clean}' ذخیره شد.")

    # ---------- جنگل تصادفی (Random Forest Regressor) ----------
    rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_c.fit(X_train, y_train)
    y_pred_rf_c = rf_c.predict(X_test)
    mae_rf_c = mean_absolute_error(y_test, y_pred_rf_c)
    rmse_rf_c = np.sqrt(mean_squared_error(y_test, y_pred_rf_c))
    print(f"  • Random Forest (Cleaned): MAE = {mae_rf_c:.4f}, RMSE = {rmse_rf_c:.4f}")
    results["rf_mae_clean"] = mae_rf_c
    results["rf_rmse_clean"] = rmse_rf_c

    # نمودار Actual vs Predicted (Random Forest, Cleaned)
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
    print(f"✅ نمودار Actual vs Predicted (RF, Cleaned) در '{path_rf_clean}' ذخیره شد.")

    # نمودار میله‌ای اهمیت ویژگی‌ها (Random Forest, Cleaned)
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
    print(f"✅ نمودار Feature Importances (Cleaned) در '{path_fi_clean}' ذخیره شد.")

    return results


# --------------------------------------------
# 7. تابع اصلی برای اجرای همهٔ مراحل
# --------------------------------------------
def main():
    # 1. اسکرپ داده‌ها و ذخیرهٔ CSV خام
    df_raw = scrape_pokemon_stats(output_csv=RAW_CSV)

    # 2. پیش‌پردازش اولیهٔ دادهٔ خام
    df_raw, X_raw, y_raw = initial_preprocessing(input_csv=RAW_CSV)

    # 3. ترسیم نمودارهای توزیع و همبستگی (Raw Data)
    plot_raw_data_insights(df_raw)

    # 4. مدل‌سازی روی دادهٔ خام و نمودارهای مربوطه
    results_raw = model_on_raw_data(df_raw, X_raw, y_raw)

    # 5. پاک‌سازی داده‌ها
    df_clean, X_clean, y_clean = clean_data(df_raw, output_csv=CLEAN_CSV)

    # 6. مدل‌سازی روی دادهٔ پاک‌شده و نمودارهای مربوطه
    results_clean = model_on_clean_data(df_clean, X_clean, y_clean)

    # 7. نمایش و مقایسهٔ نتایج
    print("\n--- مقایسهٔ نتایج قبل و بعد از پاک‌سازی داده‌ها ---")
    if results_raw:
        print("نتایج اولیه (Raw Data):")
        for k, v in results_raw.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("نتایج اولیه (Raw Data): NA")

    if results_clean:
        print("\nنتایج پس از پاک‌سازی (Cleaned Data):")
        for k, v in results_clean.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nنتایج پس از پاک‌سازی (Cleaned Data): NA")

    print("\n✅ پایان اجرای پروژه.")

if __name__ == "__main__":
    main()
