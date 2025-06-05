"""
پروژه: اسکرپ Pokémon DB – Pokedex All و تحلیل داده‌ها با ۴ الگوریتم یادگیری ماشین
مراحل:
1. جمع‌آوری داده‌ها از Pokémon DB (Pokedex All)
2. پیش‌پردازش اولیهٔ دادهٔ خام
3. مدل‌سازی (K-Means, Linear Regression, Decision Tree, Random Forest) روی دادهٔ خام
4. پاک‌سازی داده‌ها (حذف Outliers)
5. مدل‌سازی مجدد روی دادهٔ پاک‌شده
6. مقایسهٔ معیارها (Silhouette Score, MAE, RMSE) قبل و بعد از پاک‌سازی
"""

# ----------------------------
# وارد کردن تمام کتابخانه‌ها
# ----------------------------
import requests                      # برای ارسال درخواست HTTP :contentReference[oaicite:4]{index=4}
from bs4 import BeautifulSoup        # برای پارس HTML :contentReference[oaicite:5]{index=5}
import pandas as pd                  # برای کار با DataFrame و CSV 
import numpy as np                   # برای محاسبات عددی

from sklearn.cluster import KMeans   # خوشه‌بندی K-Means 
from sklearn.linear_model import LinearRegression  # رگرسیون خطی 
from sklearn.tree import DecisionTreeRegressor      # درخت تصمیم 
from sklearn.ensemble import RandomForestRegressor  # جنگل تصادفی 

from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error  # معیارها: Silhouette, MAE, RMSE 
from sklearn.model_selection import train_test_split   # تقسیم داده به Train/Test 
from sklearn.preprocessing import StandardScaler        # برای مقیاس‌بندی ویژگی‌ها در خوشه‌بندی 

import matplotlib.pyplot as plt  # برای نمایش نمودار (اختیاری) 


# --------------------------------------------
# ثابت‌ها و پارامترهای اصلی
# --------------------------------------------
RAW_CSV = "pokemon_raw.csv"
CLEAN_CSV = "pokemon_clean.csv"


# --------------------------------------------
# 1. تابع اسکرپ جدول Pokémon
# --------------------------------------------
def scrape_pokemon_stats(output_csv=RAW_CSV):
    """
    این تابع صفحهٔ Pokedex All را اسکرپ می‌کند و جدول را به DataFrame تبدیل و در خروجی ذخیره می‌کند.
    ساختار فعلی جدول Pokémon DB شبیه به Medium توضیح داده شده است (۱۰ ستون در هر سطر) :contentReference[oaicite:15]{index=15}.
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

    # استخراج سطرهای جدول (هر <tr> شامل 10 <td> است) :contentReference[oaicite:16]{index=16}
    rows = table.find("tbody").find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        # بررسی اینکه دقیقاً 10 ستون داریم
        if len(cols) < 10:
            continue

        # ستون‌ها (اندیس 0 تا 9): #, Name, Type, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed
        pokedex_no = cols[0].text.strip()                        # شماره Pokédex :contentReference[oaicite:17]{index=17}
        name = cols[1].text.strip()                               # نام Pokémon
        types = [t.text.strip() for t in cols[2].find_all("a")]  # لیست نوع (ممکن است چند نوع) :contentReference[oaicite:18]{index=18}
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
# 3. مدل‌سازی روی دادهٔ خام
# --------------------------------------------
def model_on_raw_data(df, X, y):
    """
    چهار الگوریتم را روی دادهٔ خام اجرا می‌کند:
      ۱. K-Means → محاسبۀ Silhouette Score
      ۲. Linear Regression → محاسبۀ MAE و RMSE
      ۳. Decision Tree Regressor → محاسبۀ MAE و RMSE
      ۴. Random Forest Regressor → محاسبۀ MAE و RMSE و نمایش Feature Importances
    """
    print("⏳ اجرای مدل‌ها روی دادهٔ خام ...")

    if X.shape[0] == 0:
        print("⚠️ دادهٔ خام هیچ رکوردی ندارد؛ ادامهٔ مدلسازی ممکن نیست.")
        return None

    # ۱. خوشه‌بندی با K-Means
    scaler = StandardScaler()                                  # مقیاس‌بندی داده برای خوشه‌بندی 
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)             # ۴ خوشه (عدد دلخواه) 
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)           # محاسبۀ Silhouette Score 
    print(f"  • Silhouette Score (Raw Data): {sil_score:.4f}")

    # ۲. تقسیم داده به Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )  # 

    # ۳. رگرسیون خطی (Linear Regression)
    lr = LinearRegression()                                    # 
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)            # 
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))   # 
    print(f"  • Linear Regression (Raw): MAE = {mae_lr:.4f}, RMSE = {rmse_lr:.4f}")

    # ۴. درخت تصمیم (Decision Tree Regressor)
    dt = DecisionTreeRegressor(random_state=42)                 # 
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    print(f"  • Decision Tree (Raw): MAE = {mae_dt:.4f}, RMSE = {rmse_dt:.4f}")

    # ۵. جنگل تصادفی (Random Forest Regressor)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"  • Random Forest (Raw): MAE = {mae_rf:.4f}, RMSE = {rmse_rf:.4f}")

    # نمایش اهمیت ویژگی‌ها (Feature Importances)
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
# 4. پاک‌سازی داده‌ها (Data Cleaning)
# --------------------------------------------
def clean_data(df, output_csv=CLEAN_CSV):
    """
    - حذف Pokémonهایی که Total < 100 یا Total > 700 (حذف مقادیر پرت)
    - حذف ردیف‌هایی که یکی از شش ویژگی آماری‌شان صفر است
    - ذخیرهٔ CSV پاک‌شده و بازگشت X_clean, y_clean
    """
    print("⏳ شروع پاک‌سازی داده‌ها ...")
    df_clean = df.copy()

    # حذف Outliers بر اساس Total (کمتر از 100 یا بیشتر از 700) :contentReference[oaicite:30]{index=30}
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
# 5. مدل‌سازی روی دادهٔ پاک‌شده
# --------------------------------------------
def model_on_clean_data(df_clean, X_clean, y_clean):
    """
    همان چهار الگوریتم را روی دادهٔ پاک‌شده اجرا می‌کند:
      ۱. K-Means → Silhouette Score
      ۲. Linear Regression → MAE, RMSE
      ۳. Decision Tree Regressor → MAE, RMSE
      ۴. Random Forest Regressor → MAE, RMSE + Feature Importances
    """
    print("⏳ اجرای مدل‌ها روی دادهٔ پاک‌شده ...")

    if X_clean.shape[0] == 0:
        print("⚠️ دادهٔ پاک‌شده هیچ رکوردی ندارد؛ ادامهٔ خوشه‌بندی ممکن نیست.")
        return None

    # خوشه‌بندی K-Means
    scaler_c = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(X_clean)
    kmeans_c = KMeans(n_clusters=4, random_state=42)
    clusters_c = kmeans_c.fit_predict(Xc_scaled)
    sil_score_c = silhouette_score(Xc_scaled, clusters_c)
    print(f"  • Silhouette Score (Cleaned Data): {sil_score_c:.4f}")

    # تقسیم به Train/Test
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

    # اهمیت ویژگی‌ها (Feature Importances)
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
# 6. تابع اصلی برای اجرای همهٔ مراحل
# --------------------------------------------
def main():
    # 1. اسکرپ داده‌ها و ذخیرهٔ CSV خام
    df_raw = scrape_pokemon_stats(output_csv=RAW_CSV)

    # 2. پیش‌پردازش اولیهٔ دادهٔ خام
    df_raw, X_raw, y_raw = initial_preprocessing(input_csv=RAW_CSV)

    # 3. مدل‌سازی روی دادهٔ خام
    results_raw = model_on_raw_data(df_raw, X_raw, y_raw)

    # 4. پاک‌سازی داده‌ها
    df_clean, X_clean, y_clean = clean_data(df_raw, output_csv=CLEAN_CSV)

    # 5. مدل‌سازی روی دادهٔ پاک‌شده
    results_clean = model_on_clean_data(df_clean, X_clean, y_clean)

    # 6. نمایش و مقایسهٔ نتایج
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
