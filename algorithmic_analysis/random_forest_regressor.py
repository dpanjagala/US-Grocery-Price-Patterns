# === RANDOM FOREST REGRESSOR TEMPLATE ===

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = master_df.copy()

# --------------------
# 1. Feature selection
# --------------------
features = [
    "region",     
    "category",
    "bls_cpi",
    "pce_spend",
    "is_post_2020",
    "month"
]

X = df[features]
y = df["fmap_price"]

# -------------------------------
# 2. OneHotEncode categorical vars
# -------------------------------
categorical_cols = ["region", "category", "month"]
numeric_cols = ["bls_cpi", "pce_spend", "is_post_2020"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

X_transformed = preprocessor.fit_transform(X)

# --------------------
# 3. Train/test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

# --------------------
# 4. Fit model
# --------------------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# --------------------
# 5. Evaluate
# --------------------
y_pred = rf.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --------------------
# 6. Feature importance
# --------------------
feature_names = (
    preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    + numeric_cols
)

importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=False)

display(feat_imp_df.head(20))
