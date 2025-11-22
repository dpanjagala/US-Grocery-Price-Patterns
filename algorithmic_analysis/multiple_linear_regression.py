# === MULTIPLE LINEAR REGRESSION TEMPLATE ===

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# df = your merged FMAP + BLS + PCE dataset
# Target variable example: 'fmap_price'

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

# -------------------------------------
# 2. OneHotEncode categorical variables
# -------------------------------------
categorical_cols = ["region", "category", "month"]
numeric_cols = ["bls_cpi", "pce_spend", "is_post_2020"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# --------------------
# 3. Train/test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# --------------------
# 4. Fit model
# --------------------
model = LinearRegression()
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

model.fit(X_train_transformed, y_train)

# --------------------
# 5. Evaluate
# --------------------
y_pred = model.predict(X_test_transformed)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --------------------
# 6. Get coefficients
# --------------------
feature_names = (
    preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    + numeric_cols
)

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": model.coef_
}).sort_values(by="coefficient", ascending=False)

display(coef_df.head(20))
