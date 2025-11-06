# Import libraries and load dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_path = 'datasets/pce_data.csv'
df = pd.read_csv(data_path)

print("Rows, Columns:", df.shape)
df.head()

# look at raw data characteristics

print("\n--- DATA INFO ---")
df.info()

print("\n--- DATA SUMMARY ---")
print(df.describe(include='all'))


# see quality issues

print("Missing values per column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# implementation of data cleaning

df_clean = df.copy()

# remove whitespace from cells
df = df.replace(r"^ +| +$", r"", regex=True)

#drop missing values since there aren't too many relative to how many rows there are
df_clean = df.dropna()

# write to file
df_clean.to_csv("datasets/cleaned/cleaned_pce_data.csv", index=False)


