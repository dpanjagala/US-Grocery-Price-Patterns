# Import libraries and load dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_path = 'datasets/fmap_data.xlsx'
df = pd.read_excel(data_path, sheet_name="Data")


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

# write to file
df_clean.to_csv("datasets/cleaned/cleaned_fmap_data.csv", index=False)


