import pandas as pd

areas = pd.read_csv("datasets/bls_areas.csv")
food_vals = pd.read_csv("datasets/bls_food_values.csv")
items = pd.read_csv("datasets/bls_items.csv")
series_info = pd.read_csv("datasets/bls_series_info.csv")

areas.columns = areas.columns.str.strip()
food_vals.columns = food_vals.columns.str.strip()
items.columns = items.columns.str.strip()
series_info.columns = series_info.columns.str.strip()

food_vals = food_vals[['series_id', 'year', 'value']]
series_info = series_info[['series_id', 'area_code', 'item_code']]

print(areas.columns)
print(food_vals.columns)
print(items.columns)
print(series_info.columns)

merge_areas_series = areas.merge(series_info, on="area_code", how="outer")
merge_areas_series_food = merge_areas_series.merge(food_vals, on="series_id", how="outer")
merge_all = merge_areas_series_food.merge(items, on="item_code")

merge_all.to_csv("merged_bls_table.csv", index=False)
