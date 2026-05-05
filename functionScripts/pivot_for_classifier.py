import pandas as pd

df = pd.read_csv("C:\\Users\\LocalAdmin\\Linus\\aboharbdavoudian2025-main\\Data\\merged_lightsheet_data.csv")

# remove "_normalized" from dataset name if present
df["dataset"] = df["dataset"].str.replace("_normalized", "", regex=False)

# pivot to wide format
wide_df = df.pivot(
    index="dataset",
    columns="abbreviation",
    values="count_norm"
)

# optional: sort columns
wide_df = wide_df.sort_index(axis=1)

# save for classifier
wide_df.to_csv("C:\\Users\\LocalAdmin\\Linus\\aboharbdavoudian2025-main\\Data\\lightsheet_for_classifier.csv")