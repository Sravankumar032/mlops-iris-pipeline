import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/iris.csv")

# Example: encode target if needed
if 'target' in df.columns:
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

df.to_csv("data/processed/processed.csv", index=False)
print("Preprocessed data saved.")