import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/iris.csv")

# Example: encode target if needed
if 'target' in df.columns:
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

df.to_csv("data/processed/iris_cleaned.csv", index=False)
print("Preprocessed data saved.")