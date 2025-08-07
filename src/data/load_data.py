from sklearn.datasets import load_iris
import pandas as pd
import os

def save_iris_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/iris.csv", index=False)
    print("Dataset saved to data/raw/iris.csv")

if __name__ == "__main__":
    save_iris_dataset()