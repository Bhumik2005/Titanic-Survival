import pandas as pd

df = pd.read_csv('data/train.csv')
print("shape of dataset:", df.shape)
print("\ncolumns")
print(df.columns)
print("\nfirst 5 columns")
print(df.head())
print("\nMissing data")
print(df.isnull().sum())
print("\nSurvival count:")
print(df["Survived"].value_counts())