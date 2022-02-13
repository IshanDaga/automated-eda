
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("datasets/cars.csv")
df = df.drop(["vin", "lot", "state", "country"], axis=1)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])

print(df.describe())