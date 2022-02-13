
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("datasets/cars.csv")
print(df)
print(df.columns)
# drop the following columsn: vin,lot,state,country
df = df.drop(['vin', 'lot', 'state', 'country'], axis=1)
print(df)


# use a label encoder on columns with string data and store the output in a new dataframe
le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':  # if column datatype is object (string) then convert to categorical values (numbers) and store in new dataframe called df_new 
        df[col] = le.fit_transform(df[col])  # fit and transform the column values with label encoder and store in same dataframe

        

        

