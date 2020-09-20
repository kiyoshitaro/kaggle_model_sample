
import pandas as pd


df = pd.read_csv('houseprice/clean_data.csv')
y = df["SalePrice"]
df.drop(["SalePrice"],inplace = True)
