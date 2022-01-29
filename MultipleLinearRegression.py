import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("multilinearregression.csv",sep=";")
df.head()

reg=linear_model.LinearRegression()
reg.fit(df[["alan","odasayisi","binayasi"]],df["fiyat"])

reg.predict([[250,4,5]])
reg.predict([[300,6,0]])
reg.predict([[300,4,0]])
reg.predict([[300,4,5]])
reg.predict([[355,4,6],[300,3,2],[322,6,0]])

reg.coef_ #katsayılar
reg.intercept_ #sabit değer

a=reg.intercept_
b1=reg.coef_[0]
b2=reg.coef_[0]
b3=reg.coef_[2]

x1=300
x2=4
x3=5

y=a+b1*x1+b2*x2+b3*x3
y
