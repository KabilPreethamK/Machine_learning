import pandas as pd 
from sklearn import linear_model
from word2number import w2n
import math

data = pd.read_csv("streak.csv")

#convertion string to int


reg = linear_model.LogisticRegression()
reg.fit(data[["relation","doj"]],data.streak)
print(reg.predict([[1,164]]))

