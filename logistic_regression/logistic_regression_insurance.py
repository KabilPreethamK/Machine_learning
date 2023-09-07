from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("insurance_data.csv")
plt.scatter(data.age , data.bought_insurance , color = "red" , marker="+")
logic  = linear_model.LogisticRegression()
X_train, X_test , y_train , y_test = train_test_split(data[["age"]],data.bought_insurance,test_size=0.1)
#print(X_test)
logic.fit(X_train,y_train)
#print(logic.predict(X_test))
plt.plot(X_train , logic.predict(y_train))
plt.show()