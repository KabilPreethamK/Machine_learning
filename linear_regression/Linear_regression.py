import pickle
with open("linear_reg","rb") as f:
    model = pickle.load(f)

print(model.predict([[5000]]))