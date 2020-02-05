import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np

x_test = pd.read_csv("preprocessed_test.csv")
#print(x_test.head())
#print(x_test.columns)

x_test = x_test.drop(["Unnamed: 0"],axis=1)
x_test.dropna()

# Load the Model back from file
with open("temp_models/mymodel", 'rb') as file:  
    model = pickle.load(file)

new_pred = model.predict(x_test)
#print(new_pred)

c = np.where(new_pred > 0,"Disaster","Non-Disaster")
print("\n\n")
print(c)
print("\n\n")
'''
y_test = pd.read_csv("temp/y_test_data.csv",names=["temp","output"])
y_test = y_test.drop(["temp"],axis=1)
print(y_test)
print(y_test.columns)
print(new_pred)

print("accuracy score:")
print(accuracy_score(y_test, new_pred))
print("confusion matrix for the data:")
print(confusion_matrix(y_test, new_pred))


org_data = pd.read_csv("temp/text_data_test.csv")
org_data = org_data.drop(["Unnamed: 0"],axis=1)
print(org_data)
'''
