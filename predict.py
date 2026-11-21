import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainingdata = pd.read_csv('train.csv')

print("Train shape:", trainingdata.shape)
#print("Test shape:", test.shape)

print(trainingdata.head())
print(trainingdata.info())

print("\nMissing values:")
print(trainingdata.isnull().sum().sort_values(ascending=False))

print("\nTarget statistics:")
print(trainingdata["hours_spent"].describe())