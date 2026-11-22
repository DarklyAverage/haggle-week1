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

plt.figure(figsize=(10, 5))

sns.histplot(
    trainingdata["hours_spent"],
    bins=40,
    kde=True
)

plt.title("Distribution of Approved Hours")
plt.xlabel("Hours Spent")
plt.ylabel("Number of Projects")

plt.show()


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

drop_cols = [
    "hours_spent",
    "log_hours",
    "hours_bucket",
]

X = trainingdata.drop(columns=drop_cols)
y = trainingdata["hours_spent"]

#X_test = test.drop(columns=["project_id"])

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])



