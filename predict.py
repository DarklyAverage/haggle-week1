#brooo theres so many imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


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

drop_cols = [
    "hours_spent",
    "log_hours",
    "hours_bucket",
    "project_id"
]

X = trainingdata.drop(columns=drop_cols)

#X_test = test.drop(columns=["project_id"])

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

y = trainingdata["log_hours"]

numeric_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols)
    ])),
    ("regressor", LinearRegression())
])

numeric_model.fit(X[numeric_cols], y)

#predictions = np.expm1(predictions)
#predictions = np.clip(predictions, 0, 140.7)

full_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

full_model.fit(X, y)