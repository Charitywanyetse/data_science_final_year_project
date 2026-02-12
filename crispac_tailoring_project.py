import pandas as pd

# load the dataset
df = pd.read_excel("tailoring_sales_dataset.xlsx")

print(df.head())
print(df.tail())
print(df.info())

# Exploratory data analysis.

import matplotlib.pyplot as plt

# Most sold wear

df.groupby("wear_type")["quantity_sold"].sum().plot(kind="bar")
plt.title("Most sold wear. ")
plt.ylabel("Quantity Sold. ")
plt.show()

# Monthly demand trend. 

df.groupby("month")["quantity_sold"].sum().plot()
plt.title("Monthly Demand Trend. ")
plt.xlabel("Month. ")
plt.ylabel("Quantity Sold. ")
plt.show()

# Preparing data for machine learning

# convert data.
df["date"] = pd.to_datetime(df["date"])


# encode categorical columns

df_encode = pd.get_dummies(df, columns=["sector", "wear_type", "season"])

# features and target;

X = df_encode.drop("quantity_sold", axis=1)
y = df_encode["quantity_sold"]


# TRAINING MY PREDICTION MODEL.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train model

model = RandomForestRegressor()
model.fit(X_train, y_train)

# predictions 

predictions = model.predict(X_test)

# Evaluation

mae = mean_absolute_error(y_test, predictions)
print("model mean absolute error:", mae)


# MAKING FUTURE PREDICTIONS

# prediction for the school uniforms in january

sample = Xiloc[[0]]
future_prediction = model.predict(sample)
print("predicted quantity:", future_prediction)














