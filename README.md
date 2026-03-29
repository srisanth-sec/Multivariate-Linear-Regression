# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
Import required libraries and load the dataset.

### Step2

Define the feature matrix � and target vector �
### Step3
Split the dataset into training and testing sets

### Step4
Create a Linear Regression model and train it using training data.

### Step5
Predict outputs, evaluate the model, and plot the residual errors.

## Program:
```
# Multivariate Linear Regression using scikit-learn (Updated)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing()
X = data.data        # multiple independent variables
y = data.target      # dependent variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output coefficients
print("Regression Coefficients:")
print(model.coef_)
print("Intercept:", model.intercept_)

# R2 score
print("R2 Score:", r2_score(y_test, y_pred))

# Residual plot
plt.scatter(y_pred, y_pred - y_test, color="blue", s=10, label="Test data")
plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10, label="Train data")

plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), linewidth=2)
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Errors")
plt.legend()
plt.show()







```
## Output:
<img width="793" height="606" alt="Screenshot 2026-03-29 140743" src="https://github.com/user-attachments/assets/fa777156-d913-4a4d-84d9-6e1ef2a484d9" />


### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
