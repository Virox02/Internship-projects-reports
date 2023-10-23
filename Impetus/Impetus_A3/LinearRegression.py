import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
ecommerce_df = pd.read_csv('Ecommerce_Customers.csv')

# Select only the numerical columns
numerical_columns = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
X = ecommerce_df[numerical_columns]
y = ecommerce_df['Yearly Amount Spent']

# Split the data into training and testing sets
train_size = int(0.75 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Add a column of ones to the feature matrix to represent the intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Calculate the coefficients using the formula
coefficients = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

print(coefficients)

# Add a column of ones to the feature matrix to represent the intercept term
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Calculate the predicted values
y_pred = X_test.dot(coefficients)

print(y_pred)

# Calculate the mean squared error
mse = np.mean((y_pred - y_test) ** 2)

print(mse)

def residualplot():
    # Calculate residuals
    residuals = y_test - y_pred

    # Create a residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, c='b', marker='o', label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual Line')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot for Multiple Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

residualplot()