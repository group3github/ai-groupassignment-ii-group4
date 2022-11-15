# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Classification Functions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Regression Function
from sklearn.linear_model import LinearRegression

### CLASSIFICATION PROBLEMS ###

# Load the data from the houses.csv file using pandas
my_data = pd.read_csv('houses.csv')

# remove rows with missing values e.g. NaN values as our algorithms only work with numbers
my_data.dropna()
print()

# Select input feature columns
feature_column_names = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                        'view', 'sqft_above', 'sqft_basement', 'yr_built']

X = my_data[feature_column_names].values

# Select target column
Y = my_data['condition'].values

# Split Dataset: 66% training, 33% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)

# Train using KNN Algorithm
knn = KNeighborsClassifier().fit(X_train, Y_train)
# Make predictions of target value based on test features
knn_prediction = knn.predict(X_test)

# Train using Decision Tree Algorithm
dt = DecisionTreeClassifier().fit(X_train, Y_train)
# Make predictions of target value based on test features
dt_prediction = dt.predict(X_test)

# Calculate Accuracy of KNN Algorithm Predictions
knn_accuracy = accuracy_score(Y_test, knn_prediction) * 100
print('Accuracy of KNN model: ' + str(knn_accuracy) + '%')

# Calculate Accuracy of Decision Tree Algorithm Predictions
dt_accuracy = accuracy_score(Y_test, dt_prediction) * 100
print('Accuracy of Decision Tree model: ' + str(dt_accuracy) + '%')

### REGRESSION PROBLEM ###

# Load the data from the grades.csv file using pandas
dataset = pd.read_csv('grades.csv')

# Set 'Hours' as X and 'Scores' as Y for our Linear Regression Model and plot their relationship
lr_X = dataset['Hours']
lr_Y = dataset['Scores']

plt.scatter(lr_X, lr_Y)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours / Scores Relationship')
plt.show()

# Reshape the X into a 2D array that can be accepted by the LinearRegression() function
lr_X = np.array(lr_X).reshape(-1,1)

# Split dataset: 66% training, 33% testing
lr_X_train, lr_X_test, lr_Y_train, lr_Y_test = train_test_split(lr_X, lr_Y, test_size = 0.33)

# Train the LR model on the training data
lr = LinearRegression().fit(lr_X_train, lr_Y_train)

# Plot linear regression line
line = lr.coef_ * lr_X + lr.intercept_

plt.plot(lr_X, line, color='red')
plt.scatter(lr_X, lr_Y, color='blue')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours / Scores Relationship')
plt.show()

# Calculate Accuracy of Linear Regression Model
lr_predictions = lr.predict(lr_X_test)
print('Linear Regression Model Accuracy: ' + str(lr.score(lr_X_test, lr_Y_test) * 100) + '%')
