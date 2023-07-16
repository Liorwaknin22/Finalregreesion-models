import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


class LinearRegressionModel:
    def __init__(self, learning_rate=0.005, n_iterations=6000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            y_predicted = self._approximation(X, self.weights, self.bias)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._approximation(X, self.weights, self.bias)

    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def get_coefficients(self):
        return self.weights

    def get_intercept(self):
        return self.bias


# # Create a random regression problem
# X, y = datasets.make_regression(n_samples=100, n_features=3, noise=20, random_state=4)
#
# # Split dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
# # Create an instance of the LinearRegressionModel class and fit the model
# model_custom = LinearRegressionModel()
# model_custom.fit(X_train, y_train)
#
# # Create an instance of the LinearRegression class from sklearn and fit the model
# model_sklearn = LinearRegression()
# model_sklearn.fit(X_train, y_train)
#
# # Print the coefficients and intercepts of both models
# print("Custom model coefficients: ", model_custom.get_coefficients())
# print("Sklearn model coefficients: ", model_sklearn.coef_)
#
# print("Custom model intercept: ", model_custom.get_intercept())
# print("Sklearn model intercept: ", model_sklearn.intercept_)
train_data = pd.read_csv('C:\\Users\\liorw\\Downloads\\train3.csv\\train.csv')

print("testing linear regression")
linear_data = train_data
linear_data.dropna(inplace=True)
linear_data.reset_index(drop=True, inplace=True)

X = linear_data[["host_listings_count", "host_total_listings_count", "beds", "review_scores_rating"]]
y = linear_data["expensive"]
linear_data.reset_index(drop=True, inplace=True)

print("my model coff")
my_model = LinearRegressionModel()
my_model.fit(X, y)
print("Custom model coefficients: ", my_model.get_coefficients())
print("sklean model coff ")
sk_model = SklearnLinearRegression()
sk_model.fit(X, y)
print(sk_model.coef_)
