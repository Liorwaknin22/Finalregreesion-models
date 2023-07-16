import random

import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score


class CustomPCA:
    """
    Principal component analysis (PCA):
        1. Standardize the continuous initial variables
        2.Compute the covariance matrix
        3.Compute the eigenvectors and eigenvalues of the covariance matrix
        4.Select the top-k eigenvectors by their corresponding eigenvalues
        5.Transform the original data along the axes of the principal component
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X):
        # Standardize the data
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

        # Calculate the covariance matrix
        cov_mat = np.cov(X_std, rowvar=False, bias=False)

        # Perform eigen decomposition
        w, v = LA.eig(cov_mat)

        # Sort the eigenvalues and eigenvectors in decreasing order
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]

        # Select the top n_components eigenvalues and eigenvectors
        self.eigenvalues = w[:self.n_components]
        self.eigenvectors = v[:, :self.n_components]

    def transform(self, X):
        # Standardize the data
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

        # Project the data onto the principal components
        return np.matmul(X_std, self.eigenvectors)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, n_iterations=3000):
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
        return np.dot(X, w) + b  # Create a sample dataset


class LogisticRegressionModel:
    """
    A class which implements logistic regression model with gradient descent.
    """

    def __init__(self, learning_rate=0.1, n_iterations=3000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    @staticmethod
    def _sigmoid(x):
        """
        Private method, used to pass results of the line equation through the sigmoid function.

        :param x: float, prediction made by the line equation
        :return: float
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _binary_cross_entropy(y, y_hat):
        '''
        Private method, used to calculate binary cross entropy value between actual classes
        and predicted probabilities.

        :param y: array, true class labels
        :param y_hat: array, predicted probabilities
        :return: float
        '''

        def safe_log(x):
            return 0 if x == 0 else np.log(x)

        total = 0
        for curr_y, curr_y_hat in zip(y, y_hat):
            total += (curr_y * safe_log(curr_y_hat) + (1 - curr_y) * safe_log(1 - curr_y_hat))
        return - total / len(y)

    def fit(self, X, y):
        '''
        Used to calculate the coefficient of the logistic regression model.

        :param X: array, features
        :param y: array, true values
        :return: None
        '''
        # 1. Initialize coefficients
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 2. Perform gradient descent
        for i in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            probability = self._sigmoid(linear_pred)

            # Calculate derivatives
            partial_w = (1 / X.shape[0]) * (np.dot(X.T, (probability - y)))
            partial_d = (1 / X.shape[0]) * (np.sum(probability - y))

            # Update the coefficients
            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def predict_proba(self, X):
        '''
        Calculates prediction probabilities for a given threshold using the line equation
        passed through the sigmoid function.

        :param X: array, features
        :return: array, prediction probabilities
        '''
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        '''
        Makes predictions using the line equation passed through the sigmoid function.

        :param X: array, features
        :param threshold: float, classification threshold
        :return: array, predictions
        '''
        probabilities = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probabilities]


def preprocess_data(data):
    # Drop specified columns
    data.drop(['id', 'host_id', 'has_availability', 'first_review', 'last_review'], axis=1, inplace=True)
    # Fill missing values
    data['host_response_time'] = data['host_response_time'].fillna('No Response')
    data['host_response_rate'] = data['host_response_rate'].fillna('0%')
    data['host_acceptance_rate'] = data['host_acceptance_rate'].fillna('0%')
    data['host_is_superhost'].fillna(data['host_is_superhost'].mode()[0], inplace=True)
    data['bathrooms_text'].fillna(data['bathrooms_text'].mode()[0], inplace=True)
    data['bedrooms'].fillna(data['bedrooms'].median(), inplace=True)
    data['beds'].fillna(data['beds'].median(), inplace=True)
    num_records = len(data)
    review_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                      'review_scores_checkin', 'review_scores_communication',
                      'review_scores_location', 'review_scores_value', 'reviews_per_month']
    for column in review_columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna('No Review')
        else:
            data[column].fillna(0, inplace=True)

    # Fill remaining missing values
    nights_columns = ['minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
                      'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm']
    for column in nights_columns:
        data[column].fillna(data[column].median(), inplace=True)
    data['license'].fillna('No License', inplace=True)

    # Identify columns with 't' and 'f' values
    boolean_columns = [col for col in data.columns if set(data[col].unique()) == {'f', 't'}]

    # Convert 't' and 'f' values to binary
    for col in boolean_columns:
        data[col] = data[col].map({'f': 0, 't': 1})

    # Convert to datetime format and calculate the number of days from the date to now
    date_columns = ['host_since']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
        data[col] = (datetime.now() - data[col]).dt.days

    # Apply MinMaxScaler to date columns
    scaler = MinMaxScaler()
    data[date_columns] = scaler.fit_transform(data[date_columns])

    # Convert 'amenities' to the sum of the list
    data['amenities'] = data['amenities'].apply(lambda x: len(eval(x)))

    # Create dummy variables for 'room_type'
    room_type_dummies = pd.get_dummies(data['room_type'], prefix='room_type')
    data.drop('room_type', axis=1, inplace=True)
    data = pd.concat([room_type_dummies, data], axis=1)

    # Create dummy variables for 'host_verifications'
    unique_verifications = data['host_verifications'].apply(eval).explode().unique()[:3]
    host_verifications_dummies = pd.DataFrame()
    for verification in unique_verifications:
        column_name = 'host_verification_' + verification
        host_verifications_dummies[column_name] = data['host_verifications'].apply(
            lambda row: verification in row).astype(int)

    # Concatenate the dummy variables at the beginning of the DataFrame
    data = pd.concat([host_verifications_dummies, data], axis=1)
    data.drop('host_verifications', axis=1, inplace=True)

    data = data.select_dtypes(exclude=['object'])  ##??

    # Detect anomalies in the numerical columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    anomalies = {}
    for column in numerical_columns:
        mean = data[column].mean()
        std = data[column].std()
        anomalies[column] = data[(data[column] < mean - 3 * std) | (data[column] > mean + 3 * std)]

    # Apply StandardScaler to each column (excluding 'expensive')
    scaler = StandardScaler()
    data_scaled = data.copy()  # Create a copy of the original DataFrame
    columns_to_scale = [col for col in data.columns if col != 'expensive']
    data_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data_scaled


def testLogisticRegression3(train_data):
    # Load test dataset
    test_data = pd.read_csv('C:\\Users\\liorw\\Downloads\\test3.csv\\test.csv')
    id = test_data['id']
    train_data_scaled = preprocess_data(train_data)

    X_train = train_data_scaled.drop(columns=['expensive'])
    y_train = train_data_scaled['expensive']

    X_test = preprocess_data(test_data)

    # Use your custom Logistic Regression class
    custom_lr = LogisticRegressionModel(learning_rate=0.1, n_iterations=1000)
    custom_lr.fit(X_train.values, y_train.values)
    custom_probabilities = custom_lr.predict_proba(X_test.values)
    return id, custom_probabilities


def comparing_VS_Sklearn(data):
    # transform the data
    data = preprocess_data(data)

    # Assuming 'expansive' column as target and others as features
    X = data.drop('expensive', axis=1).values
    y = data['expensive'].values

    # Perform train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check PCA
    n_components = 2
    custom_pca = CustomPCA(n_components=n_components)
    sklearn_pca = PCA(n_components=n_components)

    X_train_custom_pca = custom_pca.fit_transform(X_train)
    X_train_sklearn_pca = sklearn_pca.fit_transform(X_train)

    # Assuming the signs of the components are the same, check if they are close
    print("PCA similarity:", np.allclose(X_train_custom_pca, X_train_sklearn_pca))

    # Check Linear Regression
    custom_lr = LinearRegressionModel()
    sklearn_lr = SklearnLinearRegression()

    custom_lr.fit(X_train, y_train)
    sklearn_lr.fit(X_train, y_train)

    custom_lr_preds = custom_lr.predict(X_test)
    sklearn_lr_preds = sklearn_lr.predict(X_test)

    print("Linear Regression MSE:", mean_squared_error(y_test, custom_lr_preds))
    print("Sklearn Linear Regression MSE:", mean_squared_error(y_test, sklearn_lr_preds))

    # Check Logistic Regression
    custom_logreg = LogisticRegressionModel()
    sklearn_logreg = SklearnLogisticRegression()

    # Fit your model
    custom_logreg.fit(X_train, y_train)
    y_pred = custom_logreg.predict(X_test)
    y_pred_proba = custom_logreg.predict_proba(X_test)
    print("Your model's predicted probabilities: ", y_pred_proba[:5])

    # Fit sklearn's model
    sklearn_logreg.fit(X_train, y_train)
    y_pred_sk = sklearn_logreg.predict(X_test)
    y_pred_proba_sk = sklearn_logreg.predict_proba(X_test)
    print("Sklearn's model predicted probabilities: ", y_pred_proba_sk[:5, 1])  # Display probabilities for class 1
    print("Custom Logistic Regression report:")
    print(classification_report(y_test, y_pred))
    print("Sklearn Logistic Regression report:")
    print(classification_report(y_test, y_pred_sk))


def test_PCA_selections_Logistic(train_data, k_pca):
    # Load test dataset
    test_data = pd.read_csv('C:\\Users\\liorw\\Downloads\\test3.csv\\test.csv')
    id = test_data['id']
    train_data_scaled = preprocess_data(train_data)
    X_train = train_data_scaled.drop(columns=['expensive'])
    y_train = train_data_scaled['expensive']

    # First- PCA
    pca = CustomPCA(n_components=k_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test = preprocess_data(test_data)
    X_test_pca = pca.transform(X_test)

    # Second- feature selection
    # k_best = SelectKBest(score_func=f_regression, k=k_selection)
    # X_train_selected = k_best.fit_transform(X_train_pca, y_train)
    # X_test_selected = k_best.transform(X_test_pca)

    # Use your custom Logistic Regression class
    custom_lr = LogisticRegressionModel(learning_rate=0.1, n_iterations=3000)
    custom_lr.fit(X_train_pca, y_train.values)
    custom_probabilities = custom_lr.predict_proba(X_test_pca)
    print(f"Predicted probabilities (PCA+selection+Logistic): {custom_probabilities}")
    return id, custom_probabilities


train_data = pd.read_csv('C:\\Users\\liorw\\Downloads\\train3.csv\\train.csv')
id, prob = test_PCA_selections_Logistic(train_data, 42)
predictions_df = pd.DataFrame({'id': id, 'expensive': prob})
predictions_df.to_csv('predicted_probabilities3.csv', index=False)
comparing_VS_Sklearn(train_data)