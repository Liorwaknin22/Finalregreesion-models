import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime


class LogisticRegression:
    """
    A class which implements logistic regression model with gradient descent.
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
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
        # Clip the input to avoid overflow
        x = np.clip(x, -500, 500)
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
    data.drop(['id','host_id', 'has_availability', 'first_review', 'last_review'], axis=1, inplace=True)
    # Fill missing values
    data['host_response_time'] = data['host_response_time'].fillna('No Response')
    data['host_response_rate'] = data['host_response_rate'].fillna('0%')
    data['host_acceptance_rate'] = data['host_acceptance_rate'].fillna('0%')
    data['host_is_superhost'].fillna(data['host_is_superhost'].mode()[0], inplace=True)
    data['bathrooms_text'].fillna(data['bathrooms_text'].mode()[0], inplace=True)
    data['bedrooms'].fillna(data['bedrooms'].median(), inplace=True)
    data['beds'].fillna(data['beds'].median(), inplace=True)
    num_records = len(data)
    print(f"Number of records in the data: {num_records}")

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

    # Extract three unique values from 'host_verifications'
    unique_verifications = data['host_verifications'].apply(eval).explode().unique()[:3]

    # Create dummy variables for 'host_verifications' using the unique values
    host_verifications_dummies = pd.DataFrame()
    for verification in unique_verifications:
        column_name = 'host_verification_' + verification
        host_verifications_dummies[column_name] = data['host_verifications'].apply(
            lambda row: verification in row).astype(int)

    # Concatenate the dummy variables at the beginning of the DataFrame
    data = pd.concat([host_verifications_dummies, data], axis=1)
    data.drop('host_verifications', axis=1, inplace=True)
    data = data.select_dtypes(exclude=['object'])

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
    custom_lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    custom_lr.fit(X_train.values, y_train.values)
    custom_probabilities = custom_lr.predict_proba(X_test.values)
    print(f"Predicted probabilities (Custom Logistic Regression): {custom_probabilities}")
    return id, custom_probabilities


data = pd.read_csv('C:\\Users\\liorw\\Downloads\\train3.csv\\train.csv')
id, prob = testLogisticRegression3(data)
predictions_df = pd.DataFrame({'id': id, 'expensive': prob})
predictions_df.to_csv('predicted_probabilities.csv', index=False)
