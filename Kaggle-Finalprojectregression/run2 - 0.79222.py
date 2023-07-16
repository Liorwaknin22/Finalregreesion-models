import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from sklearn.impute import KNNImputer


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
    data.drop(['id', 'host_id', 'property_type', 'has_availability', 'first_review', 'last_review', 'license'], axis=1,
              inplace=True)
    # Fill missing values:
    response_time_to_score = {"within an hour": 4, "within a few hours": 3, "within a day": 2,
                              "a few days or more": 1, }

    def map_score(response_time):
        if pd.isnull(response_time) or response_time.strip() == '':
            return 2.5
        else:
            return response_time_to_score.get(response_time, 2.5)

    data['host_response_time_score'] = data['host_response_time'].apply(map_score)

    data['host_response_rate'] = data['host_response_rate'].str.rstrip('%').astype('float')
    # Fill NA/NaN values with the median
    median_val = data['host_response_rate'].median()
    data['host_response_rate'] = data['host_response_rate'].fillna(median_val)

    data['host_acceptance_rate'] = data['host_acceptance_rate'].str.rstrip('%').astype('float')
    # Fill NA/NaN values with the median
    median_val = data['host_acceptance_rate'].median()
    data['host_acceptance_rate'] = data['host_acceptance_rate'].fillna(median_val)

    data['host_is_superhost'].fillna(data['host_is_superhost'].mode()[0], inplace=True)
    data['host_has_profile_pic'].fillna(data['host_has_profile_pic'].mode()[0], inplace=True)
    data['host_identity_verified'].fillna('f', inplace=True)

    def map_bathrooms_text_score(bathrooms_text):
        if pd.isnull(bathrooms_text) or bathrooms_text.strip() == '':
            return 1.5  # default value for NA or empty string

        # Extract the number of baths
        number = re.findall("\d+\.\d+|\d+", bathrooms_text)
        if number:
            number_score = float(number[0])
        else:
            number_score = 0  # default value for no number information

        # Check the type of baths
        if "shared" in bathrooms_text:
            type_score = 1
        elif "private" in bathrooms_text:
            type_score = 2
        else:
            type_score = 1.5  # default value for no type information

        return number_score * type_score

    data['bathrooms_text_score'] = data['bathrooms_text'].apply(map_bathrooms_text_score)
    data.drop('bathrooms_text', axis=1, inplace=True)

    data['host_listings_count'].fillna(data['host_listings_count'].median(), inplace=True)
    data['latitude'].fillna(data['latitude'].median(), inplace=True)
    data['longitude'].fillna(data['longitude'].median(), inplace=True)
    data['accommodates'].fillna(data['accommodates'].median(), inplace=True)

    data['availability_30'].fillna(data['availability_30'].median(), inplace=True)
    data['availability_60'].fillna(data['availability_60'].median(), inplace=True)
    data['availability_90'].fillna(data['availability_90'].median(), inplace=True)
    data['availability_365'].fillna(data['availability_365'].median(), inplace=True)

    data['number_of_reviews'].fillna(data['number_of_reviews'].median(), inplace=True)
    data['number_of_reviews_ltm'].fillna(data['number_of_reviews_ltm'].median(), inplace=True)
    data['number_of_reviews_l30d'].fillna(data['number_of_reviews_l30d'].median(), inplace=True)

    data['review_scores_rating'].fillna(data['review_scores_rating'].median(), inplace=True)
    data['review_scores_accuracy'].fillna(data['review_scores_accuracy'].median(), inplace=True)
    data['review_scores_cleanliness'].fillna(data['review_scores_cleanliness'].median(), inplace=True)
    data['review_scores_checkin'].fillna(data['review_scores_checkin'].median(), inplace=True)
    data['review_scores_communication'].fillna(data['review_scores_communication'].median(), inplace=True)
    data['review_scores_location'].fillna(data['review_scores_location'].median(), inplace=True)
    data['review_scores_value'].fillna(data['review_scores_value'].median(), inplace=True)
    data['calculated_host_listings_count_entire_homes'].fillna(
        data['calculated_host_listings_count_entire_homes'].median(), inplace=True)
    data['calculated_host_listings_count_private_rooms'].fillna(
        data['calculated_host_listings_count_private_rooms'].median(), inplace=True)
    data['calculated_host_listings_count_shared_rooms'].fillna(
        data['calculated_host_listings_count_shared_rooms'].median(), inplace=True)

    data['instant_bookable'].fillna(data['instant_bookable'].mode()[0], inplace=True)

    data['host_total_listings_count'].fillna(data['host_total_listings_count'].median(), inplace=True)
    data['bedrooms'].fillna(0, inplace=True)
    data['beds'].fillna(0, inplace=True)
    data['calculated_host_listings_count'].fillna(0, inplace=True)
    data['reviews_per_month'].fillna(0, inplace=True)

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
    nights_max = ['maximum_nights', 'maximum_minimum_nights', 'maximum_maximum_nights', 'maximum_nights_avg_ntm']
    for column in nights_max:
        data[column].fillna(data[column].median(), inplace=True)

    nights_min = ['minimum_nights', 'minimum_minimum_nights', 'minimum_maximum_nights', 'minimum_nights_avg_ntm']
    for column in nights_min:
        data[column].fillna(0, inplace=True)

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
        data[col] = data[col].fillna(0.5)

    # Apply MinMaxScaler to date columns
    scaler = MinMaxScaler()
    data[date_columns] = scaler.fit_transform(data[date_columns])

    # Convert 'amenities' to the sum of the list
    data['amenities'] = data['amenities'].fillna('[]').apply(lambda x: len(eval(x)))

    room_type_to_score = {'Entire home/apt': 3, 'Private room': 2, 'Shared room': 1, 'Hotel room': 2, }

    def map_room_type_score(room_type):
        if pd.isnull(room_type) or room_type.strip() == '':
            return 0.5  # or any default value you'd like to assign
        else:
            return room_type_to_score.get(room_type, 2)

    data['room_type'] = data['room_type'].apply(map_room_type_score)
    data.drop('room_type', axis=1, inplace=True)

    # Extract three unique values from 'host_verifications'
    data['host_verifications'] = data['host_verifications'].fillna('[]')
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

    # # Detect anomalies in the numerical columns
    # numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    # anomalies = {}
    # for column in numerical_columns:
    #     mean = data[column].mean()
    #     std = data[column].std()
    #     lower_bound = mean - 3 * std
    #     upper_bound = mean + 3 * std
    #     anomalies[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    #
    # # Replace anomalies with the median value
    # for column, anomaly_df in anomalies.items():
    #     if not anomaly_df.empty:
    #         median = data[column].median()
    #         mean = data[column].mean()
    #         std = data[column].std()
    #         lower_bound = mean - 3 * std
    #         upper_bound = mean + 3 * std
    #         data.loc[(data[column] < lower_bound) | (data[column] > upper_bound), column] = median

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
predictions_df.to_csv('predicted_probabilities_2.csv', index=False)
