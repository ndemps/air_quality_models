from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import xgboost as xgb
df = pd.read_csv('city_day.csv')
df.head()

# Calculate the percentage of missing values for each column
missing_percentage = df.isnull().sum() * 100 / len(df)
print('Missing Percentage:', missing_percentage)

# Median Imputation for Numerical Columns
numerical_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Create a mapping based on existing data
aqi_bucket_mapping = {
    'Good': (0, 50),
    'Satisfactory': (51, 100),
    'Moderate': (101, 200),
    'Poor': (201, 300),
    'Very Poor': (301, 400),
    'Severe': (401, 500)
}
def map_aqi_to_bucket(aqi_value):
    for bucket, (lower, upper) in aqi_bucket_mapping.items():
        if lower <= aqi_value <= upper:
            return bucket
    return np.nan  # return NaN if the AQI value is out of known ranges

# Apply the mapping function to fill 'AQI_Bucket'
df['AQI_Bucket'].fillna(df['AQI'].apply(map_aqi_to_bucket), inplace=True)

# Calculate the percentage of missing values for each column after cleaning
missing_percentage_after = df.isnull().sum() * 100 / len(df)
print('Missing Percentage After Cleaning:', missing_percentage_after)

# Function for calculating feature importances
def feature_importance(X_train, y_train):
    # Initialize the model
    rf = RandomForestClassifier()
    # Fit the model
    rf.fit(X_train, y_train)
    # Get feature importances
    importances = rf.feature_importances_
    # Create a DataFrame for visualization
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # Sort and visualize
    feature_importances.sort_values(by='Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance')
    plt.title("Feature Importances")
    plt.show()

# Prepare the features and target variable
X = df[['PM2.5', 'PM10', 'NO2', 'CO']]
y = df['AQI_Bucket']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Data Splitting with stratification
X_temp, X_test, y_temp_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_temp, y_temp_encoded, test_size=0.25, random_state=42, stratify=y_temp_encoded)

# Use train_test_split to perform stratified sampling
_, df_sample = train_test_split(df, test_size=0.1, random_state=42, stratify=df['AQI_Bucket'])



# Create features and target from the sample
X_sample = df_sample[['PM2.5', 'PM10', 'NO2', 'CO']]
y_sample = df_sample['AQI_Bucket']

# Split the sampled data with stratification
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

# Define the reduced hyperparameter grid for RandomForestClassifier
rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

# Apply RandomizedSearchCV on the RandomForestClassifier
rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
rf_random_search.fit(X_train_sample, y_train_sample)
best_rf_params = rf_random_search.best_params_

feature_names = X.columns

# Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Call the feature importance function
feature_importance(X_train, y_train)

# Train an RFC with the best hyperparameters found
optimized_rf = RandomForestClassifier(**best_rf_params, random_state=42)
optimized_rf.fit(X_train_sample, y_train_sample)

# Use RFECV to find the optimal number of features
selector = RFECV(estimator=optimized_rf, step=1, cv=3, scoring='accuracy', n_jobs=-1)
selector = selector.fit(X_train_sample, y_train_sample)
optimal_features = X_sample.columns[selector.support_]

# Train the RFC again on the optimal feature set
X_train_optimal = X_train_sample[optimal_features]
X_test_optimal = X_test_sample[optimal_features]
optimized_rf.fit(X_train_optimal, y_train_sample)

# Function for model evaluation
def evaluate_model(model, X_val, y_val, X_test, y_test):
    y_pred_val = model.predict(X_val)
    print("Validation Results:")
    print(classification_report(y_val, y_pred_val, zero_division=1))
    y_pred_test = model.predict(X_test)
    print("Test Results:")
    print(classification_report(y_test, y_pred_test, zero_division=1))

y_pred = optimized_rf.predict(X_test_optimal)
accuracy = accuracy_score(y_test_sample, y_pred)

print(f"Accuracy after optimization: {accuracy}")
# Evaluate the optimized RFC on validation and test sets
evaluate_model(optimized_rf, X_val, y_val, X_test, y_test)

# Print a separator for clarity
print("-" * 50)
print("XGBoost Classifier Results:")

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)

# Fit the model to the training data
xgb_classifier.fit(X_train, y_train_encoded)

# Decoding the predictions for evaluation
y_pred_val = label_encoder.inverse_transform(xgb_classifier.predict(X_val))
y_pred_test = label_encoder.inverse_transform(xgb_classifier.predict(X_test))
print("Validation Results:")
print(classification_report(y_val, y_pred_val, zero_division=1))
print("Test Results:")
print(classification_report(y_test, y_pred_test, zero_division=1))
