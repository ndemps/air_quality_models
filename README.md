# Air Quality Data Analytics

This repository focuses on analyzing air quality data to predict the Air Quality Index (AQI) bucket based on various air pollutant levels. The primary objective is to understand the significance of different pollutants in determining air quality and to build predictive models that can accurately classify the AQI bucket.

## Dataset
The dataset used is named [city_day.csv](https://github.com/ndemps/air_quality_models/blob/main/Data/city_day.csv), which contains daily air quality data, including levels of various pollutants and the corresponding AQI values.

## Key Features:
1. **Data Preprocessing**: Comprehensive data cleaning, including handling missing values and median imputation for numerical columns.
2. **Feature Importance Analysis**: Utilizes RandomForestClassifier to visualize the significance of different features in predicting AQI.
3. **Model Training and Evaluation**: Implements RandomForestClassifier and XGBoost Classifier for prediction. Includes hyperparameter tuning for optimization and evaluates model performance on validation and test datasets.
4. **Stratified Sampling**: Ensures a balanced representation of AQI buckets in training, validation, and test datasets.

## Code Files:
1. **Initial Data Analytics**: [Initial_Data_Analytics_Air-Quality.py](https://github.com/ndemps/air_quality_models/blob/ada712e25ce26575d81ca764d486af1d50b3307e/Initial_Data_Analytics_Air-Quality.py)
2. **Refactored Data Analytics**: [Data-Analytics_Air-Quality_Refactored.py](https://github.com/ndemps/air_quality_models/blob/ada712e25ce26575d81ca764d486af1d50b3307e/Data-Analytics_Air-Quality_Refactored.py)
3. **Fine-tuned Data Analytics**: [Data-Analytics_Air-Quality_Fine-tuned.py](https://github.com/ndemps/air_quality_models/blob/ada712e25ce26575d81ca764d486af1d50b3307e/Data-Analytics_Air-Quality_Fine-tuned.py)

## Fine-tuned Data Analytics - Attempts at Optimizing the RFC Model:
- **XGBoost Revision**: [XGBoost](https://github.com/ndemps/air_quality_models/blob/3cba515d4777e5ce8f4c9d40dcb82f65a0dfd7b4/Data-Analytics_Air-Quality_Fine-tuned.py)
- **Stratification Revision**: [Stratification](https://github.com/ndemps/air_quality_models/blob/ada712e25ce26575d81ca764d486af1d50b3307e/Data-Analytics_Air-Quality_Fine-tuned.py)

## Usage:
To run the analysis, clone the repository, ensure you have the required libraries installed, and execute the desired `.py` file.

