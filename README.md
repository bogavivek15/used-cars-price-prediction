## Project README: Car Price Prediction

### Overview
This project focuses on predicting used car prices using a dataset containing various features of cars. The goal is to build a regression model that can accurately estimate the `selling_price` based on characteristics like `vehicle_age`, `km_driven`, `fuel_type`, `brand`, and more. The project explores both standard Ordinary Least Squares (OLS) Linear Regression and Ridge Regression to address potential overfitting.

### Dataset
The dataset used is `cardekho_dataset.csv`, which includes information on used cars. Key features include:
- `car_name`, `brand`, `model`: Car identification details.
- `vehicle_age`: Age of the vehicle.
- `km_driven`: Total kilometers driven.
- `seller_type`: Type of seller (Individual, Dealer, Trustmark Dealer).
- `fuel_type`: Fuel type (Petrol, Diesel, CNG, LPG, Electric).
- `transmission_type`: Transmission type (Manual, Automatic).
- `mileage`, `engine`, `max_power`, `seats`: Technical specifications.
- `selling_price`: The target variable to be predicted.

### Data Preprocessing
1.  **Feature Selection**: Dropped irrelevant columns such as `'Unnamed: 0'`, `'car_name'`, and `'model'`.
2.  **One-Hot Encoding**: Categorical features (`brand`, `seller_type`, `fuel_type`, `transmission_type`) were converted into numerical format using one-hot encoding, with `drop_first=True` to avoid multicollinearity.
3.  **Train-Test Split**: The data was split into training and testing sets (80% train, 20% test).
4.  **Feature Scaling**: Numerical features (`vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`) were scaled using `StandardScaler` to normalize their ranges, preventing features with larger values from dominating the model.

### Modeling
Two types of linear regression models were explored:

1.  **Ordinary Least Squares (OLS) Linear Regression**:
    - A baseline model was trained to understand the initial performance.
    - Evaluation metrics included Mean Absolute Error (MAE) and R² score on both training and test sets.
    - Overfitting was identified by a notable gap between train and test R² scores.

2.  **Ridge Regression**:
    - Implemented to address overfitting by adding an L2 regularization penalty to the loss function.
    - **Alpha Tuning**: Various `alpha` values (regularization strengths) were tested to find an optimal balance between bias and variance. A plot of R² and MAE against `alpha` values helped visualize the model's performance for different regularization strengths.
    - The coefficients of the OLS and Ridge models were compared to observe the shrinkage effect of regularization on feature importance.

### Results and Evaluation
- The OLS model showed signs of overfitting, with a higher R² on the training set compared to the test set.
- Ridge Regression successfully reduced the gap between training and test performance, indicating better generalization to unseen data.
- A manual prediction calculation for a sample car was demonstrated, confirming the consistency with `model.predict()` output.
- The stability of model performance was assessed by training the OLS model on multiple random train-test splits, highlighting the variability of evaluation metrics depending on the data split.

### Key Takeaways
- Data preprocessing, especially one-hot encoding and feature scaling, is crucial for preparing data for linear models.
- Regularization techniques like Ridge Regression are effective in mitigating overfitting and improving model generalization, particularly when dealing with many features or highly correlated predictors.
- Cross-validation or evaluating on multiple splits is important to get a robust understanding of a model's performance and its sensitivity to data partitioning.
