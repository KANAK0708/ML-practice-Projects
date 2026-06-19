# Used Car Price Prediction

## 📌 Overview
This project focuses on building a robust Machine Learning pipeline to predict the selling price of used cars based on critical vehicular characteristics (e.g., mileage, age, fuel type, transmission, and brand power). Predicting secondary market car prices accurately helps dealerships optimize margins and provides consumers with fair evaluation metrics.

---

## 🛠️ Machine Learning Pipeline & Techniques
The data processing and feature engineering stages are structured to ensure high model generalization and prevent data leakage:
* **Data Cleaning & Missing Value Handling:** Imputed structural anomalies and treated missing data points using localized median/mode strategies to handle skewed distributions cleanly.
* **Exploratory Data Analysis (EDA):** Analyzed multi-collinearity issues and high-variance independent variables.
* **Feature Engineering & Outlier Treatment:** Engineered age-based features from vintage data. Applied Interquartile Range (IQR) capping to mitigate the leverage of extreme pricing anomalies without losing valuable training instances.
* **Categorical Encoding & Scaling:** Utilized One-Hot Encoding for nominal variables. Scaled numerical features using standard normalization strategies to optimize gradient descent performance for linear models.

---

## 🔬 Algorithm Selection: Why These Models?

To solve this regression challenge, a spectrum of algorithms was tested, moving from highly interpretable baseline statistical models to complex ensemble architectures:

### 1. Linear Regression
* **Why:** Serves as our foundational baseline. It offers absolute interpretability, showing the direct, proportional impact of every single feature (like mileage or vehicle age) on the target price.

### 2. Regularized Linear Models (Ridge & Lasso)
* **Why:** Baseline linear models are highly susceptible to overfitting when handling multi-collinearity or one-hot encoded categorical vectors. 
  * **Ridge Regression ($L_2$ Regularization):** Shrinks coefficients uniformly to minimize the impact of collinear features.
  * **Lasso Regression ($L_1$ Regularization):** Enforces sparsity by driving less important feature coefficients completely to zero, effectively acting as an embedded feature selection layer.

### 3. Random Forest Regressor
* **Why:** Used car pricing data often contains complex, non-linear relationships and interactions (e.g., the combined effect of an old luxury brand vs. a new economy brand). Random Forest handles non-linear decision boundaries implicitly and remains highly resilient to outliers.

---

## 🔄 The Role of Cross-Validation

### Why Use Cross-Validation?
Relying on a single train-test split introduces **sample bias**, meaning the model's performance score could heavily depend on the random luck of how the data was split. Cross-validation provides a deterministic, reliable estimate of how the model will perform on completely unseen real-world data.

### Where Should It Be Used?
* **Model Evaluation:** K-Fold Cross-Validation (typically $K=5$ or $K=10$) is executed exclusively on the training subset to gauge model stability.
* **Hyperparameter Tuning:** It is integrated within the search grids to ensure that hyperparameter choices are selected based on their generalized performance across multiple validation folds, rather than overfitting to a single static validation set.

> ⚠️ **Crucial Rule:** Feature scaling and preprocessing pipelines must be wrapped *inside* each cross-validation loop (e.g., using scikit-learn's `Pipeline`) to completely avoid **data leakage** from the validation slices into training steps.

---

## ⚙️ Hyperparameter Tuning Effects

Hyperparameter optimization was conducted via grid/random search strategies to transition models from underfit baselines to optimal generalization zones.

| Model | Key Parameters Tuned | Observable Optimization Effects |
| :--- | :--- | :--- |
| **Ridge / Lasso** | Alpha ($\alpha$) Regularization Strength | Tuning $\alpha$ strikes the right bias-variance tradeoff. Setting it too high underfits the model, while setting it too low mimics standard overfitted Linear Regression. |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | Limiting `max_depth` and raising `min_samples_split` restricts individual tree memorization, dramatically closing the gap between training error and validation error. |

---

## 📊 Evaluation Metrics
The final models are scrutinized using complementary performance metrics:
* **MAE (Mean Absolute Error):** Indicates the average absolute monetary error expected per prediction.
* **MSE & RMSE (Root Mean Squared Error):** penalizes larger prediction errors heavily, identifying if the model makes catastrophic pricing errors on rare vehicles.
* **$R^2$ Score (Coefficient of Determination):** Outlines the percentage of variance in used car prices explained by our feature space.