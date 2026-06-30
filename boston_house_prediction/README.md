# 🏠 Boston House Price Prediction using Machine Learning

A Machine Learning regression project that predicts house prices using the **Boston Housing Dataset**. This project compares the performance of **Decision Tree Regressor** and **Support Vector Regression (SVR)** by evaluating them on multiple regression metrics and visualizing their predictions.

---

## 📌 Project Overview

The objective of this project is to build regression models capable of predicting the median value of owner-occupied homes based on various housing and demographic features.

The project includes:

- Data loading from OpenML
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature correlation analysis
- Model training
- Model evaluation
- Model comparison
- Visualization of predictions

---

## 📂 Dataset

- **Dataset:** Boston Housing Dataset
- **Source:** OpenML
- **Target Variable:** `MEDV` (Median value of owner-occupied homes)

The dataset contains several housing-related attributes such as:

- Crime Rate
- Residential Land Zoned
- Industrial Area
- Number of Rooms
- Property Tax Rate
- Pupil-Teacher Ratio
- Accessibility to Highways
- Nitric Oxide Concentration
- Percentage of Lower Status Population
- and more.

---

## ⚙️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📚 Machine Learning Algorithms

### 1️⃣ Decision Tree Regressor

- Non-linear regression algorithm
- Does not require feature scaling
- Easy to interpret
- Captures complex relationships in the data

### 2️⃣ Support Vector Regression (SVR)

- Kernel: RBF
- Requires feature scaling
- Effective for non-linear regression problems
- Uses Support Vector Machines for prediction

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset information
- Statistical summary
- Missing value detection
- Duplicate value removal
- Correlation Heatmap
- Target Variable Distribution
- Outlier Detection using Boxplots

---

## 🔄 Data Preprocessing

- Duplicate removal
- Train-Test Split (80:20)
- Feature Scaling using **StandardScaler** (for SVR)

> Note:
> Decision Tree Regressor does not require feature scaling, whereas SVR performs significantly better on scaled data.

---

## 📈 Model Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📉 Visualizations

The project generates the following visualizations:

- Correlation Heatmap
- Target Variable Distribution
- Boxplots for Outlier Detection
- Actual vs Predicted (Decision Tree)
- Actual vs Predicted (SVR)
- Model Performance Comparison

---

## 🛠️ Hyperparameter Tuning

The project also demonstrates tuning parameters for better model performance.

### Decision Tree

- max_depth
- min_samples_split
- min_samples_leaf

### SVR

- Kernel
- C
- Gamma
- Epsilon

---

## 📁 Project Structure

```
Boston_House_Prediction/
│
├── boston.py
├── README.md
├── requirements.txt
└── images/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/KANAK0708/ML-practice-Projects.git
```

Move into the project directory

```bash
cd ML-practice-Projects
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python boston.py
```

---

## 📋 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Or install them manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 📌 Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling (SVR)
      │
      ▼
Train Decision Tree
      │
      ▼
Train SVR
      │
      ▼
Evaluate Models
      │
      ▼
Compare Performance
      │
      ▼
Visualize Predictions
```

---

## 📊 Performance Metrics

The project compares both models using:

- MAE
- MSE
- RMSE
- R² Score

The comparison helps determine which regression model performs better for the Boston Housing dataset.

---

## 🔮 Future Improvements

- Perform GridSearchCV for hyperparameter optimization
- Add Random Forest Regressor
- Add XGBoost Regressor
- Implement Cross Validation
- Save trained models using Joblib/Pickle
- Deploy using Streamlit or Flask
- Create an interactive dashboard

---


