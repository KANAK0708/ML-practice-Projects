import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import seaborn as sns

df=pd.read_csv("used_car_dataset.csv")

# ==========================================================
print(" ######### USED CAR PRICE PREDICTION USING MACHINE LEARNING ##########")
# ==========================================================

print("############### Starting the pipeline  #################")
########## data preprocessing ###########

print(df.head())

print(df.shape)
print(df.columns)

print(df.info())

print(df.isnull().sum())

print(df.describe())
print(df.columns.to_list())

print("duplicate values",df.duplicated().sum())
print("duplicatedvalues dropped",df.drop_duplicates(inplace=True))





df['AskPrice'] = (
    df['AskPrice']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('₹', '', regex=False)
)

df['AskPrice'] = pd.to_numeric(df['AskPrice'], errors='coerce')

df['kmDriven'] = (
    df['kmDriven']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('km', '', regex=False)
)

df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')

num_cols = ['Age', 'kmDriven', 'AskPrice']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())




# BASIC EDA

sns.histplot(df['AskPrice'], kde=True)
plt.title("AskPrice Distribution")
plt.show()



sns.heatmap(
    df[['Age','kmDriven','AskPrice']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("correlation matrix")

plt.show()


# Feature Engineering
# Remove Multicollinearity
if 'Year' in df.columns:
    df.drop('Year', axis=1, inplace=True)

df.drop(
    ['PostedDate','AdditionInfo'],
    axis=1,
    inplace=True,
    errors="ignore"
)


# OUTLIER TREATMENT

#visualize

sns.boxplot(x=df['AskPrice'])
plt.show()

# IQR Method

Q1 = df['AskPrice'].quantile(0.25)
Q3 = df['AskPrice'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

df = df[
    (df['AskPrice'] >= lower) &
    (df['AskPrice'] <= upper)
]




# encoding categorical values

df=pd.get_dummies(df,columns=["Brand","model","Transmission","Owner","FuelType"],drop_first=True)

#feature selection

x=df.drop("AskPrice",axis=1)

y=df["AskPrice"]



X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# feature scaling-------------required for LINEAR AND LOGISTIC REGRESSION, SVM , KNN , NEURAL NETWORKS


scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Model Building

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)


# Prediction

y_pred=lr.predict(X_test_scaled)

# Model Evaluation

mae=mean_absolute_error(y_test,y_pred)
print(mae)


mse=mean_squared_error(y_test,y_pred)
print(mse)

r2=r2_score(y_test,y_pred)
print(r2)   


from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    lr,
    scaler.fit_transform(x),
    y,
    cv=5,
    scoring='r2'
)

print(scores.mean())




#### RANDOM FOREST #####


rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(X_train,y_train)

y_pred_rf=rf.predict(X_test)

print("\nRANDOM FOREST")

print("MAE :", mean_absolute_error(y_test, y_pred_rf))
print("MSE :", mean_squared_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2  :", r2_score(y_test, y_pred_rf))



rf_cv = cross_val_score(
    rf,
    x,
    y,
    cv=5,
    scoring='r2'
)

print("CV R2 Mean:", rf_cv.mean())

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_rf)
    ]
})

print("\nModel Comparison")
print(results)






####### using RIDGE AND LASSO REGRESSION to avoid overfitting in LINEAR REGRESSION ###############


from sklearn.linear_model import Ridge

ridge= Ridge(alpha=1.0)
ridge.fit(X_train_scaled,y_train)
y_pred_Ridge=ridge.predict(X_test_scaled)


## evaluation  to see the difference after applying  regularization techniques

print("evaluation metrics after applying ridge")

print("R2:", r2_score(y_test, y_pred_Ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_Ridge)))



from sklearn.linear_model import Lasso

lasso=Lasso(alpha=0.1)
lasso.fit(X_train_scaled,y_train)
y_pred_lasso=lasso.predict(X_test_scaled)


print("evaluation metrics after applying lasso")

print("R2:", r2_score(y_test, y_pred_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

### checking selected features 

lasso_coeff=pd.DataFrame({"feature":x.columns,"Coefficient":lasso.coef_})
print(lasso_coeff[lasso_coeff["Coefficient"] != 0])


########## HYPERPARAMETER TUNING ###########

from sklearn.model_selection import GridSearchCV

models = {"Linear Regression": (LinearRegression(),{}),

    "Ridge": (Ridge(),{"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=10000),{ "alpha": [0.001, 0.01, 0.1, 1, 10]}),
    "Random Forest": (RandomForestRegressor(random_state=42),{"n_estimators": [100, 200],"max_depth": [10, 20, None]})
}



results = []

for name, (model, params) in models.items():

    grid = GridSearchCV(estimator=model,param_grid=params,cv=5,scoring='r2',n_jobs=-1)

    # Use scaled data for linear models
    if name in ["Linear Regression", "Ridge", "Lasso"]:
        grid.fit(X_train_scaled, y_train)

        y_pred = grid.predict(X_test_scaled)

    else:
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

    results.append({"Model": name,"Best Params": grid.best_params_,"CV Score": grid.best_score_,"R2": r2_score(y_test, y_pred),"RMSE": np.sqrt(mean_squared_error(y_test, y_pred))})

    print(f"\n{name}")
    print("Best Params:", grid.best_params_)
    print("CV Score:", grid.best_score_)



results_df = pd.DataFrame(results)

print(results_df)



print("############# PIPELINE COMPLETED SUCCESSFULLY ###############")