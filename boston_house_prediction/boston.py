import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_openml


from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")

b = fetch_openml(name='boston', version=1, as_frame=True)
print(b)
df = b.data.copy()
df['MEDV'] = b.target


# df=pd.DataFrame(data=b.data,columns=b.feature_names)
# print(df)

print(df.head())
print(df.shape)

df.info()



print(df.describe())

print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates removed")


plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

corr_target=df.corr()["MEDV"].sort_values(ascending=False)
print(corr_target)

sns.histplot(df["MEDV"],kde=True)
plt.show()


plt.figure(figsize=(15,8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

x=df.drop("MEDV",axis=1)
y=df["MEDV"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#### for decision tree scaling is not needed but for SVR it is mandatory

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)

# DECISION TREE

dt_model=DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train,y_train)

dt_pred=dt_model.predict(x_test)


print("MAE :",mean_absolute_error(y_test,dt_pred))

print("MSE :",mean_squared_error(y_test,dt_pred))

print("RMSE :",np.sqrt(mean_squared_error(y_test,dt_pred)))

print("R2 :",r2_score(y_test,dt_pred))

# svr 
from sklearn.svm import SVR

svr_model = SVR(kernel='rbf')

svr_model.fit(x_train_scaled,y_train)

svr_pred=svr_model.predict(x_test_scaled)


print("MAE :",mean_absolute_error(y_test,svr_pred))

print("MSE :",mean_squared_error(y_test,svr_pred))

print("RMSE :",np.sqrt(mean_squared_error(y_test,svr_pred)))

print("R2 :",r2_score(y_test,svr_pred))




### hyperparameter tuning

DecisionTreeRegressor(max_depth=5,min_samples_split=10,min_samples_leaf=5,random_state=42)

max_depth=[3,5,7,10,None]

SVR(kernel='rbf',C=100,gamma="scale",epsilon=0.1)


# VISUALIZATION (ACTUAL VS PREDICTED)

#decision tree

plt.scatter(y_test,dt_pred)

plt.xlabel("actual")
plt.ylabel("predicted")

plt.title("decision tree")

plt.show()


# svr 
plt.scatter(y_test,svr_pred)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("SVR")

plt.show()

# model comparison


comparison = pd.DataFrame({
    'Model':['Decision Tree','SVR'],
    'R2 Score':[
        r2_score(y_test,dt_pred),
        r2_score(y_test,svr_pred)
    ]
})

print(comparison)
































































