#Import the library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

#Load the csv into a dataframe
df = pd.read_csv('Fuel_Consumption_2000-2022.csv')
df.head(5)

df.columns #12 colums: 'YEAR', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS', 'TRANSMISSION', 'FUEL', 'FUEL CONSUMPTION', 'HWY (L/100 km)','COMB (L/100 km)', 'COMB (mpg)', 'EMISSIONS'

#Check data types and the information of this data frame
df.dtypes
df.info()

#Check duplicated value
df.duplicated().sum() #Output: 1 duplicated row
df.drop_duplicates(inplace=True) #Remove duplicated row

#Check missing values
df.isnull().sum() #So, no missing values here

#Data Selection
#We have object(5), so let's convert categorical columns into numerical ones.
#If we remove all categorical columns, the model will be constructed maybe not meaningful or we'll miss some important informations.
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

#Because the 'EMISSIONS' is estimated by 'FUEL CONSUMPTION', so we cannot use 'EMISSION' to predict it.
df = df.drop("EMISSIONS", axis= 1)
df_encoded = df

df_encoded["MAKE"] = encoder.fit_transform(df["MAKE"])
df_encoded["MODEL"] = encoder.fit_transform(df["MODEL"])
df_encoded["VEHICLE CLASS"] = encoder.fit_transform(df["VEHICLE CLASS"])
df_encoded["TRANSMISSION"] = encoder.fit_transform(df["TRANSMISSION"])
df_encoded["FUEL"] = encoder.fit_transform(df["FUEL"])

#Extract the feature columns and target columns before modelling.
features = df_encoded.drop("FUEL CONSUMPTION", axis=1)
target = df_encoded["FUEL CONSUMPTION"]
df

sns.boxplot(df['ENGINE SIZE'])
plt.show()

#Descriptive Statistics
df.describe() #Summary of this dataframe
corr_matrix = df.corr(numeric_only=True) #Check correlation between all variables
print(corr_matrix)
corr_matrix['FUEL CONSUMPTION']
sns.heatmap(corr_matrix, cmap='YlGnBu')
plt.show()

#Split data into training set and testing set (80:20)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Standard scaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
print('Linear Regression:')
print('MSE:', mean_squared_error(y_test, lr_pred))
print('MAE:', mean_absolute_error(y_test, lr_pred))
print('R-squared:', r2_score(y_test, lr_pred))

#XGBoost
import xgboost as xgb
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# Define the hyperparameters for the XGBoost model
params = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 0
}

# Train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict the target values for the testing set
xgb_pred = model.predict(dtest)

# Calculate the mean squared error between the actual and predicted target values
print('MSE:', mean_squared_error(y_test, xgb_pred))
print('MAE:', mean_absolute_error(y_test, xgb_pred))
print('R-squared:', r2_score(y_test, xgb_pred))

#GBM
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor().fit(X_train,y_train)
gbm_pred = gbr.predict(X_test)
print('MSE:', mean_squared_error(y_test, gbm_pred))
print('MAE:', mean_absolute_error(y_test, gbm_pred))
print('R-squared:', r2_score(y_test, gbm_pred))


#Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor().fit(X_train,y_train)
dt_pred = dt.predict(X_test)
print('MSE:', mean_squared_error(y_test, dt_pred))
print('MAE:', mean_absolute_error(y_test, dt_pred))
print('R-squared:', r2_score(y_test, dt_pred))

#CatBoost
from catboost import CatBoostRegressor
cat = CatBoostRegressor(verbose=False).fit(X_train,y_train)
cat_pred = cat.predict(X_test)
print('MSE:', mean_squared_error(y_test, cat_pred))
print('MAE:', mean_absolute_error(y_test, cat_pred))
print('R-squared:', r2_score(y_test, cat_pred))