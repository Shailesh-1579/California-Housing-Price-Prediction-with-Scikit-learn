import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1. load the dataset
data_one=pd.read_csv("housing.csv")

#2. Stratified test set
data_one["income_category"]=pd.cut(data_one["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
split=StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=20)
for train_index,test_index in split.split(data_one,data_one["income_category"]):
    training_set=data_one.loc[train_index].drop("income_category",axis=1)
    testing_set=data_one.loc[test_index].drop("income_category",axis=1)

#we will use the copy of the data
data_one=training_set.copy()

#3.seperate features and labels
data_one_labels=data_one["median_house_value"].copy()
data_one=data_one.drop("median_house_value",axis=1)



#4. seperate categorical and numrical columns
num_attribute=data_one.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribute=["ocean_proximity"]

#5.make pipeline 

# for numerical columns
num_pipeline=Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

#for categorical column

cat_pipeline=Pipeline([
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

#full pipeline

full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribute),
    ("cat",cat_pipeline,cat_attribute)
])

#6. data transformation
data_final=full_pipeline.fit_transform(data_one)

print(data_final)


#7. train the model

#linear regression model
lin_reg= LinearRegression()
lin_reg.fit(data_final,data_one_labels)
linpreds=lin_reg.predict(data_final)
#lin_rmse=root_mean_squared_error(data_one_labels,linpreds)
lin_rmse= -cross_val_score(lin_reg,data_final,data_one_labels,scoring="neg_root_mean_squared_error",cv=10)
#print(f"The root mean sqaured error for the linear regression model is:{lin_rmse}")  #68669.60634944153
print(pd.Series(lin_rmse).describe())


#decision tree model
dec_reg= DecisionTreeRegressor()
dec_reg.fit(data_final,data_one_labels)
decpreds=dec_reg.predict(data_final)
#dec_rmse=root_mean_squared_error(data_one_labels,linpreds)
dec_rmse= -cross_val_score(dec_reg,data_final,data_one_labels,scoring="neg_root_mean_squared_error",cv=10)
#print(f"The root mean sqaured error for the decision tree model is:{dec_rmse}")
print(pd.Series(dec_rmse).describe())


#random forest model
ranfor_reg= RandomForestRegressor()
ranfor_reg.fit(data_final,data_one_labels)
ranforpreds=ranfor_reg.predict(data_final)
#ranfor_rmse=root_mean_squared_error(data_one_labels,ranforpreds)
ranfor_rmse= -cross_val_score(ranfor_reg,data_final,data_one_labels,scoring="neg_root_mean_squared_error",cv=10)
#print(f"The root mean sqaured error for the random forest model is:{ranfor_rmse}")
print(pd.Series(ranfor_rmse).describe())