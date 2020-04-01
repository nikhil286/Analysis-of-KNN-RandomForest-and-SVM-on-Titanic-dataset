#import required libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#extracting dataset 
df = pd.read_csv("Titanic.csv")

#Columns Name and PassengerId have no significance in prediction. So drop them.
df.drop(["name", "body"], axis=1, inplace=True)

#Replacing missing values of Age with the median Age. 
#You can also try replacing Age with mean.
df["age"]=df["age"].fillna(df["age"].median())

#Repalce other NaN values with 0.
df.fillna(0, inplace=True)

#function to convert textual data to numeric data.
def text_to_numeric(df):
    col_arr=df.columns.values
    
    for i in col_arr:
        ref_dict={}
        def convert_to_int(val):
            return ref_dict[val]
        
        if df[i].dtype != np.int64 and df[i].dtype != np.float64:
            value_arr=df[i].values.tolist()
            unique_ele=set(value_arr)
            c=0
            for element in unique_ele:
                if element not in ref_dict:
                    ref_dict[element]=c
                    c+=1
                    
            df[i]=list(map(convert_to_int, df[i]))
    return df

df=text_to_numeric(df)

print("After processing dataset \n\n:", df.head())

X=df.values[:,1:10]
y=df.values[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

#Scaling data
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.fit_transform(X_test)

#Using KNN
model1=neighbors.KNeighborsClassifier()
model1.fit(X_train,y_train)

print("\n Accuracy achieved using KNN ::", model1.score(X_test,y_test))

#Using Random Forest
model2=RandomForestClassifier()
model2.fit(X_train,y_train)

print(" Accuracy achieved using RandomForestClassifier ::", model2.score(X_test, y_test))

#Using SVM
model3=svm.SVC(kernel = "linear")
model3.fit(X_train,y_train)

print(" Accuracy achieved using SVM ::", model3.score(X_test, y_test))
