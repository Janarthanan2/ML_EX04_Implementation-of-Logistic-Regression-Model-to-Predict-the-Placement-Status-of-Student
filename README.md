# Ex04 - Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Janarthanan V K
RegisterNumber:  212222230051
*/
```
```python
import pandas as pd
df = pd.read_csv("Placement_Data.csv")
df.head()
df.isnull().sum()
df1 = df.copy()
df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x = df1.iloc[:, : -1]
y = df1["status"]

from sklearn.model_selection import train_test_split
x_train, x_test ,y_train ,y_test = train_test_split(x, y, test_size = 0.2, random_state=34)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = "liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test ,y_pred)

print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)

model.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

### Data Information:
<img src="https://github.com/Janarthanan2/ML_EX04_Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393515/13b68f50-91f8-45fc-a777-fb8c8f699447" width=60%>

### Data Status:
<img src="https://github.com/Janarthanan2/ML_EX04_Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393515/abb5862a-5c56-4ea5-80a3-b6930e897ac9" height="250">

### Accuracy Value:
<img src="https://github.com/Janarthanan2/ML_EX04_Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393515/a4de98ef-0bf8-4d1f-80b3-e7ee3ce28e40">

### Confusion Value:
<img src="https://github.com/Janarthanan2/ML_EX04_Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393515/899efffa-4a88-4e5b-bc23-51444c8cf504">

### Classification Report:
<img src="https://github.com/Janarthanan2/ML_EX04_Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393515/71906223-fc16-496d-9740-691bcf3e1ee1" width=30%>

### Prediction of LR:
<img src="https://github.com/Janarthanan2/ML_EX04_Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393515/f45651db-212c-4b9e-bf1f-7928914b8ee0">

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
