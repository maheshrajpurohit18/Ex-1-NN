<H3>ENTER YOUR NAME: MAHESH RAJ PUROHIT J</H3>
<H3>ENTER YOUR REGISTER NO: 212222240058</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 23.08.2024 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```
```python
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
```
```python
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
```
```python
df.isnull().sum()
```
```python
df.duplicated()
```
```python
df.describe()
```
```python
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
```
```python
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
```
```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:
#### DATASET:

![image](https://github.com/user-attachments/assets/4d7c82ad-5739-4fe5-bb7a-ed69dfac38fe)

#### DROPPING THE UNWANTED DATASET:

![image](https://github.com/user-attachments/assets/515792e2-28c6-4bf6-ac4b-69adb3b4c35f)

#### CHECKING NULL VALUES:

![image](https://github.com/user-attachments/assets/8d93499f-2a83-40dd-aa80-f7b8d3fe4672)

#### CHECKING FOR DUPLICATION:

![image](https://github.com/user-attachments/assets/27d2e1b4-2263-42ab-a7fc-e48763362697)

#### DESCRIBING THE DATASET:

![image](https://github.com/user-attachments/assets/55458a45-1a25-4175-90d8-51d1570db1de)

#### SCALING THE DATASET:

![image](https://github.com/user-attachments/assets/3384a557-4b4c-4a5d-936c-4b5547f6dd42)

#### X FEATURES:

![image](https://github.com/user-attachments/assets/360feb21-be30-4bc5-b1d4-7afe6d2687c4)

#### Y FEATURES:

![image](https://github.com/user-attachments/assets/333fdee7-d153-47a9-93c0-189f05f6c7d7)


#### SPLITTING THE TRAINING AND TESTING DATASET:

![image](https://github.com/user-attachments/assets/04b05ee0-a8db-4b8f-b3b9-cfd22e6fe93c)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.

