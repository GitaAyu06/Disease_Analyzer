# Disease_Analyzer
This program gives information about patients condition towards certain diseases based on their health conditions (such as, blood pressure, BMI, Ca levels, etc.) using Naive Bayes Algorithm. 

![Flowchart](Flowchart.jpg)

PROGRAM TUTORIALS
======

This tutorial will explain several steps to build Diseases_Analyzer Program (without GUI). This program use Naive Bayes algorithm to make predictions about patient diagnosis towards certain diseases. So, it calculates several parameters to make a prediction.


Requirements
--------

To make this program, you need to install:
  - Python (the author use Python 3.6.5)
  - Pandas
  - Numpy
  - Xlsxwriter
  - Matplotlib
  - SkLearn
  - Tkinter (if you want to use GUI in your program)

Build The Code
-----------

1. Import libraries
```
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from collections import Counter as ct
```
```
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

```
2. Import data
```
data_input = input("Enter your file name : ")
data = pd.read_csv(data_input)
```
3. Choose reference column 
```
base_data = input("Enter your data name : ")
```
4. Mapping data in reference column
```
data_p = input ("Enter \'Positive\' Variable data Notation : ")
if data_p == "0" or data_p == "1" :
    basedata = int(data_p)
else :
    basedata = str(data_p)

data["Patient Condition"] = np.where(
data[base_data]==basedata,"Positive","Negative")
```
5. Determine parameters
```
n_o_d= input("How much your data parameter ? ")
columns = int(n_o_d)
variable_array = [ input("Data Parameter {} : ".format (i+1)) for i in range(columns)]
variable_array.append("Patient Condition")
data = data[variable_array].dropna(axis=0,how='any')
```
6. Split data as train set and test set
```
train, test = train_test_split(data, test_size=0.6, random_state=int(4))
gnb = GaussianNB()
newarr = []
newarr.extend(variable_array)
newarr.remove("Patient Condition")

gnb.fit(train[newarr].values, train["Patient Condition"])
result = gnb.predict(test[newarr])
```
7. Calculate prediction accuracy
```
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          test.shape[0],
          (test["Patient Condition"] != result).sum(),
          100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
          ))

test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
test_data["Patient Condition"] = result
```
8. Print test set
```
print (test_data)
```
9. Count "Positive" and "Negative" patient
```
counts = ct(result)
count_p = counts['Positive']
count_n = counts['Negative']
```
10. Show diagram that show Positive" and "Negative" patient proportion
```
slices = [count_p,count_n]
cols = ['b','c']
plt.pie(slices, labels=['Positve','Negative'],colors = cols,shadow=True,startangle=90,autopct='%1.1f%%')
plt.title("Patient Condition")
plt.legend()
plt.show()
```


