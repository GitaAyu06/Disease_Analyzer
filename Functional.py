#import library
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from collections import Counter as ct

#import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import data
data_input = input("Enter your file name : ")
data = pd.read_csv(data_input)

#Base data that used for prediction
base_data = input("Enter your data name : ")
data_p = input ("Enter \'Positive\' Variable data Notation : ")
if data_p == "0" or data_p == "1" :
    basedata = int(data_p)
else :
    basedata = str(data_p)

data["Patient Condition"] = np.where(
    data[base_data]==basedata,"Positive","Negative")

#mapping data
n_o_d= input("How much your data parameter ? ")
columns = int(n_o_d)
variable_array = [ input("Data Parameter {} : ".format (i+1)) for i in range(columns)]
variable_array.append("Patient Condition")
data = data[variable_array].dropna(axis=0,how='any')

#split dataset
train, test = train_test_split(data, test_size=0.6, random_state=int(4))
gnb = GaussianNB()
newarr = []
newarr.extend(variable_array)
newarr.remove("Patient Condition")

gnb.fit(train[newarr].values, train["Patient Condition"])
result = gnb.predict(test[newarr])

# Print Performance Indicator
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          test.shape[0],
          (test["Patient Condition"] != result).sum(),
          100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
          ))

test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
test_data["Patient Condition"] = result
print (test_data)


'''
counts = ct(y_pred)
count_Male = counts['Male']
count_Female = counts['Female']
count_Infant = counts['Infant']

slices = [count_Male,count_Female,count_Infant]
cols = ['r','g','b','k']
plt.pie(slices, labels=['Male','Female','Infant'],colors = cols,shadow=True,startangle=90,autopct='%1.1f%%')
plt.title("Test")
plt.legend()
plt.show()
'''

