# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv('/kaggle/input/fraud-detection/fraudTrain.csv')
test_data = pd.read_csv('/kaggle/input/fraud-detection/fraudTest.csv')

x_train_data = data.copy()
x_train_data.drop('is_fraud', axis=1, inplace=True)

x_test_data = test_data.copy()
x_test_data.drop('is_fraud', axis=1, inplace=True)

def makeLabledData(data):
    label_encoder = LabelEncoder()
    for column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data


y_train_data = data['is_fraud']
x_train_data = makeLabledData(x_train_data)

x_test_data = makeLabledData(x_test_data)
y_test_sol = test_data['is_fraud']


def makeLogisticPred(x_train,y_train,x_test,y_test_sol):
    log_reg_model = LogisticRegression(max_iter = 1000)
    log_reg_model.fit(x_train,y_train)
    y_test_pred = log_reg_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc

def makeDecisionTreePred(x_train,y_train,x_test,y_test_sol):
    dec_tree_model = DecisionTreeClassifier()
    dec_tree_model.fit(x_train,y_train)
    y_test_pred = dec_tree_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc

def makeRandomForestPred(x_train,y_train,x_test,y_test_sol):
    ran_for_model = RandomForestClassifier()
    ran_for_model.fit(x_train,y_train)
    y_test_pred = ran_for_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc


log_reg_acc = makeLogisticPred(x_train_data,y_train_data,x_test_data,y_test_sol) * 100
dec_tree_acc = makeDecisionTreePred(x_train_data,y_train_data,x_test_data,y_test_sol) * 100
ran_for_acc = makeRandomForestPred(x_train_data,y_train_data,x_test_data,y_test_sol) * 100
print('Result is Ready!')


print(f'The Accuracy we Got from the \nLogistic Regression: {log_reg_acc}\nDecision Tree: {dec_tree_acc}\nRandom Forest: {ran_for_acc}')