# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv",encoding="latin1")


x_col = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

x_data = data[x_col].copy()
y_data = data['Exited'].copy()

def label_data(data_frame, columns_to_label):
    labeled_data_frame = data_frame.copy()
    label_encoder = LabelEncoder()
    
    for column in columns_to_label:
        labeled_data_frame[column] = label_encoder.fit_transform(labeled_data_frame[column])
    
    return labeled_data_frame

x_data = label_data(x_data,['Geography','Gender'])

x_train, x_test, y_train, y_test_sol = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


def makeLogisticRegressionPred(x_train,y_train,x_test,y_test_sol):
    log_reg_model = LogisticRegression(max_iter = 1000)
    log_reg_model.fit(x_train,y_train)
    y_test_pred = log_reg_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc

def makeRandomForestPred(x_train,y_train,x_test,y_test_sol):
    ran_for_model = RandomForestClassifier()
    ran_for_model.fit(x_train,y_train)
    y_test_pred = ran_for_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc

def makeGradientBoostingPred(x_train,y_train,x_test,y_test_sol):
    gra_bst_model = GradientBoostingClassifier()
    gra_bst_model.fit(x_train,y_train)
    y_test_pred = gra_bst_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc


log_reg_acc = makeLogisticRegressionPred(x_train,y_train,x_test,y_test_sol) * 100
ran_for_acc = makeRandomForestPred(x_train,y_train,x_test,y_test_sol) * 100
gra_bst_acc = makeGradientBoostingPred(x_train,y_train,x_test,y_test_sol) * 100
print('ok')


print(f'The Accuracy we Got from \nLogistic Regression: {log_reg_acc} \nRandom Forest: {ran_for_acc} \nGradient Boosting: {gra_bst_acc}')