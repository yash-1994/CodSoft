# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC  # Import Support Vector Machine classifier
from sklearn.naive_bayes import MultinomialNB  # Import Naive Bayes classifier
from sklearn.linear_model import LogisticRegression # Import accuracy_score function


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin1')

tfidf_vec = TfidfVectorizer()
x_data = tfidf_vec.fit_transform(data['v2'])
y_data = data['v1']

x_train, x_test, y_train, y_test_sol = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

def makeLogisticRegressionPred(x_train, y_train, x_test, y_test_sol):
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(x_train, y_train)
    y_test_pred = log_reg_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc

def makeSupportVectorMachinePred(x_train, y_train, x_test, y_test_sol):
    svm_model = SVC()
    svm_model.fit(x_train, y_train)
    y_test_pred = svm_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc

def makeNaiveBayesPred(x_train, y_train, x_test, y_test_sol):
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    y_test_pred = nb_model.predict(x_test)
    acc = accuracy_score(y_test_sol, y_test_pred)
    return acc


log_reg_acc = makeLogisticRegressionPred(x_train,y_train,x_test,y_test_sol) * 100
svm_acc = makeSupportVectorMachinePred(x_train,y_train,x_test,y_test_sol) * 100
nb_acc = makeNaiveBayesPred(x_train,y_train,x_test,y_test_sol) * 100
print('Result is Ready!')


print(f'The Accuracy we Got from \nLogistic Regression: {log_reg_acc} \nRandom Forest: {svm_acc} \nGradient Boosting: {nb_acc}')