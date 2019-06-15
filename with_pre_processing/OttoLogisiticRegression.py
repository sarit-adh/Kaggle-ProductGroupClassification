import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import csv




otto_train_ds = pd.read_csv("../train.csv")
otto_test_ds = pd.read_csv("../test.csv")


def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print otto_train_ds.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#Applying per row:
print "\nMissing values per row:"
print otto_train_ds.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row

num_records = otto_train_ds.shape[0]
num_features = otto_train_ds.shape[1] - 2

otto_train_ds['target'] = otto_train_ds['target'].replace(to_replace="^Class_",value="",regex=True).astype('category')

#preprocessing
#feature_columns = otto_train_ds.columns.difference(['target','id'])
#otto_train_ds[feature_columns] = preprocessing.scale(otto_train_ds[feature_columns]);

#feature,target split
X_train = otto_train_ds.iloc[:,1:num_features+1]
y_train = otto_train_ds.iloc[:,-1]
X_test  = otto_test_ds.iloc[:,1:]
IDs_test = [[(i)] for i in otto_test_ds.iloc[:,0]]




#prediction in test set
regressionClassifier = LogisticRegression(C=10)
regressionClassifier.fit(X_train, y_train)

prediction =  regressionClassifier.predict_proba(X_test)
prediction_with_id = np.hstack((IDs_test,prediction))
prediction_with_id[:,0] = prediction_with_id[:,0].astype(int)
   
np.savetxt('submission_logistic_regression', prediction_with_id,delimiter=',',fmt=','.join(['%i'] + ['%f']*9))

with open('submission_logistic_regression', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write( "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n"+ content)



