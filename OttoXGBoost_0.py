import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
import csv

#first installed using http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/ , had errors
# then installed using conda install -c mndrake xgboost, same error (module not found)
import xgboost



otto_train_ds = pd.read_csv("train.csv")
otto_test_ds = pd.read_csv("test.csv")


num_records = otto_train_ds.shape[0]
num_features = otto_train_ds.shape[1] - 2

otto_train_ds['target'] = otto_train_ds['target'].replace(to_replace="^Class_",value="",regex=True).astype('category')

#otto_train_ds.groupby("target").target.count().plot(kind='bar')
#plt.show()

X_train = otto_train_ds.iloc[:,1:num_features+1]
y_train = otto_train_ds.iloc[:,-1]
X_test  = otto_test_ds.iloc[:,1:]
IDs_test = [[int(i)] for i in otto_test_ds.iloc[:,0]]

xgboost_regressor = xgboost.XGBRegressor()
xgboost_regressor.fit(X_train,y_train)
prediction = xgboost_regressor.predict(X_test)

prediction_with_id = np.hstack((IDs_test,prediction))

prediction_with_id[:,0] = prediction_with_id[:,0].astype(int)

#submissionFile = open('submission.csv', 'w')  
# submissionFile.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
# with submissionFile:  
   # writer = csv.writer(submissionFile)
   # writer.writerows(prediction_with_id)
   
np.savetxt('submission.csv', prediction_with_id,delimiter=',',fmt=','.join(['%i'] + ['%f']*9))

with open('submission.csv', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write( "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n"+ content)