import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import csv



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


#the predict proba can be used as an additional feature
#print randomForestClassifier.predict(X_test)	#Predict class
#print randomForestClassifier.predict_log_proba(X_test)	#Predict class log-probabilities
#print randomForestClassifier.predict_proba(X_test)	#Predict class probabilities

#seed = 1
#scoring = 'accuracy' #other possible values precision, recall, f_score
#kfold = model_selection.KFold(n_splits=10, random_state=seed)

#n_estimators_list = [10,20,50,100]

#n_estimators_results_list =[]
#n_estimators_score_dict = {}
#for N_value in n_estimators_list:
#	randomForestClassifier = RandomForestClassifier(n_estimators=N_value)
#	cv_results = model_selection.cross_val_score(randomForestClassifier, X_train, y_train, cv=kfold, scoring=scoring)
#
#	n_estimators_score_dict[N_value] =  cv_results.mean()
#	n_estimators_results_list.append(cv_results)


#best_n = max(n_estimators_score_dict, key=n_estimators_score_dict.get)	
#print n_estimators_score_dict
#print "Best value for number of estimators is: " + str(best_n)

#Compare Hyperparameters
#fig = plt.figure()
#fig.suptitle('Hyperparameter Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(n_estimators_results_list)
#ax.set_xticklabels(n_estimators_list)
#plt.show()

#prediction in test set
randomForestClassifier = RandomForestClassifier(n_estimators=1000)
randomForestClassifier.fit(X_train, y_train)

prediction =  randomForestClassifier.predict_proba(X_test)

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