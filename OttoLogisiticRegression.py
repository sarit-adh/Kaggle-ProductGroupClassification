import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
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
IDs_test = [[(i)] for i in otto_test_ds.iloc[:,0]]

seed = 1
scoring = 'accuracy' #other possible values precision, recall, f_score
kfold = model_selection.KFold(n_splits=10, random_state=seed)

C_values_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

C_score_results_list =[]
C_score_dict = {}
for C_value in C_values_list:
	regression_model = LogisticRegression(C=C_value)
	cv_results = model_selection.cross_val_score(regression_model, X_train, y_train, cv=kfold, scoring=scoring)

	C_score_dict[C_value] =  cv_results.mean()
	C_score_results_list.append(cv_results)

best_C = max(C_score_dict, key=C_score_dict.get)	
print C_score_dict
print "Best value for C is: " + str(best_C)

#Compare Hyperparameters
fig = plt.figure()
fig.suptitle('Hyperparameter Comparison')
ax = fig.add_subplot(111)
plt.boxplot(C_score_results_list)
ax.set_xticklabels(C_values_list)
plt.show()


#prediction in test set
regressionClassifier = LogisticRegression(C=best_C)
regressionClassifier.fit(X_train, y_train)

prediction =  regressionClassifier.predict_proba(X_test)
prediction_with_id = np.hstack((IDs_test,prediction))
prediction_with_id[:,0] = prediction_with_id[:,0].astype(int)
   
np.savetxt('submission.csv', prediction_with_id,delimiter=',',fmt=','.join(['%i'] + ['%f']*9))

with open('submission.csv', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write( "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n"+ content)



