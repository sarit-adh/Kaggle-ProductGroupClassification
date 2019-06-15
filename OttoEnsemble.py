import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


otto_train_ds = pd.read_csv("train.csv")
otto_test_ds = pd.read_csv("test.csv")


num_records = otto_train_ds.shape[0]
num_features = otto_train_ds.shape[1] - 2

#could have used label encoder instead
otto_train_ds['target'] = otto_train_ds['target'].replace(to_replace="^Class_",value="",regex=True).astype('category')



#otto_train_ds.groupby("target").target.count().plot(kind='bar')
#plt.show()

X_train = otto_train_ds.iloc[:,1:num_features+1]
y_train = otto_train_ds.iloc[:,-1]
X_test  = otto_test_ds.iloc[:,1:]
IDs_test = [[int(i)] for i in otto_test_ds.iloc[:,0]]


#stratifiedShuffleSplit(y_train, test_size=0.05, random_state=1)

classifiers = []

X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,random_state=1)

#logisticRegressionClassifier = LogisticRegression(random_state=1,C=1)
#logisticRegressionClassifier.fit(X_train,y_train)
#print('Logistic Regression Classifier LogLoss {score}').format(score=log_loss(y_val,logisticRegressionClassifier.predict_proba(X_val)))
#classifiers.append(logisticRegressionClassifier)

randomForestClassifier = RandomForestClassifier(n_estimators=1000, random_state=1,n_jobs=-1)
randomForestClassifier.fit(X_train,y_train) 
print('Random Forest Classifier LogLoss {score}').format(score=log_loss(y_val,randomForestClassifier.predict_proba(X_val)))
classifiers.append(randomForestClassifier)

#randomForestClassifier2 = RandomForestClassifier(n_estimators=1000, random_state=2, n_jobs=-1)
#randomForestClassifier2.fit(X_train,y_train)
#print('Random Forest Classifier 2 LogLoss {score}').format(score=log_loss(y_val,randomForestClassifier2.predict_proba(X_val)))
#classifiers.append(randomForestClassifier2)
#
#extraTreeClassifier = ExtraTreeClassifier()
#extraTreeClassifier.fit(X_train,y_train)
#print('Extra Tree Classifier 2 LogLoss {score}').format(score=log_loss(y_val,extraTreeClassifier.predict_proba(X_val)))
#classifiers.append(extraTreeClassifier)

kNeighborsClassifier = KNeighborsClassifier(n_neighbors=5)
kNeighborsClassifier.fit(X_train,y_train)
print('K Neighbors Classifier LogLoss {score}').format(score=log_loss(y_val,kNeighborsClassifier.predict_proba(X_val)))
classifiers.append(kNeighborsClassifier)

### finding the optimal weights
predictions =[]
for classifier in classifiers:
        predictions.append(classifier.predict_proba(X_val))

def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight,prediction in zip(weights,predictions):
                final_prediction += weight*prediction
        return log_loss(y_val, final_prediction)
starting_values = [0.5]* len(predictions)

constraints_dict = ({'type' : 'eq' , 'fun' : lambda w: 1-sum(w)})
bounds = [(0,1)] * len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP' , bounds=bounds, constraints = constraints_dict)

print('Ensemble Score: {best_score}').format(best_score=res['fun'])
print('Best Weights: {weights}').format(weights=res['x'])


#def make_prediction(weights,classifiers,test_features):
#	prediction =0 
#	for weight,classifier in zip(weights,classifiers):
#		prediction += weight * classifier.predict_proba(test_features)
#	return prediction
#
#prediction =  make_prediction(res['x'],classifiers,X_test)
#
#prediction_with_id = np.hstack((IDs_test,prediction))
#
#prediction_with_id[:,0] = prediction_with_id[:,0].astype(int)
#   
#np.savetxt('submission.csv', prediction_with_id,delimiter=',',fmt=','.join(['%i'] + ['%f']*9))
#
#with open('submission.csv', 'r+') as f:
#        content = f.read()
#        f.seek(0, 0)
#        f.write( "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n"+ content)	