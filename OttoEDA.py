import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.decomposition import PCA

otto_train_ds = pd.read_csv("train.csv")
otto_test_ds = pd.read_csv("test.csv")


num_records = otto_train_ds.shape[0]
num_features = otto_train_ds.shape[1] - 2

otto_train_ds['target'] = otto_train_ds['target'].replace(to_replace="^Class_",value="",regex=True).astype('category')

otto_train_ds.groupby("target").target.count().plot(kind='bar').set(xlabel='class',ylabel='count')
plt.show()

X_train = otto_train_ds.iloc[:,1:num_features+1]
y_train = otto_train_ds.iloc[:,-1]
X_test  = otto_test_ds.iloc[:,1:]
IDs_test = otto_test_ds.iloc[:,1]


#use random forest classifier to get feature importance
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, y_train)

feaure_score_pairs = zip(otto_train_ds.columns.values[1:num_features+1], randomForestClassifier.feature_importances_) # zip with features_list
feaure_score_pairs = sorted(feaure_score_pairs, key=lambda x: x[1], reverse= True)


features, scores = zip(*feaure_score_pairs)


fig,(ax1,ax2) = plt.subplots(nrows=2)
sns.barplot(features[0:21],scores[0:21],ax=ax1).set(xlabel="Feature",ylabel="Feature scores",title="most important features")
sns.barplot(features[-10:],scores[-10:],ax=ax2).set(xlabel="Feature",ylabel="Feature scores",title="least important features")

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_train)

print pca_result

print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

print df





