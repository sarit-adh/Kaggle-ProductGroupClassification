import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from sklearn import model_selection
import graphlab as gl

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

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

# Check performance on internal validation set



boostedTreeClassifier = gl.boosted_trees_classifier.create(gl.SFrame("train.csv"), **params)


#prediction in test set

prediction =  boostedTreeClassifier.predict_topk(gl.SFrame("test.csv"),output_type='probability',k=9)
#prediction_with_id = np.hstack((IDs_test,prediction))
#prediction_with_id[:,0] = prediction_with_id[:,0].astype(int)

prediction['id'] = prediction['id'].astype(int) + 1
prediction = prediction.unstack(['class','probability'],'probs').unpack('probs','')
prediction = prediction.sort('id')
prediction.save("submission_boosted_tree.csv")
   
#np.savetxt('submission_boosted_tree.csv', prediction,delimiter=',',fmt=','.join(['%i'] + ['%f']*9))

#with open('submission_boosted_tree.csv', 'r+') as f:
#        content = f.read()
#        f.seek(0, 0)
#        f.write( "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n"+ content)



