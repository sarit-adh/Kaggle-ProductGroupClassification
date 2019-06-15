import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

X_embedded = TSNE(n_components=2).fit_transform(X_train)
print X_embedded

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train)