import matplotlib.pyplot as plt
import seaborn as sns

x = ["KNN","randomForest","logisticRegression","XGBoost","XGBoost(tuning)","XGBoost+KNN","KNN+randomForest"]
y = [2.34,0.55,0.67,0.65,0.48,0.47,0.52]

sns.barplot(x,y)