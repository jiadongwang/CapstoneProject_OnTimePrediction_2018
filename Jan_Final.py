

# In[1]:


#Import environments:
import sys
stdoutOrigin=sys.stdout 
sys.stdout = open("log13.txt", "w")
import pandas as pd
import numpy as np
import sklearn
import os
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
# In[ ]:


#Load Data:

airport = pd.read_excel(("~/Jan_2018/january_20181.xlsx" ))
airport.shape

airport.isnull().values.any()

airport.isnull().sum()

# In[ ]:


columns_to_drop = ["ORIGIN_STATE_ABR","DEST_CITY_NAME","DEST_STATE_ABR","ORIGIN_CITY_NAME"]


# In[ ]:


airport.fillna(value=0, inplace=True)
airport.isnull().values.any()

# In[ ]:


airport = pd.get_dummies(airport, columns=['ORIGIN','DEST'])
airport.head() 

# In[ ]:

airport.drop(labels=columns_to_drop, axis=1, inplace=True)
train_x, test_x, train_y, test_y = train_test_split(airport.drop('ARR_DEL15', axis=1), airport['ARR_DEL15'], test_size=0.2, random_state=101)




# In[ ]:


train_x.shape
# In[ ]:


test_x.shape
# In[ ]:

gboost = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.085, loss='deviance', max_depth=5,
              max_features=5, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=120,
              min_weight_fraction_leaf=0.29, n_estimators=60000,
              n_iter_no_change=None, presort='auto', random_state=200,
              subsample=1.0, tol=0.001, validation_fraction=0.3,
              verbose=0, warm_start=False)
gboost.fit(train_x, train_y) 
# In[ ]:


y_gboost_pred = gboost.predict(test_x)

labels = [0, 1]
cm = confusion_matrix(test_y, y_gboost_pred,labels)

gboost_accuracy = str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))
gboost_recall = str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))

print('GB Accuracy: ' + gboost_accuracy +'%')
print('Recall: ' + gboost_recall +'%')
print('Confusion matrix:')
print(cm)

fpr, tpr, _ = roc_curve(test_y, y_gboost_pred)
auc = np.trapz(fpr,tpr)
print('Area under the ROC curve: ' + str(auc))

fig = plt.figure(1)
plt.plot(fpr,tpr,color='green')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')

fig = plt.figure(2)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for Gradient Boosting classifier with original data')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# In[ ]:
dtree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=9,
            max_features=12, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=115,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
dtree.fit(train_x, train_y)

y_dtree_pred = dtree.predict(test_x)

labels = [0, 1]
cm = confusion_matrix(test_y, y_dtree_pred,labels)

dtree_accuracy = str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))
dtree_recall = str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))

print('DT Accuracy: ' + dtree_accuracy +'%')
print('Recall: ' + dtree_recall +'%')
print('Confusion matrix:')
print(cm)

fpr, tpr, _ = roc_curve(test_y, y_dtree_pred)
auc = np.trapz(fpr,tpr)
print('Area under the ROC curve: ' + str(auc))

fig = plt.figure(1)
plt.plot(fpr,tpr,color='green')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')

fig = plt.figure(2)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for Decision Tree classifier with original data')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


rforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=50, max_features=6, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=8, min_samples_split=7,
            min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rforest.fit(train_x, train_y)

# In[ ]:


y_rforest_pred = rforest.predict(test_x)

labels = [0, 1]
cm = confusion_matrix(test_y, y_rforest_pred,labels)

rforest_accuracy = str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))
rforest_recall = str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))

print('RF Accuracy: ' + rforest_accuracy +'%')
print('Recall: ' + rforest_recall +'%')
print('Confusion matrix:')
print(cm)

fpr, tpr, _ = roc_curve(test_y, y_rforest_pred)
auc = np.trapz(fpr,tpr)
print('Area under the ROC curve: ' + str(auc))

fig = plt.figure(1)
plt.plot(fpr,tpr,color='green')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')

fig = plt.figure(2)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for Random Forest classifier with original data')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

reg = linear_model.LogisticRegression(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=10, max_iter=2000, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=2, solver='warn',
          tol=0.00008, verbose=90, warm_start=True)
reg.fit(train_x, train_y)

y_reg_pred = reg.predict(test_x)

labels = [0, 1]
cm = confusion_matrix(test_y, y_reg_pred,labels)

reg_accuracy = str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))
reg_recall = str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))

print('LR Accuracy: ' + reg_accuracy +'%')
print('Recall: ' + reg_recall +'%')
print('Confusion matrix:')
print(cm)

fpr, tpr, _ = roc_curve(test_y, y_reg_pred)
auc = np.trapz(fpr,tpr)
print('Area under the ROC curve: ' + str(auc))

fig = plt.figure(1)
plt.plot(fpr,tpr,color='green')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')

fig = plt.figure(2)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion Matrix for Linear Regression')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

sys.stdout.close()
sys.stdout=stdoutOrigin
