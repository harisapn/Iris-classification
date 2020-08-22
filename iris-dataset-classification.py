#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[6]:


#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #specifying names of columns
dataset = read_csv(url, names=names)

#Explore dataset
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())


# In[7]:


#class distribution
print(dataset.groupby('class').size())


# In[10]:


#Data visualization
dataset.plot(kind="box",subplots=True,layout=(2,2),sharex=False,sharey=False) #boxes
pyplot.show()


# In[11]:


dataset.hist() #histogram
pyplot.show()


# In[12]:


#Multivariate plots
scatter_matrix(dataset)
pyplot.show()


# In[62]:


#Evaluate some algorithms
array=dataset.values

X=array[:,0:4]
y=array[:,4]
X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape


# In[76]:


#Build model
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#evaluate each model
results = []
names = []
#lr=LogisticRegression(solver='liblinear', multi_class='ovr')
#lr.fit(X_train,Y_train)
#results=cross_val_score(lr,X_train,Y_train)
#results



for name, model in models: 
    model.fit(X_train,Y_train)
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
 
  


# In[77]:


#Visualize the model performance

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[79]:


#Make predictions
mojmodel=LinearDiscriminantAnalysis()
mojmodel.fit(X_train,Y_train)
predictions=mojmodel.predict(x_test)
print(predictions)


# In[80]:


print(y_test)


# In[82]:


#Evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:




