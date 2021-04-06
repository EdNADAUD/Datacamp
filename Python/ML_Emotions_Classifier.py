#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import time
import seaborn as sn
from sklearn import *
from sklearn.linear_model import *
from sklearn.neural_network import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


# In[3]:


dataPi = pd.read_csv('/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/Dataset/trainset/trainset.csv')


# On doit utiliser le dataset train pour tester les résultats étant donné que dans le dataset test il n'y a pas les labels.

# In[4]:


X_dataPi = dataPi[dataPi.columns[1:-1]]
y_dataPi = dataPi[dataPi.columns[-1]]


# ## Mais pourquoi? Commençons par étudier la répartission des données:

# In[5]:


emotions = {0:"neutre", 1:"colère", 3:"dégoût", 4:"peur", 5:"joie", 6:"tristesse", 7:"surprise"}
print("Il y a ", dataPi.shape[0] , "exemples en tout dans le dataset dont :")
[print ( dataPi[dataPi['label'] == a].shape[0], "exemples d'images de classe",a,"(",emotions[a],")") for a in np.sort(dataPi['label'].unique())];


# In[6]:


plt.figure(figsize=(7,7))
plt.title("Répartission des données de la base train")
plt.pie([ dataPi[dataPi['label'] == a].shape[0] for a in np.sort(dataPi['label'].unique())], autopct='%1.1f%%',labels=emotions.values());


# ### Comme nous le voyons ici les classes de notre base de test ont des données très inégalement réparties. Comment faire pour résoudre ce problème? Nous devons effectuer des recherches.

# # Apprentissage déséquilibré ou  imbalenced learning
# 
# 
# ### Bibliographie:
# 
# ### <i> Machine Learning from Imbalanced Data Sets 101</i>, Foster Provost, NY University
# #### https://www.aaai.org/Papers/Workshops/2000/WS-00-05/WS00-05-001.pdf
# 
# ### <i>Learning from imbalanced data: open challenges and future directions</i>, Bartosz Krawczyk, Wrocław University of Technology
# #### https://link.springer.com/article/10.1007/s13748-016-0094-0?TB_iframe=true&error=cookies_not_supported&code=a3e33168-782e-41e5-8585-e731754069d2
# 
# ### <i>Imbalenced Learning</i>, Haibo He & Yunqian Ma
# #### https://books.google.fr/books?hl=fr&lr=&id=CVHx-Gp9jzUC&oi=fnd&pg=PT9&dq=machine+learning+imbalanced+data&ots=2iLmIhzp5g&sig=vMMBD-6KVEKXxjOeFA8pPd6B3rI#v=onepage&q=machine%20learning%20imbalanced%20data&f=false
# 
# https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/
# 
# ### Ajinkya More | Resampling techniques and other strategies
# #### https://www.youtube.com/watch?v=-Z1PaqYKC1w

# In[7]:


import math




def dist(point1, point2, df):
    point1, point2 = str(point1), str(point2)
    return  (  (df[' x_' + point2] - df[' x_' + point1])**(2) +  (df[' y_' + point2] - df[' y_' + point1])**(2)   )**(0.5)

def dist_normalisee(point1, point2, df, coef): # normalisée par la distance interoculaire
    point1, point2 = str(point1), str(point2)
    return  (((  (df[' x_' + point2] - df[' x_' + point1])**(2) +  (df[' y_' + point2] - df[' y_' + point1])**(2)   )**(0.5))/coef)

def angl(point1, point2, point3, df):
    point1, point2, point3 = str(point1), str(point2), str(point3)
    a = np.array([df[' x_' + point1],df[' y_' + point1]])
    b = np.array([df[' x_' + point2],df[' y_' + point2]])
    c = np.array([df[' x_' + point3],df[' y_' + point3]])
    
    angle = np.degrees(np.arctan(c[1]-b[1], c[0]-b[0]) - np.arctan(a[1]-b[1], a[0]-b[0]))
 
    angle[angle<0] = angle[angle<0] + 360
    return angle


# In[8]:


from sklearn.preprocessing import StandardScaler
X_new_ft = X_dataPi.copy()

X_new_ft['distOculaire'] = (  (X_new_ft.loc[: , " x_42":" x_47"].mean(axis=1) - X_new_ft.loc[: , " x_36":" x_41"].mean(axis=1))**(2) +  (X_new_ft.loc[: , " y_42":" y_47"].mean(axis=1) - X_new_ft.loc[: , " y_36":" y_41"].mean(axis=1))**(2)   )**(0.5)


for i in range(0,68):
    for j in range(i,68): # afin de ne pas recalculer deux fois la meme distance
        if (i != j): # on ne calcule pas la distance d'un point avec lui même
            #X_new_ft['dist'+str(i)+"_"+str(j)]  = dist(i, j, X_dataPi)
            X_new_ft['dist_norm'+str(i)+"_"+str(j)]  = dist_normalisee(i, j, X_dataPi,X_new_ft['distOculaire'])

for i in range(0,68):
    for j in range(0,68): # afin de ne pas recalculer deux fois le même angle
        for k in range(i,68):
            if (i != j and j != k and i != k): # on ne calcule pas d'angle entre 3 points quand deux d'entre eux sont les mêmes             
                X_new_ft['angl'+str(i)+"_"+str(j)+"_"+str(k)] = angl(i, j, k, X_dataPi)
    print('clacul pour les angles a=',i)

X_new_ft.drop(X_new_ft.iloc[:, 0:137], inplace = True, axis = 1) 

ColonnesDf = X_new_ft.columns
indexDf = X_new_ft.index
standardScaler = StandardScaler()
X_new_ft = standardScaler.fit_transform(X_new_ft)

X_new_ft = pd.DataFrame(X_new_ft, index=indexDf, columns=ColonnesDf)

X_nf_train, X_nf_test, y_dataPiTrain, y_dataPiTest = model_selection.train_test_split(X_new_ft, y_dataPi, train_size=0.75, test_size=0.25,shuffle=False)


# In[9]:


from sklearn.model_selection import GridSearchCV
parameters = { 'n_neighbors':[1,2,3],'weights':['uniform','distance'],'metric':['euclidean','manhattan']}
#svc = svm.SVC()
#knnModel = KNeighborsClassifier()
gscv = GridSearchCV(KNeighborsClassifier(), parameters,verbose=1,cv=3,n_jobs = -1)
gsres = gscv.fit(X_nf_train, y_dataPiTrain)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[10]:


knnModels = []
KnnResultsTest = []
KnnResultsTrain = []

knnModel = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
knnModel.fit(X_nf_train,y_dataPiTrain)
print(knnModel.score(X_nf_test,y_dataPiTest))

yPredicted=knnModel.predict(X_nf_test)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[11]:


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)

pca.fit(X_nf_train)

XPCA = pca.transform(X_nf_train)

XPCADf = pd.DataFrame(data = XPCA, columns = ['pc1', 'pc2', 'pc3'])
colors = {0:'red', 1:'green', 3:'blue', 4:'pink',5:'yellow', 6:'cyan', 7:'purple'}
fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(XPCADf.pc1, XPCADf.pc2, XPCADf.pc3, c=y_dataPiTrain.map(colors),alpha=0.5);


# In[12]:


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=2)

pca.fit(X_nf_train)

XPCA = pca.transform(X_nf_train)

XPCADf = pd.DataFrame(data = XPCA, columns = ['pc1', 'pc2'])
colors = {0:'red', 1:'green', 3:'blue', 4:'black',5:'yellow', 6:'cyan', 7:'purple'}
fig = plt.figure()
plt.scatter(XPCADf.pc1, XPCADf.pc2, c=y_dataPiTrain.map(colors),alpha=0.5);


# ## Apprentissage Non supervise
# 

# In[13]:


from sklearn.cluster import KMeans
knnModels = []
KnnResultsTest = []
KnnResultsTrain = []

knnModel = KMeans(n_clusters=7, algorithm='auto')
knnModel.fit(X_nf_train,y_dataPiTrain)
print(knnModel.score(X_nf_test,y_dataPiTest))

yPredicted=knnModel.predict(X_nf_test)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[14]:


'''clf = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='tanh', solver='sgd', batch_size=1, alpha=0, learning_rate='adaptive', verbose=0)
clf.fit(X_nf_train,y_dataPiTrain)
print(clf.score(X_nf_test,y_dataPiTest))
yPredicted=clf.predict(X_nf_test)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");'''


# In[17]:


#rapportDeResampling(X_nf_train, y_dataPiTrain,X_nf_test, y_dataPiTest,DecisionTreeClassifier(class_weight='balanced'),emotions.values() )


# In[20]:


from sklearn.preprocessing import MinMaxScaler


# In[21]:


X_nf_train_copy = X_nf_train.copy()

X_nf_train_copy = MinMaxScaler().fit_transform(X_nf_train_copy)
X_nf_train_copy = pd.DataFrame(X_nf_train_copy, index=X_nf_train.index, columns=ColonnesDf)


# In[22]:


X_nf_train_copy.head(2)


# In[23]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=100)
bestfeatures = bestfeatures.fit(X_nf_train_copy,y_dataPiTrain)
dfscores = pd.DataFrame(bestfeatures.scores_)
dfcolumns = pd.DataFrame(X_nf_train_copy.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
featureScores.sort_values("Score",ascending=False, inplace = True) 
featureScores[0:100]


# In[24]:


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
pca.fit(X_nf_train_copy)

XPCA = pca.transform(X_nf_train_copy)

XPCADf = pd.DataFrame(data = XPCA, columns = ['pc1', 'pc2', 'pc3'])
colors = {0:'red', 1:'green', 3:'blue', 4:'pink',5:'yellow', 6:'cyan', 7:'purple'}
fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(XPCADf.pc1, XPCADf.pc2, XPCADf.pc3, c=y_dataPiTrain.map(colors),alpha=0.5);


# In[25]:


from imblearn.over_sampling import ADASYN
X_new_ft_PCA = X_new_ft.copy()

pca = PCA(n_components=722)
pca.fit(X_new_ft_PCA)
XPCA = pca.transform(X_new_ft_PCA)


# In[26]:


X_nf_train_pca, X_nf_test_pca, y_dataPiTrain_pca, y_dataPiTest_pca = model_selection.train_test_split(XPCA, y_dataPi, train_size=0.75, test_size=0.25,shuffle=True)
X_nf_train_pca, y_dataPiTrain_pca = ADASYN().fit_resample(X_nf_train_pca,y_dataPiTrain_pca)


# In[27]:


knnModels = []
KnnResultsTest = []
KnnResultsTrain = []

knnModel = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 1, weights= 'uniform')
knnModel.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(knnModel.score(X_nf_test_pca,y_dataPiTest_pca))

yPredicted=knnModel.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[28]:


print(classification_report(y_dataPiTest_pca,yPredicted,zero_division=0))


# In[29]:


from sklearn.model_selection import GridSearchCV
parameters = { 'n_neighbors':range(1,10),'weights':['uniform','distance'],'metric':['minkowski','euclidean','manhattan']}
#svc = svm.SVC()
#knnModel = KNeighborsClassifier()
gscv = GridSearchCV(KNeighborsClassifier(), parameters,verbose=1,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[30]:


from sklearn.tree import DecisionTreeClassifier
parameters = { 'criterion':['gini', 'entropy'],'max_depth':range(1,722,50)}
gscv = GridSearchCV(DecisionTreeClassifier(), parameters,verbose=1,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[31]:


clf = DecisionTreeClassifier(criterion='entropy', max_depth = 422)
clf.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(clf.score(X_nf_test_pca,y_dataPiTest_pca))

yPredicted=clf.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[32]:


from sklearn.ensemble import RandomForestClassifier
parameters = { 'n_estimators':range(1,300,50),'max_depth':range(1,722,100),'criterion':['gini', 'entropy']}
gscv = GridSearchCV(RandomForestClassifier(), parameters,verbose=1,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[82]:


clf = RandomForestClassifier(criterion='gini',n_estimators=201, max_depth = 401)
clf.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(clf.score(X_nf_test_pca,y_dataPiTest_pca))

yPredicted=clf.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle Randomforest");


# In[95]:


from sklearn.svm import LinearSVC
parameters = { 'penalty':['l1', 'l2'],'dual':[True, False],'C':np.arange(0.1, 5, 0.5)}
gscv = GridSearchCV(LinearSVC(), parameters,verbose=1,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[81]:


clf = LinearSVC(C=0.1, penalty='l1',dual = False,max_iter=20000) 
clf.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(clf.score(X_nf_test_pca,y_dataPiTest_pca))

yPredicted=clf.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle linear SVC");


# In[36]:


print(classification_report(y_dataPiTest_pca,yPredicted,zero_division=0))


# In[72]:


from sklearn.svm import NuSVC
parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'nu':np.arange(0, 1, 0.3),'gamma':['scale', 'auto']}
gscv = GridSearchCV(NuSVC(), parameters,verbose=10,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# #### Il y a quelques erreurs qui ne sont pas vraiment problématiques, elles correspondent au fait que le F1 score peut donner des scores négatifs, ce qui peut être mal géré, comme expliqué dans l'article suivant: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

# In[73]:


from sklearn.svm import SVC
parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'C':np.arange(0, 3, 0.5),'gamma':['scale', 'auto']}
gscv = GridSearchCV(SVC(), parameters,verbose=1,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[80]:


from sklearn.svm import SVC
clf = SVC(C=0.5, kernel='linear',max_iter=20000) 
clf.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(clf.score(X_nf_test_pca,y_dataPiTest_pca))

yPredicted=clf.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle SVC")


# In[74]:


from sklearn.neural_network import MLPClassifier
parameters = {'hidden_layer_sizes':[(20),(100),(20,20),(20,20,20),(100,100), (100,100,100),(200),(200,200),(200,200,200)]}
gscv = GridSearchCV(MLPClassifier(), parameters,verbose=1,cv=5,n_jobs = -1, scoring='f1_micro')
gsres = gscv.fit(X_nf_train_pca,y_dataPiTrain_pca)
print("best score",gsres.best_score_, "best estimator",gsres.best_estimator_, "best params", gsres.best_params_)


# In[77]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200,200 )) 
clf.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(clf.score(X_nf_test_pca,y_dataPiTest_pca))

yPredicted=clf.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle MLP");


# In[78]:


classifier = LogisticRegressionCV(cv=20, n_jobs=-1)
classifier.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(classifier.score(X_nf_test_pca,y_dataPiTest_pca))


# In[79]:


yPredicted=classifier.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle Logistic regression");


# # VOTING CLASSIFIER

# In[83]:


from sklearn.ensemble import VotingClassifier
model1=SVC(C=0.5, kernel='linear',max_iter=20000) 
model2=MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200,200 )) 
model3=LinearSVC(C=0.1, penalty='l1',dual = False,max_iter=20000) 
model4=LogisticRegressionCV(cv=20, n_jobs=-1)
model5= VotingClassifier([('SVC',model1),
                           ('MLPClassifier',model2),
                           ('LinearSVC',model3)],
                             voting='hard')

for model in (model1,model2,model3,model4,model5):
    model.fit(X_nf_train_pca,y_dataPiTrain_pca)
    print(model.score(X_nf_test_pca,y_dataPiTest_pca))


# In[84]:


yPredicted=model.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle voting");


# # BAGGING 

# pour des model singulier avec de l'over fitting

# In[85]:


from sklearn.ensemble import BaggingClassifier


# In[86]:


modelbaggin=BaggingClassifier(SVC(C=0.5, kernel='linear',max_iter=20000) ,n_estimators=5)
modelbaggin.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(modelbaggin.score(X_nf_test_pca,y_dataPiTest_pca))


# In[87]:


yPredicted=modelbaggin.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle bagging with SVC");


# In[88]:


modelbaggin=BaggingClassifier(DecisionTreeClassifier(criterion='entropy', max_depth = 422),n_estimators=200)
modelbaggin.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(modelbaggin.score(X_nf_test_pca,y_dataPiTest_pca))


# In[89]:


yPredicted=modelbaggin.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle bagging with decision three");


# avantage des decisionTree= vitesse d'execution et d'entrainement

# 
# # BOOSTING

# In[90]:


from sklearn.ensemble import  AdaBoostClassifier


# In[91]:


modelboosting=AdaBoostClassifier(n_estimators=1000)
modelboosting.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(modelboosting.score(X_nf_test_pca,y_dataPiTest_pca))


# In[92]:


yPredicted=modelbaggin.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle boosting");


# # Stacking

# Si on a des models individuel tres entrainée

# In[93]:


from sklearn.ensemble import StackingClassifier
modelstacking= StackingClassifier([('SVC',model1),
                           ('MLPClassifier',model2),
                           ('LinearSVC',model3)],
                         final_estimator=DecisionTreeClassifier())
modelstacking.fit(X_nf_train_pca,y_dataPiTrain_pca)
print(modelstacking.score(X_nf_test_pca,y_dataPiTest_pca))


# In[94]:


yPredicted=modelbaggin.predict(X_nf_test_pca)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest_pca,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle stacking");


# In[ ]:




