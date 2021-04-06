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


# In[2]:


dataPi = pd.read_csv('trainset/trainset.csv')


# On doit utiliser le dataset train pour tester les résultats étant donné que dans le dataset test il n'y a pas les labels.

# In[3]:


X_dataPi = dataPi[dataPi.columns[1:-1]]
y_dataPi = dataPi[dataPi.columns[-1]]

X_dataPiTrain, X_dataPiTest, y_dataPiTrain, y_dataPiTest = model_selection.train_test_split(X_dataPi, y_dataPi, train_size=0.75, test_size=0.25)


# On va déjà essayer avec un knn juste pour voir:

# In[4]:


knnModels = []
KnnResultsTest = []
#knnClassificationTime = []
KnnResultsTrain = []
for k in range (1,10):
    knnModels.append(KNeighborsClassifier(n_neighbors=k, algorithm='brute'))
    knnModels[k-1].fit(X_dataPiTrain,y_dataPiTrain)
    
    #start_time=time.time()
    knnModels[k-1].predict(X_dataPiTest)
    #knnClassificationTime.append(time.time() - start_time)
    
    KnnResultsTest.append(knnModels[k-1].score(X_dataPiTest,y_dataPiTest))
    KnnResultsTrain.append(knnModels[k-1].score(X_dataPiTrain,y_dataPiTrain))


# In[5]:


plt.figure(figsize=(15,7))
plt.title("Influence de K taux de reconnaissance")
plt.ylabel('score')
plt.xlabel('nombre de neurones de la couche cachée')
plt.plot(range(1,10),KnnResultsTest,label="Resultats sur base test")
plt.plot(range(1,10) ,KnnResultsTrain,label="Resultats sur base train")
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5, fontsize = 'large');


# ## Interprétation naïve : notre classifieur semble déjà avoir un taux de reconnaissance qui est intéressant avec environ 70% de réussite pour un K = 3 sans aucun prétraitement des données.
# 
# ## Vérifions la matrice de confusion:

# In[6]:


emotions = {0:"neutre", 1:"colère", 3:"dégoût", 4:"peur", 5:"joie", 6:"tristesse", 7:"surprise"}
yPredicted=knnModels[2].predict(X_dataPiTest)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[7]:


print ("La précision du modèles pour les classes différentes de 'neutre' est de :")
print(sum(y_dataPiTest[yPredicted == y_dataPiTest] != 0) / sum(y_dataPiTest != 0))
print ("Nous pouvons aussi utiliser la fonction balanced accuracy de sickitlearn qui nous donne une précision de:")
from sklearn.metrics import balanced_accuracy_score
print(balanced_accuracy_score(y_dataPiTest, yPredicted))


# In[8]:


print(classification_report(y_dataPiTest,yPredicted,zero_division=0))


# ## Nous en déduisons que le résultat est bien plus catastrophique que ce que l'on aurait pu croire. En effet le modèle a tendance à classer quasiment tous les éléments en neutre. Nous remarquons que le taux de précision est très mauvaise
# 

# ## Mais pourquoi? Commençons par étudier la répartission des données:

# In[9]:


print("Il y a ", dataPi.shape[0] , "exemples en tout dans le dataset dont :")
[print ( dataPi[dataPi['label'] == a].shape[0], "exemples d'images de classe",a,"(",emotions[a],")") for a in np.sort(dataPi['label'].unique())];


# In[10]:


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

# In[11]:


''' 
clfArray = []
clfResultsTrain = []
clfResultsTest = []
i = 0;
for numNeuroneCache in [int(1.4**x) for x in range(2,20)]:
    localClfArray = []
    localClfResultsTest = []
    localClfResultsTrain = []
    for iterateur in range (3): 
        print("Entrainement, avec C=", numNeuroneCache)
        #print("Itération numéro : ", iterateur)
        localClfArray.append(MLPClassifier(hidden_layer_sizes=numNeuroneCache,max_iter=500, activation='tanh', solver='sgd', batch_size=1, alpha=0, learning_rate='adaptive', verbose=0))
        localClfArray[iterateur].fit(X_dataPiTrain,y_dataPiTrain)
        localClfResultsTest.append(localClfArray[iterateur].score(X_dataPiTest,y_dataPiTest))
        localClfResultsTrain.append(localClfArray[iterateur].score(X_dataPiTrain,y_dataPiTrain))
    clfArray.append(localClfArray)
    clfResultsTest.append(localClfResultsTest)
    clfResultsTrain.append(localClfResultsTrain)
    i += 1
'''


# # Arbre de décision
# 
# ## Nous utilisons le paramètre class_weight = balanced afin de chercher à améliorer le score des classes minoritaires.

# In[12]:


from sklearn.tree import DecisionTreeClassifier
TreesArray = []
TreesResultsTrain = []
TreesResultsTest = []
TreesBalancedResultsTrain = []
TreesBalancedResultsTest = []

for max_depth in [int(1.4**x) for x in range(2,15)]:
    localTreesArray = []
    localTreesResultsTest = []
    localTreesResultsTrain = []
    localTreesBalancedResultsTest = []
    localTreesBalancedResultsTrain = []
    for iterateur in range (10): 
        #print("Entrainement, avec max_depth=", max_depth)
        #print("Itération numéro : ", iterateur)
        localTreesArray.append(DecisionTreeClassifier(max_depth=max_depth,class_weight='balanced'))
        localTreesArray[iterateur].fit(X_dataPiTrain,y_dataPiTrain)
        localTreesResultsTest.append(localTreesArray[iterateur].score(X_dataPiTest,y_dataPiTest))
        localTreesResultsTrain.append(localTreesArray[iterateur].score(X_dataPiTrain,y_dataPiTrain))
        localTreesBalancedResultsTest.append(balanced_accuracy_score(y_dataPiTest, localTreesArray[iterateur].predict(X_dataPiTest)))
        localTreesBalancedResultsTrain.append(balanced_accuracy_score(y_dataPiTrain, localTreesArray[iterateur].predict(X_dataPiTrain)))
    TreesArray.append(localTreesArray)
    TreesResultsTest.append(localTreesResultsTest)
    TreesResultsTrain.append(localTreesResultsTrain)
    TreesBalancedResultsTest.append(localTreesBalancedResultsTest)
    TreesBalancedResultsTrain.append(localTreesBalancedResultsTrain)


# ## Ici nous allons traçer une figure qui prendra en compte la métrique de score habituelle ainsi que la métrique Balanced

# In[13]:


plt.figure(figsize=(15,7))
plt.title("Influence de la profondeur maximale le taux de reconnaissance & balanced (10 itérations)")
plt.ylabel('score ou score balanced')
plt.xlabel('Profondeur maximale des arbres')
plt.annotate('Profondeur maximale ='+ str(int(1.4**([np.mean(result) for result in TreesResultsTest].index(max([np.mean(result) for result in TreesResultsTest]))+2)))+', score moyen sur base test = ' + str(max([np.mean(result) for result in TreesResultsTest])), xy= (int(1.4**([np.mean(result) for result in TreesResultsTest].index(max([np.mean(result) for result in TreesResultsTest]))+2)),max([np.mean(result) for result in TreesResultsTest])), xytext=( 15,0.85) ,arrowprops=dict(facecolor='black', shrink=0.05),)
plt.annotate('Profondeur maximale ='+ str(int(1.4**([np.mean(result) for result in TreesBalancedResultsTest].index(max([np.mean(result) for result in TreesBalancedResultsTest]))+2)))+', score moyen sur base test = ' + str(max([np.mean(result) for result in TreesBalancedResultsTest])), xy= (int(1.4**([np.mean(result) for result in TreesBalancedResultsTest].index(max([np.mean(result) for result in TreesBalancedResultsTest]))+2)),max([np.mean(result) for result in TreesBalancedResultsTest])), xytext=( 15,0.5) ,arrowprops=dict(facecolor='black', shrink=0.05),)
plt.plot([int(1.4**x) for x in range(2,15)],[max(result) for result in TreesResultsTest],label ="Resultats max données test", color="green", linestyle =':')
plt.plot([int(1.4**x) for x in range(2,15)],[np.mean(result) for result in TreesResultsTest],label ="Resultats moyens données test", color="green")

plt.plot([int(1.4**x) for x in range(2,15)],[max(result) for result in TreesResultsTrain],label ="Resultats max données train",color="blue", linestyle =':')
plt.plot([int(1.4**x) for x in range(2,15)],[np.mean(result) for result in TreesResultsTrain],label ="Resultats moyens données train",color="blue")

plt.plot([int(1.4**x) for x in range(2,15)],[max(result) for result in TreesBalancedResultsTest],label ="Resultats Balanced max données test",color="red", linestyle =':')
plt.plot([int(1.4**x) for x in range(2,15)],[np.mean(result) for result in TreesBalancedResultsTest],label ="Resultats Balancedmoyens données test",color="red")

plt.plot([int(1.4**x) for x in range(2,15)],[max(result) for result in TreesBalancedResultsTrain],label ="Resultats Balanced max données train",color="pink", linestyle =':')
plt.plot([int(1.4**x) for x in range(2,15)],[np.mean(result) for result in TreesBalancedResultsTrain],label ="Resultats Balanced moyens données train",color="pink")


#plt.xscale('log')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.5, fontsize = 'large');


# In[14]:


bestTree = DecisionTreeClassifier(max_depth=7,class_weight='balanced').fit(X_dataPiTrain,y_dataPiTrain)
print(classification_report(y_dataPiTest,bestTree.predict(X_dataPiTest),zero_division=0))


# In[15]:


arrayScores = []
for i in range (10):
    arbre = DecisionTreeClassifier(class_weight='balanced').fit(X_dataPiTrain,y_dataPiTrain)
    arrayScores.append(balanced_accuracy_score(y_dataPiTest, arbre.predict(X_dataPiTest)))
print ("Le score balanced optenu par un modèle d'arbre sans limite de profondeur est de: ", np.mean(arrayScores))


# ## Le score balanced d'environ 32% constitue déjà une grande amélioration par rapport aux 15% du MLP.

# In[16]:


matriceConfusion =metrics.confusion_matrix(y_dataPiTest, arbre.predict(X_dataPiTest))
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion de la classification par arbre de décision");


# # Forêt

# In[17]:


from sklearn.ensemble import RandomForestClassifier
arrayScores = []
for i in range (10):
    arbre = RandomForestClassifier(n_estimators=25,class_weight='balanced').fit(X_dataPiTrain,y_dataPiTrain)
    arrayScores.append(balanced_accuracy_score(y_dataPiTest, arbre.predict(X_dataPiTest)))
print ("Le score moyen optenu par un modèle d'arbre sans limite de profondeur est de: ", np.mean(arrayScores))


# ### Après plusieurs essais avec les forêts je ne trouve rien de concluant, l'arbre semble faire mieux à chaque fois.

# In[18]:


from sklearn.svm import SVC
arrayScores = []
for i in range (10):
    mdl = SVC(C=100,gamma='auto',class_weight='balanced').fit(X_dataPiTrain,y_dataPiTrain)
    arrayScores.append(balanced_accuracy_score(y_dataPiTest, mdl.predict(X_dataPiTest)))
print ("Le score moyen optenu par un modèle d'arbre sans limite de profondeur est de: ", np.mean(arrayScores))


# In[19]:


metrics.confusion_matrix(y_dataPiTest, mdl.predict(X_dataPiTest))


# ## SVC semble tout classer en neutre

# # Nous allons simporter la bibliotèhèque 
# # !pip install imbalanced-learn
# 
# 

# In[20]:


from imblearn.over_sampling import SMOTE


# ## Méthodes pour équilibrer la classification, => avoir un recall maximal pour les classes monoritaires tout en essayant de garder une grande précision pour la classe majoritaire

# # Méthodes avec la bibliothèque imblearn
# 
# #### https://imbalanced-learn.org/stable/under_sampling.html#
# 
# ## I) weighted classes
# ## II) Undersampling
# - ### naïve random undersampling
# - ### Near miss v1
# - ### Near miss v2
# - ### Near miss v3
# - ### Condensed Nearest Neighbour
# 
# ## III) Oversampling
# ## IV) combination
# ## V) autre
# 

# ![title](grid_search_workflow.png)
# ## source : https://scikit-learn.org/stable/modules/cross_validation.html

# # Fonction qui renvoit un résumé de la performance des différentes méthodes de resampling des données en fonction d'un modèle.

# In[21]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek



def rapportDeResampling(X_train, y_train,X_test,y_test, model, targetLabels = None):

    
    #Si il n'y a pas de labels fournis, on les récupère directement sur les données:
    nbClass = len([a for a in np.sort(y_train.unique())])
    if targetLabels  == None:
        targetLabels = [a for a in np.sort(y_train.unique())]
    
    ResamplingMethods = [RandomUnderSampler(),NearMiss(version=1),NearMiss(version=2),NearMiss(version=3),CondensedNearestNeighbour(),EditedNearestNeighbours(),RepeatedEditedNearestNeighbours(),RandomOverSampler(),SMOTE(),ADASYN(),SMOTEENN(),SMOTETomek()]
    
    
    
    ResamplingMethodsDicArray = [
        {
            
            "name": "Random Under Sampler",
            "family": "Under sampling",
            "method": RandomUnderSampler()

        },
        {
            "name": "Near Miss V1",
            "family": "Under sampling",
            "method": NearMiss(version=1)
        },
        {
            "name": "Near Miss V2",
            "family": "Under sampling",
            "method": NearMiss(version=2)
        },
        {
            "name": "Near Miss V3",
            "family": "Under sampling",
            "method": NearMiss(version=3)
        },
        {
            "name": "Condensed Nearest Neighbour",
            "family": "Under sampling",
            "method": CondensedNearestNeighbour()
        },
        {
            "name": "Edited Nearest Neighbours",
            "family": "Under sampling",
            "method": EditedNearestNeighbours()
        },
        {
            "name": "Repeated Edited Nearest Neighbours",
            "family": "Under sampling",
            "method": RepeatedEditedNearestNeighbours()
        },
        {
            "name": "Random Over Sampler",
            "family": "Over sampling",
            "method": RandomOverSampler()
        },
        {
            "name": "SMOTE",
            "family": "Over sampling",
            "method": SMOTE()
        },
        {
            "name": "ADASYN",
            "family": "Over sampling",
            "method": ADASYN()
        },
        {
            "name": "SMOTEENN",
            "family": "Combined",
            "method": SMOTEENN()
        },
        {
            "name": "SMOTETomek",
            "family": "Combined",
            "method": SMOTETomek()
        }    
    ]
 
    
    lines = ((len(ResamplingMethodsDicArray)+1)//2) + (len(ResamplingMethodsDicArray)+1)%2

    xyPlot = range(2,(len(ResamplingMethodsDicArray)+1) + 1)
    
    fig = plt.figure(1,figsize=(15,20))

    ax = fig.add_subplot(lines,2,1)
    
        
    model.fit(X_train, y_train)
    clr = classification_report(y_test,model.predict(X_test),zero_division=0,output_dict=True)
 
    ax.set_title( "Répartission initiale des données de la base"+"\n"+"score global="+ str(model.score(X_test,y_test))+"\n"+"précision globale="+ str(clr['weighted avg']['precision'])+"\n"+"précision moyenne des classes="+ str(clr['macro avg']['precision'])+"\n"+"Rappel moyen des classes="+ str(clr['macro avg']['recall']))

    ax.pie([ y_train[y_train == a].shape[0] for a in np.sort(y_train.unique())], autopct='%1.1f%%',labels=targetLabels)
    for p in range(len(ResamplingMethodsDicArray)):
        
        X_resampled, y_resampled = ResamplingMethodsDicArray[p]["method"].fit_resample(X_train,y_train)   
            #certaines méthodes peuvent enlever tous les exemples d'une calsse il faut donc revoir le label
        if len([a for a in np.sort(y_resampled.unique())])  != nbClass:
            LocalTargetLabel = [a for a in np.sort(y_resampled.unique())]
        else:
            LocalTargetLabel = targetLabels
           
        
        model.fit(X_resampled, y_resampled)
        clr = classification_report(y_test,model.predict(X_test),zero_division=0,output_dict=True)
        
        ResamplingMethodsDicArray[p]['GlobalAccuracy'] = model.score(X_test,y_test)
        ResamplingMethodsDicArray[p]['GlobalPrecision'] = clr['weighted avg']['precision']
        ResamplingMethodsDicArray[p]['MeanClassPrecision'] = clr['macro avg']['precision']
        ResamplingMethodsDicArray[p]['MeanClassRecall'] = clr['macro avg']['recall']
        
        ax = fig.add_subplot(lines,2,xyPlot[p])
        ax.set_title( ResamplingMethodsDicArray[p]["name"]+ ", méthode de type: "+ ResamplingMethodsDicArray[p]["family"]+"\n"+"score global="+ str(ResamplingMethodsDicArray[p]['GlobalAccuracy'])+"\n"+"précision globale="+ str(ResamplingMethodsDicArray[p]['GlobalPrecision'])+"\n"+"précision moyenne des classes="+ str(ResamplingMethodsDicArray[p]['MeanClassPrecision'])+"\n"+"Rappel moyen des classes="+ str(ResamplingMethodsDicArray[p]['MeanClassRecall']))
        
        ax.pie([ y_resampled[y_resampled == a].shape[0] for a in np.sort(y_resampled.unique())], autopct='%1.1f%%',labels=LocalTargetLabel)
    plt.tight_layout(pad=2) 
    plt.show()


# In[22]:


rapportDeResampling(X_dataPiTrain, y_dataPiTrain,X_dataPiTest, y_dataPiTest,DecisionTreeClassifier(class_weight='balanced'),emotions.values() )


# In[23]:


import math

def dist(point1, point2, df):
    point1, point2 = str(point1), str(point2)
    return  (  (df[' x_' + point2] - df[' x_' + point1])**(2) +  (df[' y_' + point2] - df[' y_' + point1])**(2)   )**(0.5)


# In[32]:


X_AC = X_dataPiTrain.copy()
for i in range(0,68):
    for j in range(i,68): # afin de ne pas recalculer deux fois la meme distance
        if (i != j): # on ne calcule pas la distance d'un point avec lui même
            X_AC['dist'+str(i)+"_"+str(j)]  = dist(i, j, X_dataPiTrain)
            
X_AC.drop(X_AC.iloc[:, 0:136], inplace = True, axis = 1) 


# In[33]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=100)
bestfeatures = bestfeatures.fit(X_AC,y_dataPiTrain)
dfscores = pd.DataFrame(bestfeatures.scores_)
dfcolumns = pd.DataFrame(X_AC.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
featureScores.sort_values("Score",ascending=False, inplace = True) 
featureScores[0:20]


# In[34]:


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
pca.fit(X_AC)

XPCA = pca.transform(X_AC)

XPCADf = pd.DataFrame(data = XPCA, columns = ['pc1', 'pc2', 'pc3'])
colors = {0:'red', 1:'green', 3:'blue', 4:'pink',5:'yellow', 6:'cyan', 7:'purple'}
fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(XPCADf.pc1, XPCADf.pc2, XPCADf.pc3, c=y_dataPiTrain.map(colors),alpha=0.5);


# In[35]:


X_AC


# In[36]:


X_new_ft = X_dataPi.copy()
for i in range(0,68):
    for j in range(i,68): # afin de ne pas recalculer deux fois la meme distance
        if (i != j): # on ne calcule pas la distance d'un point avec lui même
            X_new_ft['dist'+str(i)+"_"+str(j)]  = dist(i, j, X_dataPi)
            
X_new_ft.drop(X_new_ft.iloc[:, 0:136], inplace = True, axis = 1) 

X_nf_train, X_nf_test, y_dataPiTrain, y_dataPiTest = model_selection.train_test_split(X_new_ft, y_dataPi, train_size=0.75, test_size=0.25,shuffle=False)


# In[37]:


knnModels = []
KnnResultsTest = []
KnnResultsTrain = []

knnModel = KNeighborsClassifier(n_neighbors=2, algorithm='brute')
knnModel.fit(X_nf_train,y_dataPiTrain)
print(knnModel.score(X_nf_test,y_dataPiTest))

yPredicted=knnModel.predict(X_nf_test)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[42]:


clf = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=500, activation='tanh', solver='sgd', batch_size=1, alpha=0, learning_rate='adaptive', verbose=0)
clf.fit(X_nf_train,y_dataPiTrain)
print(clf.score(X_nf_test,y_dataPiTest))
yPredicted=clf.predict(X_nf_test)
matriceConfusion = metrics.confusion_matrix(y_dataPiTest,yPredicted)
sn.heatmap(matriceConfusion, annot=True,cmap="OrRd",vmax = 5,xticklabels = emotions.values(),yticklabels = emotions.values());
plt.title("Matrice de confusion du modèle naïf");


# In[43]:


rapportDeResampling(X_nf_train, y_dataPiTrain,X_nf_test, y_dataPiTest,DecisionTreeClassifier(class_weight='balanced'),emotions.values() )


# In[ ]:




