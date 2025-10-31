import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
data.drop(['Unnamed: 32', 'id'], inplace=True, axis=1)
data = data.rename(columns={"diagnosis": "target"})

sns.countplot(data["target"])
#plt.show()  # 👈 sadece grafiği gösterir

print(data.target.value_counts())  # 👈 hedef değişkenin dağılımını yazdırır

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

print(len(data))
print(data.head())
print("Data Shape",data.shape)
data.info()

describe=data.describe()

"""
standardization
missing value: none
"""




# %% EDA

#Correlation
corr_matrix=data.corr()
sns.clustermap(corr_matrix, annot=True, fmt=".2f")
plt.title("Correlation Between Features")
#plt.show()

#
threshold = 0.75
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()

sns.clustermap(data[corr_features].corr(), annot=True, fmt=".2f")
plt.title("Correlation Between Features w Corr Threshold 0.75")
#plt.show()

"""
there are some correlated features

"""

# box plot
data_melted=pd.melt(data,id_vars="target",
                    var_name="features",
                    value_name="value")
plt.figure()
sns.boxplot(x="features",y="value",hue= "target", data=data_melted)
plt.xticks(rotation=90)
#plt.show()

"""
standardization-normalization
"""

# pair plot
sns.pairplot(data[corr_features],diag_kind="kde",markers="+",hue="target")
#plt.show()

"""
skewness
"""

# %% outlier
y=data.target
x=data.drop(["target"], axis=1)
columns= x.columns.tolist()

clf=LocalOutlierFactor()
y_pred=clf.fit_predict(x)
x_score=clf.negative_outlier_factor_

outlier_score=pd.DataFrame()
outlier_score["score"]=x_score

#threshold
threshold= -2.5
filtre=outlier_score["score"]<threshold
outlier_index=outlier_score[filtre].index.tolist()
plt.figure()

plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color="blue",s=50,label="Outliers")

plt.scatter(x.iloc[:,0],x.iloc[:,1], color="k", s=3, label= "Data Points" )

radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r", facecolors="none",label="Outlier_Score")
plt.legend()
plt.show()

#drop outliers
x=x.drop(outlier_index)
y=y.drop(outlier_index).values


# %% Train Test Split
test_size=0.3
X_train, X_test,Y_train,Y_test=train_test_split(x, y,test_size=test_size,random_state=42)

# %%

scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train_df=pd.DataFrame(X_train,columns=columns)
X_train_df_describe=X_train_df.describe()
X_train_df["target"]=Y_train


# box plot
data_melted=pd.melt(X_train_df,id_vars="target",
                    var_name="features",
                    value_name="value")

plt.figure()
sns.boxplot(x="features",y="value",hue= "target", data=data_melted)
plt.xticks(rotation=90)
#plt.show()

# pair plot
sns.pairplot(X_train_df[corr_features],diag_kind="kde",markers="+",hue="target")
#plt.show()


# %% Basic KNN Method
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
y_pred= knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc=accuracy_score(Y_test, y_pred)
score=knn.score(X_test, Y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)

# %% Choose Best Parameters

def KNN_Best_Params(x_train,x_test,y_train,y_test):
    k_range=list(range(1,31))
    weight_options=["uniform","distance"]
    print()
    param_grid=dict(n_neighbors=k_range,weights=weight_options)
    knn=KNeighborsClassifier()
    grid=GridSearchCV(knn,param_grid,cv=10,scoring="accuracy")
    grid.fit(x_train,y_train)
    print("Best training parameters: {}".format(grid.best_params_))

    print()
    
    knn=KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    y_pred_test=knn.predict(x_test)
    y_pred_train=knn.predict(x_train)
    
    cm_test= confusion_matrix(y_test, y_pred_test)
    cm_train= confusion_matrix(y_train, y_pred_train)
    
    acc_test=accuracy_score(y_test, y_pred_test)
    acc_train=accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ", cm_test)
    print("CM Test: ", cm_train)    
    
    return grid

grid=KNN_Best_Params(X_train, X_test,Y_train,Y_test)


"""
v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
"""
    
 # %%
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# PCA uygulama
pca = PCA(n_components=2)
X_reduced_pca = pca.fit_transform(x_scaled)

# DataFrame oluşturma
pca_data = pd.DataFrame(X_reduced_pca, columns=["p1", "p2"])
pca_data["target"] = y

# Görselleştirme
sns.scatterplot(x="p1", y="p2", hue="target", data=pca_data, palette="Set1")
plt.title("PCA: p1 vs p2")
plt.show()

# Visualize
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

grid.fit(X_reduced_pca, y)

# Görselleştirme
h = .05
x_min, x_max = X_reduced_pca[:, 0].min() - 1, X_reduced_pca[:, 0].max() + 1
y_min, y_max = X_reduced_pca[:, 1].min() - 1, X_reduced_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

plt.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=y, cmap=cmap_bold,
            edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k=%i, weights='%s')" %
          (len(np.unique(y)), grid.best_estimator_.n_neighbors, grid.best_estimator_.weights))
plt.show()  

#%% NCA

nca=NeighborhoodComponentsAnalysis(n_components=2,random_state=42)
nca.fit(x_scaled,y)
X_reduced_nca=nca.transform(x_scaled)
nca_data=pd.DataFrame(X_reduced_nca, columns=["p1","p2"])
nca_data["target"]=y
sns.scatterplot(
    x="p1", y="p2", 
    hue="target", 
    data=nca_data,
    palette="Set1",
    s=60,          # nokta boyutunu artır
    edgecolor=None # siyah kenarlıkları kaldır
)
plt.title("NCA: p1 vs p2")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Target", loc="best")
plt.show()

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(
    X_reduced_nca, y, test_size=0.3, random_state=42
)
grid_nca=KNN_Best_Params(X_train_nca, X_test_nca,Y_train_nca,Y_test_nca)


# Görselleştirme
h = .2
x_min, x_max = X_reduced_nca[:, 0].min() - 1, X_reduced_nca[:, 0].max() + 1
y_min, y_max = X_reduced_nca[:, 1].min() - 1, X_reduced_nca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

plt.scatter(X_reduced_nca[:, 0], X_reduced_nca[:, 1], c=y, cmap=cmap_bold,
            edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k=%i, weights='%s')" %
          (len(np.unique(y)), grid.best_estimator_.n_neighbors, grid.best_estimator_.weights))
plt.show()  

#%% Find Wrong Decision
knn= KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca, Y_train_nca)
y_pred_nca=knn.predict(X_test_nca)
acc_test_nca= accuracy_score(y_pred_nca, Y_test_nca)
knn.score(X_test_nca, Y_test_nca)

test_data=pd.DataFrame()
test_data["X_test_nca_p1"]= X_test_nca[:,0]
test_data["X_test_nca_p2"]=  X_test_nca[:,1]
test_data["y_pred_nca"]=y_pred_nca
test_data["y_test_nca"]=Y_test_nca

plt.figure()
sns.scatterplot(
    x="X_test_nca_p1",
    y="X_test_nca_p2",
    hue="y_test_nca",
    data=test_data,
    palette="coolwarm",
    alpha=0.7
)

diff=np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(
    test_data.iloc[diff, 0],
    test_data.iloc[diff, 1],
    color="red",
    label="Wrong Classified",
    edgecolors="k",
    s=100
)
