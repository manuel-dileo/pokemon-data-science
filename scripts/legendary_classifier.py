import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import math
from sklearn import preprocessing

def is_legendary(pokemon):
    df = pd.read_csv("../dataset/pokemon.csv")
    df.set_index('name',inplace=True)
    
    X = df
    y = np.array(df['is_legendary'])
    
    features = ['percentage_male','capture_rate','base_egg_steps']
    X_f = X.filter(features)
    X_f['percentage_male'] = list(map(lambda x: -1 if math.isnan(x) else x, X_f['percentage_male'].values))
    
    data = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(X_f.values))
    df_totrain = pd.DataFrame(data,columns=features,index=X_f.index)
    clf = tree.DecisionTreeClassifier()
    clf.fit(data,y)
    
    return clf.predict(np.asarray([df_totrain.loc[pokemon].values]))
    
def train(model_class, X, y, num_fold_cross_val, scaling=StandardScaler(), dim_reduction=None):
        
    X_std = scaling.fit_transform(X)  if scaling is not None else X
    
    X_std = dim_reduction.fit_transform(X_std) if dim_reduction is not None else X_std

    val = cross_val_score(model_class(), X_std, y, cv=num_fold_cross_val)
        
    return val.mean()

warnings.filterwarnings('ignore')

pokemon = input("Insert pokemon name\n")
print("legendary") if is_legendary(pokemon) else print("no-legendary")
