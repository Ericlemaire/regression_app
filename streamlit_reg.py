#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:10:18 2023

@author: lemaireeric
"""

# L'application ici designée doit pouvoir servir à l'accomplissement de tâche de regression
# avec des jeux de donnéesdifférents. Elle doit permettre l'usage de plusieurs modèles, et leur réglage.
# Le but de l'application n'est pas de trasnformer les données. La préparation des données doit se faire en amont.


# Titre de l'application
import streamlit as st


st.set_page_config(layout="wide")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import xgboost as xgb
from xgboost import XGBRegressor
import shap
import plotly.express as px
from streamlit_shap import st_shap
from PIL import Image



st.title("XGBoost pour la Régression")


st.sidebar.image("logoTT.png", width=150)

st.subheader("Introduction")
"""
   Cette application est destinée à permettre à tout utilisateur de charger des données ous format csv. Le jeu de données 
   doit être préallablement nettoyé, encodé afin de pouvoir 
   régler et entraîner des modèles de régression avec la librairie XGBoost. Une fois les données chargées, l'utilisation peut sélectionner les variables qui lui paraissent 
   pertinentes pour son problème. Lorsqu'il pense avoir trouvé un modèle intéressant, ou meême au cours du processus de sélection du modèle, il peut découvrir comment 
   les prédictions sont effectuées grâce à l'usage de la librairie SHAP. 
   En fin de parcours, il peut sauvegarder son modèle et le charger sur son ordinateur afin de pouvoir le réutiliser aillers.'
    """
   
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mode d'emploi")
"""
Voici les grandes étapes à suivre dans l'utilisation de cette application: '
 1)Charger les données
 2) Afficher les données (optionnel pour l'utilisateur')
 3) Séparer X et y de façon sélective. et afficher le réusltat.
 4) Création d'un jeu d'entraînement et d'un jeu de test (possibilité de régler la proportion
 5) Standisation (multiples options)
 6) Sélection de modèles
 7) Sélection et réglages des hyperparamètres
 8) Entraînement avec validation croiée (possibilité de choisir le nombre de split)
 9) Affichages des scores.
 10) Graphique des résidus (histogramme ou nuage de points)
 11) Interprétation vis avec multiplies graphiques.
    """

with col2: 
    st.image("icone.jpeg", width = 250)




# 1) Chargement des données
uploaded_file = st.file_uploader(
    "Télécharger un fichier de données (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)


    
    # 2) Affichage des données (optionnel)
    if st.sidebar.checkbox("Afficher les données"):
        st.dataframe(data)

    # 3) Séparation X et y
    X_col = st.sidebar.multiselect("Choisir les colonnes pour X", data.columns)
    y_col = st.sidebar.selectbox("Choisir la colonne pour y", data.columns)
    X = data[X_col]
    y = data[y_col]

with st.expander("X (Caractéristiques):"):
    if uploaded_file is not None:
        st.write(X)
    else:
    
        st.write("Chargez des données pour commencer la modélisation")

with st.expander("Variable cible"):
    if uploaded_file is not None:
        st.write(y)
    else: 
        st.write("Chargez des données pour commencer la modélisation")


    # 4) Création d'un jeu d'entraînement et d'un jeu de test
@st.cache_data 
def Split(X, y, test_size): 
    X_train, X_test, y_train, y_test=train_test_split(
        X, y, test_size = test_size, random_state = 42)
    return X_train, X_test, y_train, y_test

test_size=st.sidebar.slider("Proportion du jeu de test", 0.1, 0.5, 0.2)

S = st.sidebar.checkbox("Je crée je jeu d'entraînement")



if S: 
    Splittage = Split(X,y,test_size)
else: 
    st.write("Créer un jeu d'entraînement pour commencer la modélisation")



X_train = Splittage[0]
X_test = Splittage[1]
y_train = Splittage[2] 
y_test = Splittage[3] 

with st.expander("X_train"):
     st.write(X_train.shape)
     st.write(X_train)

with st.expander("X_test"):
    st.write(X_test.shape)
    st.write(X_test)


# 5) Standarisation
standardize=st.sidebar.checkbox("Standardiser les données")
if standardize:
    scaler=StandardScaler()
    SX_train=scaler.fit_transform(X_train)
    SX_test=scaler.transform(X_test)
    with st.expander("Je regarde les données standardisées"):
        st.dataframe(SX_train)

# Tutoriel sur XGBoost : 



# Slider permettant de régler le nombre de Folds (Plis): 
plis = st.sidebar.slider(label= "Nombre de plis pour la validation croisée", 
                         min_value= 0, 
                         max_value= 10, 
                         value= 5, 
                         step=1 
                         )

train_size0 = st.sidebar.slider(label="partition du jeu d'entraînement", 
                                min_value=0.1,
                                max_value=1.0, 
                                value=0.8, 
                                step=0.05
    )

# Réglage du modèle XGBoost : 
# AJOUTER un lien vers la documentation de l'algorithme. 

st.sidebar.header('Réglages du modèle')
num_estimators = st.sidebar.slider('Nombre d\'estimateurs', min_value=100, max_value=3000, value=1500, step=100)
max_depth = st.sidebar.slider('Profondeur maximale de l\'arbre', min_value=1, max_value=5, value=2, step=1)
learning_rate = st.sidebar.slider('Taux d\'apprentissage', min_value=0.001, max_value=1.0, value=0.1, step=0.001)
Subsample = st.sidebar.slider('Sous-ensemble de colonnes', min_value=0.1, max_value=1.0, value=0.5, step=0.05)
reg_lambda = st.sidebar.slider('Regularisation Lambda', min_value=0.0, max_value=10.0**3, value=0.5, step=0.1)
min_child_weight = st.sidebar.slider('Min_child_weight', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
colsample_bytree = st.sidebar.slider('Colsamplebytree', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
min_split_loss = st.sidebar.slider('Min_split_loss', min_value=0.0, max_value=100.0, value=0.0, step=0.5)
reg_alpha = st.sidebar.slider("Reg_alpha", min_value=0.0 , max_value=10.0**3, value =0.0, step =1.0 )


# REMARQUES : 
# METTRE LE LIEN VERS LA DOCUMENTATION DES HYPERPARAMÈTRES.
# On pourrait classer les hyperparamètres par fonctions. 
# On pourrait également ajouter une note (i) afin d'aiguiller le réglage. 









# Dictionnaire d'hyperparamètres : 

param = {
    'n_estimators':[num_estimators],
    'max_depth': [max_depth], 
    'learning_rate':[learning_rate], 
    'subsample': [Subsample], 
    'reg_lambda': [reg_lambda], 
    'reg_alpha': [reg_alpha],
    'min_child_weight': [min_child_weight],
    'colsample_bytree': [colsample_bytree], 
    'min_split_loss': [min_split_loss] 
    }


from sklearn.model_selection import ShuffleSplit


cv = ShuffleSplit(n_splits = plis, 
                  train_size = train_size0, 
                  random_state=0
                  )


# Fonction d'entraînement alternative  avec GridSearchCV : 

from sklearn.model_selection import GridSearchCV
#resultats = []

@st.cache_data
def model_training_grid(_model, X, y_train, param, _cv, random_state=0): 
    
    #scoring = {"R2": "r2", "MSE": "neg_mean_squared_error"}
        # instanciation de la grille 
    grid = GridSearchCV(_model, param, cv = _cv, scoring = "r2", refit = True, n_jobs = -1)
        # Entrainement
    grid.fit(X, y_train)
    y_pred = grid.predict(X)
#   resultats.append(grid.cv_results_)
    resultats = grid.cv_results_
    
    residus = y_train - y_pred
    param_m = grid.get_params(deep=True)
    
    st.metric(label = "Score r2" , value = grid.score(X, y_train).round(2))

    
    return resultats, y_pred, residus, param_m




st.subheader("Résultats")

xgb1 = XGBRegressor()

modele = model_training_grid(xgb1, SX_train, y_train, param, cv)

# predictions = model_training_grid(xgb1, SX_train, y_train, param, cv)[1]
# residus = model_training_grid(xgb1, SX_train, y_train, param, cv)[2]
# param_m = model_training_grid(xgb1, SX_train, y_train, param, cv)[3]


# Création et affichage du dataframe résumant l'entraînement : 
resultats = pd.DataFrame(modele[0])
st.dataframe(resultats)
  

           
# Entraînement du modèle : 
    
@st.cache_data
def model_training(SX_train, y_train, num_estimators,max_depth, learning_rate, subsample, reg_lambda, reg_alpha, min_child_weight, colsample_bytree, min_split_loss):
    
    xgb = XGBRegressor(
            n_estimators=num_estimators, 
            max_depth=max_depth, 
            learning_rate = learning_rate, 
            subsample = Subsample,
            reg_lambda = reg_lambda,
            reg_alpha = reg_alpha, 
            min_child_weight = min_child_weight,
            colsample_bytree = colsample_bytree,
            min_split_loss =min_split_loss,
            random_state = 0
                     )

    model = xgb.fit(SX_train, y_train)
    y_pred = xgb.predict(SX_train)
    residus = y_train - y_pred
    scores = xgb.score(SX_train,y_train)
    st.metric(label = "Score r2" , value = scores.round(3))
        
    return model, y_pred, residus, scores

#Go = st.sidebar.button("Lancement de l'entraînement")
#if Go:
#modele = model_training(SX_train, y_train, num_estimators,max_depth, learning_rate, Subsample, reg_lambda, reg_alpha, min_child_weight, colsample_bytree, min_split_loss)
        
# Représentation graphique  et dataframe des résidus : 
    
with st.expander("Dataframe avec prédiction et résidus + historgramme des résidus"):
    st.write(X_train.shape)
 
    X_train['y_train']= y_train 
    X_train["Résidus"]=modele[2]
    X_train["Prédictions"]= modele[1]
    st.dataframe(X_train)
    fig = px.histogram(X_train["Résidus"])
    st.plotly_chart(fig)
  
      
#col1, col2 = st.columns(2)
 
#with col1: 
#    import plotly.graph_objects as go

st.subheader('Diagramme des résidus')
import plotly.graph_objects as go
fig = px.scatter(X_train, x='y_train', y="Prédictions", hover_name= X_train.index) 

# Ajouter une ligne à votre nuage de points
fig.add_trace(go.Scatter(x=X_train["y_train"], 
                         y=X_train["y_train"],
                         mode='lines', 
                         name='Valeurs réèlles',
                         line=dict(
                             color='Red',
                             width=2
                             )))

    
#fig.add_trace(go.scatter(x=X_train['y_train'], y=X_train['y_train'],
 #                   mode='markers', name='markers'))
st.plotly_chart(fig)
       
# Tuto Shap : 
#titre_tuto = st.sidebar.subheader("Tutoriel pour comprendre les valeurs de Shapley")
#st.subheader("Tutoriel pour comprendre les valeurs de Shapley")

#tuto_shap = st.sidebar.video("https://youtu.be/VB9uV-x0gtg")
#st.video("https://youtu.be/UUZxRct8rIk")

#st.write(np.version.version)

shap.initjs()


@st.cache_data 
def shap_vals(_model,X): 
    explainer = shap.Explainer(_model)
    shap_values = explainer(X)
    #sf_shap = pd.DataFrame(data=shap_values.values,columns = list(X.columns))
    return explainer, shap_values #, df_shap

#st.write(SX_train)
#st.write(SX_train.shape)
X_train2 = X_train.iloc[0: , 0:-3]
#st.write(X_train2.shape)

SX_train = pd.DataFrame(SX_train, columns = list(X_train2.columns))

explainer = shap_vals(modele[0], SX_train)[0]
shap_values = shap_vals(modele[0], SX_train)[1]
#df_shap = shap_vals(modele[0], SX_train)[2]

#st.write(shap_values.values)
#st.write(list(X.columns))
df_shap = pd.DataFrame(shap_values.values, columns = list(X.columns))
with st.expander("df des valeurs de shapley"): 
    st.dataframe(df_shap)




# GRAHPIQUES D'INTERPRETATION 


st.subheader("Interprétation du modèle")

import matplotlib.pyplot as plt
shap.plots.initjs()
shap.initjs()



# GARPHIQUE 1: VUE SYNOPTIQUE 

with st.expander("Vue d'ensemble des prédictions"):
    # Force plot global 
   st_shap(shap.force_plot(explainer.expected_value, shap_values.values, SX_train), height=400, width=1000)



# GRAPHIQUE 2 : GLOBALE
with st.expander("Interpétation Globale du modèle 2 : le poids des variables"):
    # summarize the effects of all the features
    st_shap(shap.plots.beeswarm(shap_values))
    st_shap(shap.plots.bar(shap_values))


# GRAPHIQUE 4 : LE RÔLE DES VARIABLE (DEPENDANCE PLOT)
with st.expander("Influence relative des variables"):
    # create a dependence scatter plot to show the effect of a single feature across the whole dataset
    option_deplot = st.selectbox("Je sélectionne l'influence de':",options = X_train2.columns )
    option_deplot_2 = st.selectbox(" 'Avec quelle variable voulez-vous colorer les points ?':",options = X_train2.columns )
    
    st_shap(shap.plots.scatter(shap_values[:, option_deplot], color = shap_values[: , option_deplot_2]))

# GRAPHIQUE 4 : LOCALE 
    
with st.expander("Interprétation locale"):
    option_waterfall = st.selectbox('Je sélectionne:', options = range(len(X_train2.T.columns)), format_func = lambda i: X_train.index[i])
    
    st_shap(shap.plots.waterfall(shap_values[option_waterfall]))
    st_shap(shap.plots.force(shap_values[option_waterfall]))
    
    st.metric(label = "Valeur réèlle", value = data[y_col][option_waterfall])
    st.metric(label = "Erreur de prédiction", value = X_train["Résidus"][option_waterfall].round(2))
                
                
                
                
                
                
                
                
                
                
# Entraîner le modèle sur le jeu de test.
if st.sidebar.button("Evaluation du modèle sur le jeu de test"):
   scores_test = modele.score(SX_test, y_test)
   st.metric(label = "Score sur le jeu de test", value = scores_test)
  
  
# Réentraîner le modèle sur l'ensemble du jeu de données et sauvergarde du modèle.
if st.sidebar.button("Je réentraîne le modèle sur l'ensemble du jeu de données."):
   sc = StandardScaler()
   X_sc = sc.fit_transform(X)
   modele = modele.fit(X,y)
   st.metric(labe = "Score sur le jeu de test", value = modele.score(X,y))



# SAUVEGARDE DU MODELE : 


nom = st.sidebar.text_input(label = "Je nomme et sauvegarde mon modèle", value = "")
path_model = st.sidebar.text_input(label = "/chemin/vers/votre/modele.pkl", value = "{nom}.pkl")
  
if nom:
   import joblib
# save
   joblib.dump(modele, "{nom}.pkl")
   joblib.dump(modele, path_model)


#st.download_button(label="Je télécharge le modèle",
#    data=,
#    file_name= "{nom}.pkl")
  
  
# Affichage de l'application
#if st.button("Réinitialiser"):
#   st.caching.clear_cache()

                
                
                
                
                
                
                
                
                
                
                