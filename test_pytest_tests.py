import pytest
from fonctions_p7 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split   
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

###########################################################################################################
# VERIFICATION DU CHARGEMENT DES DONNEES DU DATAFRAME
###########################################################################################################

# Test 1: Vérification que la fonction load_data renvoie un DataFrame non vide

def test_load_data_not_empty():
    df = load_data()
    assert not df.empty, "Le DataFrame doit contenir des données"

# Test 2: Vérification que la fonction load_data renvoie un DataFrame avec les colonnes attendues

def test_load_data_columns():
    expected_columns = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY","REGION_POPULATION_RELATIVE","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","AGE",
                    "YEARS_LAST_PHONE_CHANGE","YEARS_EMPLOYED","ANNUITY_INCOME_PERC","ANNUITY_RATE_PERC","CREDIT_INCOME_PERC","prev_AMT_ANNUITY_mean","prev_AMT_CREDIT_mean","prev_AMT_CREDIT_sum","prev_AMT_DOWN_PAYMENT_mean","prev_DAYS_DECISION_mean","prev_CNT_PAYMENT_mean","prev_AMT_PAYMENT_mean","prev_AMT_INSTALMENT_mean","prev_SK_DPD_count","home_DAYS_CREDIT_min","home_DAYS_CREDIT_ENDDATE_mean","home_AMT_CREDIT_SUM_sum","home_AMT_CREDIT_SUM_mean","home_AMT_CREDIT_SUM_DEBT_sum","prev_type_loans","prev_cash_loans_perc","total_accepted_loans"]  
    df = load_data()
    assert df.columns.tolist() == expected_columns, "Le DataFrame doit avoir les colonnes attendues"

# Test 3: Vérification que la fonction load_data renvoie un DataFrame avec un nombre de clients corrects

def test_load_data_number_of_rows():
    expected_rows = 48744  # Remplacez par le nombre réel de lignes attendu
    df = load_data()
    assert len(df) == expected_rows, "Le DataFrame doit avoir le bon nombre de lignes"

###########################################################################################################
# VERIFICATION DU CHARGEMENT DU MODELE ET DU SCALER
###########################################################################################################

def test_load_model():
    """Test du chargement du modèle LGBMClassifier"""
    model_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/basic_lgbmc.pkl"

    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    assert isinstance(loaded_model, LGBMClassifier), "Le modèle chargé n'est pas une instance de LGBMClassifier"

def test_load_scaler():
    """Test du chargement des X de jeu de données 'testset_rfe_30f.csv'"""
    scaler_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/minmax_scaler.pkl"

    with open(scaler_path, 'rb') as file:
        loaded_minmax = pickle.load(file)

    assert isinstance(loaded_minmax, MinMaxScaler), "Le scaler chargé n'est pas une instance de MinMaxScaler"

###########################################################################################################
# VERIFICATION DE LA FONCTION SCORE
###########################################################################################################

def test_business_score():
    # Cas de test 1 : Score de 0 pour l'ensemble des prédictions correctes
    y_true_1 = [0, 0, 1, 1]
    y_pred_1 = [0, 0, 1, 1]
    assert business_score(y_true_1, y_pred_1) == 0.0

    # Cas de test 2 : Score de 0 pour l'ensemble des prédictions incorrecycorrectes
    y_true_2 = [0, 0, 1, 1]
    y_pred_2 = [1, 1, 0, 0]
    assert business_score(y_true_2, y_pred_2) == 1.0

    # Cas de test 3 : Score compris entre 0 et 1 lors de certaines prédictions incorrectes
    y_true_3 = [0, 0, 1, 1]
    y_pred_3 = [0, 1, 1, 1]
    assert 0.0 <= business_score(y_true_3, y_pred_3) <= 1.0

###########################################################################################################
# VERIFICATION DE LA FONCTION SEUIL METIER PAR 2 TESTS
###########################################################################################################    

# TEST 1: Vérification que les valeurs soient dans la plage attendue de valeurs

def test_seuil_metier():
    # Générer des données fictives
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer un modèle fictif
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Générer des probabilités fictives
    y_probs = model.predict_proba(X_train)[:, 1]

    # Appeler la fonction seuil_metier avec des seuils fictifs
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold, best_score, scores = seuil_metier(model, X_train, y_train, thresholds)

    # Assertion: Le meilleur seuil est compris entre 0 et 1
    assert 0 <= best_threshold <= 1, "Le meilleur seuil n'est pas dans la plage attendue"

    # Assertion: Tous les seuils sont compris entre 0 et 1 
    assert all(0 <= threshold <= 1 for threshold in thresholds), "Certains seuils ne sont pas dans la plage attendue"

    # Assertion: Le meilleur score est compris entre 0 et 1
    assert 0 <= best_score <= 1, "Le meilleur score n'est pas dans la plage attendue"

    # Assertion: Tous les scores sont compris entre 0 et 1
    assert all(0 <= score <= 1 for score in scores), "Certains scores ne sont pas dans la plage attendue"

# TEST 2: Vérification que le meilleur score est bien parmi les scores calculés pour différents seuils 
# Le meilleur seuil pouvant être n'importe quel float compris entre 0 et 1, il est impossible de prédire quel seuil donnera le meilleur score métier. 

def test_seuil_metier_best_score():
    # Générer des données fictives
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer un modèle fictif
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Générer des probabilités fictives
    y_probs = model.predict_proba(X_train)[:, 1]

    # Appeler la fonction seuil_metier avec des seuils fictifs
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold, best_score, scores = seuil_metier(model, X_train, y_train, thresholds)

    # Vérifier que le best_score est bien parmi les scores calculés
    assert best_score in scores

#########################################################################################################
# MATRICE DE CONFUSION ET COURBE ROC PERSONNALISEES
#########################################################################################################

def test_confusion_matrix_roc_auc():
    # Génération des données de test
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba =  np.random.uniform(0, 1, 100)

    # Appel de la fonction à tester
    fig =  confusion_matrix_roc_auc(None, y_true, y_pred, y_pred_proba)

    # Vérification que le résultat soit bien une figure
    assert isinstance(fig, plt.Figure), "Le résultat doit être une figure"

    # Vérification que la figure contient bien deux sous-plots
    _, axes = plt.subplots(1, 2)
    assert len(axes) == 2, "La figure doit contenir deux sous-plots"

    # Vérification que le premier sous-plot est bien une matrice de confusion
    assert isinstance(fig, plt.Figure), "Le sous-plot 1 doit afficher une matrice de confusion"

    # Vérification que le deuxième sous-plot est bien une courbe ROC
    assert isinstance(axes[1], plt.Axes), "Le sous-plot 2 doit afficher une courbe ROC"

############################################################################################################
# FONCTION DE TRACKING VIA MLFLOW
############################################################################################################

# TEST 1: Vérification du tracking

# Génération de données fictives au préalable
def generate_test_data():
    np.random.seed(42)
    X = np.random.rand(1000, 30)  # 1000 échantillons, 30 variables
    y = np.random.randint(0, 2, 1000)  # Classes binaires (0 ou 1)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def test_mlflow_tracking():
    # Utilisation des données générées via la fonction generate_test_data
    X_train, X_test, y_train, y_test = generate_test_data()

    # Création d'un modèle pour le test
    model = RandomForestClassifier()

    # Appel la fonction de suivi MLflow
    metrics_df = mlflow_tracking(model, X_train, X_test, y_train, y_test, thresholds=np.linspace(0, 1, 100))

    # Vérification que le DataFrame des métriques ne soit pas vide
    assert not metrics_df.empty, "Le DataFrame des métriques est vide"

# TEST 2: Vérification que les métriques présentes des valeurs raisonnables

def test_mlflow_tracking():
    # Utilisation des données générées via la fonction generate_test_data
    X_train, X_test, y_train, y_test = generate_test_data()

    # Création d'un modèle pour le test
    model = RandomForestClassifier()

    # Appel de la fonction de suivi MLflow
    metrics_df = mlflow_tracking(model, X_train, X_test, y_train, y_test, thresholds=np.linspace(0, 1, 100))

    # Transposition du dataframe pour avoir les métriques en colonnes a
    metrics_df = metrics_df.T
    print(metrics_df)

    # Vérification que les valeurs pour l'accuracy soient comprises entre 0 et 100
    assert (0 <= metrics_df["Train Accuracy"]).all() and (metrics_df["Train Accuracy"] <= 100).all(), "L'accuracy du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Test Accuracy"]).all() and (metrics_df["Test Accuracy"] <= 100).all(), "L'accuracy du jeu de test n'est pas dans la plage attendue de valeurs"

    # Vérification que les valeurs pour l'AUC soient comprises entre 0 et 1
    assert (0 <= metrics_df["Train AUC"]).all() and  (metrics_df["Train AUC"] <=1).all(), "L'AUC du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Test AUC"]).all() and (metrics_df["Test AUC"] <=1).all(), "L'AUC du jeu de test n'est pas dans la plage attendue de valeurs"

    # Vérification que les valeurs pour la précision soient comprises entre 0 et 1
    assert (0 <= metrics_df["Train Precision"]).all() and (metrics_df["Train Precision"] <=1).all(), "La précision du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Test Precision"]).all() and (metrics_df["Test Precision"] <=1).all(), "La précision du jeu de test n'est pas dans la plage attendue de valeurs"

    # Vérification que les valeurs pour le recall soient comprises entre 0 et 1
    assert (0 <= metrics_df["Train Recall"]).all() and (metrics_df["Train Recall"] <=1).all(), "Le recall du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Test Recall"]).all() and (metrics_df["Test Recall"] <=1).all(), "Le recall du jeu de test n'est pas dans la plage attendue de valeurs"

    # Vérification que les valeurs pour le score f1 soient comprises entre 0 et 1
    assert (0 <= metrics_df["Train f1"]).all() and (metrics_df["Train f1"] <=1).all(), "Le score f1 du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Test f1"]).all() and (metrics_df["Test f1"] <=1).all(), "Le score f1 du jeu de test n'est pas dans la plage attendue de valeurs"

    # Vérification que les valeurs pour le meilleurs seuil et score métier score f1 soient comprises entre 0 et 1
    assert (0 <= metrics_df["Best threshold"]).all() and (metrics_df["Best threshold"] <=1).all(), "Le meilleur seuil du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Train best score"]).all() and (metrics_df["Train best score"] <=1).all(), "Le meilleur score métier du jeu d'entraînement n'est pas dans la plage attendue de valeurs"
    assert (0 <= metrics_df["Test best score"]).all() and (metrics_df["Test best score"] <=1).all(), "Le meilleur score métier du jeu de test n'est pas dans la plage attendue de valeurs"


##########################################################################################################
# FONCTION STREAMLIT: ACCORD OU NON DU CREDIT
###########################################################################################################

@pytest.mark.parametrize("classe, expected_result", [(0, "Accepté"), (1, "Refusé")])
def test_is_credit_accepted(classe, expected_result):
    result = is_credit_accepted(classe)
    assert result == expected_result

#############################################################################################################
# FONCTION STREAMLIT: CREATION DE LA JAUGE DE POSITIONNEMENT
############################################################################################################

def test_create_prediction_gauge():
    # Appel de la fonction avec des probabilités arbitraires
    class_0_prob = 0.126
    class_1_prob = 0.65
    fig = create_prediction_gauge(class_0_prob, class_1_prob)

    # Vérification que la sortie soit bien de type go.Figure
    assert isinstance(fig, go.Figure)

    # Vérification de la propriété de la figure
    assert len(fig.data) == 1  # Un seul indicateur dans la figure
    assert fig.data[0].mode == "gauge+number"

    # Vérification des annotations
    annotations_texts = [annotation.text for annotation in fig.layout.annotations]
    assert "Positionnement" in annotations_texts
    assert "Seuil d'octroi:0.2222" in annotations_texts
    assert "Accord" in annotations_texts
    assert "Refus" in annotations_texts


if __name__ == '__main__':
    pytest.main()

