from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import datetime
import pickle
import plotly.express as px
import plotly.graph_objects as go


###########################################################################################
# FONCTION SCORE BUSINESS: La score est commpris entre 0 et 1 avec le meilleur score de 0
###########################################################################################

def business_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return (10*fn + fp) / sum([tn, fp, 10*fn, tp])

#############################################################################################""
# FONCTION SEUIL METIER: Seuil personnalisé pour une meilleire détection des clients à risque
##############################################################################################

def business_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return (10*fn + fp) / sum([tn, fp, 10*fn, tp])

def seuil_metier(model, X_train, y_train, thresholds):
    '''Recherche du meilleur seuil via l'utilisation du score métier.'''
    
    y_probs = model.predict_proba(X_train)[:, 1]
    
    best_score = float('inf')
    best_threshold = 0
    scores = []
    
    for threshold in thresholds:
        y_pred = [1 if prob > threshold else 0 for prob in y_probs]
        
        # Utilisation de la fonction business_score
        score = business_score(y_train, y_pred)
        
        scores.append(score)
        
        if score < best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold, best_score, scores

################################################################################################
# MATRICE DE CONFUSION ET COURBE ROC
################################################################################################

def confusion_matrix_roc_auc(model, y_true, y_pred, y_pred_proba):
    
    
    '''
    Fonction permettant la visualisation de la matrice de confusion et de la courbe ROC
    --------------------------------------------------------------------------------------
    Matrice de confusion pouvant être réalisée avec seuil de base (0,5) ou le seuil métier
    '''
    
    fig = plt.figure(figsize=(20,15))

    # Calcul de AUC-ROC
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Matrice de confusion
    plt.subplot(221)
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
    plt.title('Matrice de confusion')

    # Courbe ROC
    plt.subplot(222)
    fpr,tpr,_ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='orange', linewidth=5, label=f'AUC = {roc_auc:.4f}')
    plt.title('Courbe ROC')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.legend()
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')  # ligne de référence pour un modèle aléatoire

    plt.tight_layout()
    
    return fig

########################################################################################################
# FONCTION DE TRACKING VIA MLFLOW
###########################################################################################################

def business_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return (10 * fn + fp) / (tn + fp + 10 * fn + tp)

def seuil_metier(model, X, y, thresholds):
    y_probs = model.predict_proba(X)[:, 1]
    best_score = float('inf')
    best_threshold = 0
    scores = []
    
    for threshold in thresholds:
        y_pred = [1 if prob > threshold else 0 for prob in y_probs]
        score = business_score(y, y_pred)
        scores.append(score)
        
        if score < best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold, best_score, scores

def mlflow_tracking(model, X_train, X_test, y_train, y_test, thresholds=np.linspace(0, 1, 100)):
    
    # Obtention du nombre de variables
    num_features = X_train.shape[1]
    
    # Nom du projet :
    mlflow.set_experiment('Projet7_Modelisations')

    with mlflow.start_run() as run:
        name = str(model).split('(')[0]
        mlflow.set_tag("mlflow.runName", datetime.datetime.today().strftime('%Y-%m-%d') + ' - ' + str(model).split('(')[0])
        
        # Obtention du nombre de features
        mlflow.log_param("Number of Features", num_features)
        
         # Entraînement du modèle
        start_time_fit = time.time()
        model.fit(X_train, y_train)
        end_time_fit = time.time()

         # Recherche du meilleur seuil sur l'ensemble d'entraînement
        best_threshold_train, best_score_train, _ = seuil_metier(model, X_train, y_train, thresholds)

         # Prédiction sur l'ensemble de test en utilisant le seuil optimal du train
        y_probs_test = model.predict_proba(X_test)[:, 1]
        y_pred_test_threshold = [1 if prob > best_threshold_train else 0 for prob in y_probs_test]
    
         # Prédiction sur l'ensemble d'entraînement en utilisant le seuil optimal du train
        y_probs_train = model.predict_proba(X_train)[:, 1]
        y_pred_train_threshold = [1 if prob > best_threshold_train else 0 for prob in y_probs_train]
        
        # Enregistrement des hyperparamètres
        mlflow.log_params(model.get_params())

        # Enregistrement des métriques
         # Calcul du temps d'entraînement
        training_time = end_time_fit - start_time_fit
        mlflow.log_metric("Training Time", training_time)
        
         # Enregistrement de l'accuracy pour les jeux d'entraînement et de test avec le seuil optimal
        acc_train = round(accuracy_score(y_train, y_pred_train_threshold) * 100, 4)
        mlflow.log_metric("Train Accuracy", acc_train)
        
        acc_test = round(accuracy_score(y_test, y_pred_test_threshold) * 100, 4)
        mlflow.log_metric("Test Accuracy", acc_test)
        
         # Enregistrement de l'AUC pour les jeux d'entraînement et de test 
        y_train_proba = model.predict_proba(X_train)[:, 1]
        auc_train = round(roc_auc_score(y_train, y_train_proba), 4)
        mlflow.log_metric("Train AUC", auc_train)
        
        y_test_proba = model.predict_proba(X_test)[:, 1]
        auc_test = round(roc_auc_score(y_test, y_test_proba), 4)
        mlflow.log_metric("Test AUC", auc_test)
        
         # Enregistrement du recall pour les jeux d'entraînement et de test avec le seuil optimal
        recall_train = round(recall_score(y_train, y_pred_train_threshold), 4)
        mlflow.log_metric("Train Recall", recall_train)
        
        recall_test = round(recall_score(y_test, y_pred_test_threshold, zero_division=0), 4)
        mlflow.log_metric("Test Recall", recall_test)
        
         # Enregistrement de la précision pour les jeux d'entraînement et de test avec le seuil optimal
        precision_train = round(precision_score(y_train, y_pred_train_threshold, zero_division=0), 4)
        mlflow.log_metric("Train Precision", precision_train)
        
        precision_test = round(precision_score(y_test, y_pred_test_threshold, zero_division=0), 4)
        mlflow.log_metric("Test Precision", precision_test)
        
         # Enregistrement du score f1 pour les jeux d'entraînement et de test
        f1_train = round(f1_score(y_train, y_pred_train_threshold), 4)
        mlflow.log_metric("Train f1", f1_train)
        
        f1_test = round(f1_score(y_test, y_pred_test_threshold, zero_division=0), 4)
        mlflow.log_metric("Test f1", f1_test)
        
         # Seuil métier et meilleur score métier
        mlflow.log_metric("Train Meilleur seuil", round(best_threshold_train, 4)) 
        mlflow.log_metric("Train Meilleur score metier", round(best_score_train, 4))
        best_score_test = business_score(y_test, y_pred_test_threshold)
        mlflow.log_metric("Test Meilleur score metier",  round(best_score_test, 4))
        
         # Sauvegarde du modèle
        mlflow.sklearn.log_model(model, "model")
        
        # Fin du RUN
        mlflow.end_run()
        
        # Création d'un DataFrame avec les métriques
        metrics_dict = {
            "Model": [str(model).split('(')[0]],
            "Training Time": round(training_time, 4),
            "Train Accuracy": round(acc_train, 4),
            "Test Accuracy": round(acc_test, 4),
            "Train AUC": round(auc_train, 4),
            "Test AUC": round(auc_test,4 ),
            "Train Recall": round(recall_train ,4),
            "Test Recall": round(recall_test, 4),
            "Train Precision": round(precision_train, 4),
            "Test Precision": round(precision_test, 4),
            "Train f1": round(f1_train, 4),
            "Test f1": round(f1_test, 4),
            "Best threshold": round(best_threshold_train, 4),
            "Train best score": round(best_score_train, 4),
            "Test best score": round(best_score_test, 4)
        }

        metrics_df = pd.DataFrame(metrics_dict).T

        return metrics_df
    
######################################################################################################
# FONCTION DE CHARGEMENT DU DATAFRAME, DU MODELE ET DU SCALER
######################################################################################################

def load_data():
    return pd.read_csv("testset_rfe_30f.csv", sep=",")


def test_load_model():
    """Test du chargement du modèle LGBMClassifier"""
    model_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/basic_lgbmc.pkl"

    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

def load_scaler():
    """Test du chargement des X de jeu de données 'testset_rfe_30f.csv'"""
    scaler_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/minmax_scaler.pkl"
    with open(scaler_path, 'rb') as file:
        loaded_minmax = pickle.load(file)


##########################################################################################################
# FONCTION STREAMLIT: ACCORD OU NON DU CREDIT
###########################################################################################################

def is_credit_accepted(classe):
    if classe == 0:
        return "Accepté"
    else:
        return "Refusé"

#############################################################################################################
# FONCTION STREAMLIT: CREATION DE LA JAUGE DE POSITIONNEMENT
############################################################################################################

def create_prediction_gauge(class_0_prob, class_1_prob):
    
    fig = go.Figure()

    # Création de la jauge comportant 2 couleurs selon le seuil métier
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=class_1_prob,
        domain={"x": [0.1, 0.6], "y": [0.55, 1]},
        gauge={
            "axis": {"range": [0, 1], "showticklabels": False},
            "bar": {"color": "black", "thickness": 0.15},
            "steps": [
                {"range": [0, 0.2222], "color": "#FFFF00"},  # Quadrant jaune
                {"range": [0.2222, 1], "color": "#FF0000"},  # Quadrant rouge
            ],
        }
    ))
        # Ajout du titre Positionnement au-dessus de la jauge
    fig.add_annotation(
        x=0.35, y=1.38,
        text="Positionnement",
        showarrow=False,
        font=dict(size=24, family="Verdana, sans-serif", color="black")
    )

    # Ajout du seuil d'octroi (seuil métier)
    fig.add_annotation(
        x=0.35, y=1.25,
        text="Seuil d'octroi:0.2222",
        showarrow=False,
        font=dict(size=18, family="Verdana, sans-serif", color="black")
    )

    # Ajout des annotations de texte pour les étiquettes
    fig.add_annotation(
        x=0.12, y= 0.85,
        text="Accord",
        showarrow=False,
        font=dict(size=18, family="Verdana, sans-serif", color="black"),
        textangle= -70
    )

    fig.add_annotation(
        x=0.18, y=1.1,
        text="Refus",
        showarrow=False,
        font=dict(size=18, family="Verdana, sans-serif", color="black"),
        textangle= -35
    )
    return fig
