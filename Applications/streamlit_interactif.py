import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
import shap
from shap import TreeExplainer, Explainer
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import io
st.set_option('deprecation.showPyplotGlobalUse', False)


# Chargement du modèle depuis un fichier pickle
model_local_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/basic_lgbmc.pkl"
model_github_url = "https://github.com/colple/Implementez_un_modele_de_scoring/raw/main/basic_lgbmc.pkl"

try:
    with open(model_local_path, 'rb') as model_file:
        model = pickle.load(model_file)
except:
    response = requests.get(model_github_url)
    if response.status_code == 200:
        model_bytes = io.BytesIO(response.content)
        model = pickle.load(model_bytes)
    else:
        st.error(f"Le chargement du modèle depuis GitHub a échoué avec le code d'état {response.status_code}")
        st.stop()


# Définition de l'URL de l'API
api_url = "https://modele-scoring-credits-c459a33a2133.herokuapp.com/predict"

# Déclaration de la variable uploaded_file en dehors de la fonction afin de permettre le chargement d'un jeu de données
uploaded_file = None

# Chargement du jeu de données au lancement de l'application
@st.cache_data
def load_data():
    if uploaded_file is not None:
        # Chargement depuis le fichier téléchargé
        return pd.read_csv(uploaded_file, sep=",")
    
    # Chargement depuis GitHub
    github_url = "https://github.com/colple/Implementez_un_modele_de_scoring/blob/main/Datas/testset_rfe_30f.csv?raw=true"
    try:
        return pd.read_csv(github_url, sep=",")
    except Exception as e:
        # En cas d'échec, essaie de charger localement
        local_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/testset_rfe_30f.csv"
        try:
            return pd.read_csv(local_path, sep=",")
        except FileNotFoundError:
            st.error(f"Le chargement des données a échoué avec l'erreur : {str(e)}")
            st.stop()

# Chargement des données
df_complet = load_data()

# Copie du fichier avant normalisation des données
df_complet_scaled = df_complet.copy()

# Chargement du scaler avec pickle
minmax_scaler_local_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/minmax_scaler.pkl"
minmax_scaler_github_url = "https://github.com/colple/Implementez_un_modele_de_scoring/blob/main/minmax_scaler.pkl?raw=true"

try:
    loaded_minmax = pickle.load(open(minmax_scaler_local_path, 'rb'))
except:
    response = requests.get(minmax_scaler_github_url)
    if response.status_code == 200:
        minmax_scaler_bytes = io.BytesIO(response.content)
        loaded_minmax = pickle.load(minmax_scaler_bytes)
    else:
        st.error(f"Le chargement du MinMax Scaler depuis GitHub a échoué avec le code d'état {response.status_code}")
        st.stop()


# Retrait de la colonne "SK_ID_CURR" si elle est présente
if 'SK_ID_CURR' in df_complet_scaled.columns:
    df_complet_scaled.drop('SK_ID_CURR', axis=1, inplace=True)

# Normalisation avec le MinMax Scaler du jeu d'entraînement
df_complet_scaled_values = loaded_minmax.transform(df_complet_scaled)

# Création d'un nouveau Dataframe avec les valeurs transformées
df_complet_scaled = pd.DataFrame(df_complet_scaled_values, columns=df_complet_scaled.columns)

# Ajout la colonne SK_ID_CURR
df_complet_scaled["SK_ID_CURR"] = df_complet["SK_ID_CURR"]

#########################################################################################################
# TITRE PRINCIPAL DE LA PAGE
#########################################################################################################

custom_css = """
<style>
    .custom-title {
        font-family: Verdana, sans-serif;
        font-size: 32px;
        font-weight: bold;
        margin-top: -76px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Titre personnalisé
st.markdown('<div class="custom-title">Décision sur l\'octroi du prêt</div>', unsafe_allow_html=True)

###############################################################################################################
# BARRE LATERALE
###############################################################################################################

# Logo de la société
st.sidebar.image("logo_pret_a_depenser.png", use_column_width=True)

# Texte introductif
st.sidebar.markdown('<h2 style="font-family: Verdana, sans-serif; font-size: 18px;">Bienvenue dans notre société proposant des prêts à la consommation en toute transparence.</h2>', unsafe_allow_html=True)

# Ligne vide pour la création d'un espace
st.sidebar.text("")

# Chargement des données non standardisées
with st.sidebar:
    uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

# Création d'un curseur pour le zoom dans la barre latérale
with st.sidebar:
    st.write("<p style='font-size: 14px; font-weight: bold; margin: 0;'>Zoom</p>", unsafe_allow_html=True)
    zoom_level = st.sidebar.slider("", 50, 400, 100)

# Utilisation de CSS personnalisé pour appliquer le zoom dans la barre latérale
st.sidebar.markdown(
    f"""
    <style>
        .sidebar .sidebar-content {{
            transform: scale({zoom_level / 100.0});
            transform-origin: top left;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Utilisation de CSS personnalisé pour l'application du zoom au contenu principal
st.markdown(
    f"""
    <style>
        div.stApp {{
            transform: scale({zoom_level / 100.0});
            transform-origin: top left;
        }}
    </style>
    """,
    unsafe_allow_html=True)

# Ligne vide pour la création d'un espace
st.sidebar.text("")

# Personnalisation de la largeur de la boîte de sélection avec des balises HTML et CSS
st.write("""
<style>
div[data-baseweb="select"] {
    width: 280px;  
    margin: 0; 
}
</style>
""", unsafe_allow_html=True)

# Fonction permettant d'afficher si le crédit est accepté ou non
def is_credit_accepted(classe):
    if classe == 0:
        return "Accepté"
    else:
        return "Refusé"

# Création d'une boîte de sélection pour les clients avec le texte personnalisé
with st.sidebar:
    st.write("<p style='font-size: 14px; font-weight: bold; margin: 0;'>Sélectionnez un client par identifiant :</p>", unsafe_allow_html=True)  
    selected_client = st.selectbox("", df_complet['SK_ID_CURR'].tolist())

##############################################################################################################
# CONTENU DE LA PAGE PRINCIPALE
##############################################################################################################

df_api = df_complet.copy()

api_data = {
    "dataframe_split": {
        "index": df_api.index.tolist(),
        "columns": df_api.columns.tolist(),
        "data": df_api.values.tolist()
    }
}

headers = {'Content-Type': 'application/json'}

# Envoi de la requête à l'API
response = requests.post(api_url, json=api_data, headers=headers)

# Traitement de la réponse par l'API
if response.status_code == 200:
    api_response = response.json()
    
    # Extraction des colonnes prédites de la réponse de l'API
    results_target_best = api_response['target']
    class_0_proba = api_response['class_0_proba']
    class_1_proba = api_response['class_1_proba']

    # Ajout des colonnes prédites au dataFrame df_api
    df_api['target'] = results_target_best
    df_api['class_0_proba'] = class_0_proba
    df_api['class_1_proba'] = class_1_proba


# Ajout des colonnes dans le dataframe original standardisé
df_complet_scaled["Prediction_Class_0"] = df_api['class_0_proba']  
df_complet_scaled["Prediction_Class_1"] = df_api['class_1_proba']
df_complet_scaled["Classe"] = df_api['target']

# Ajout des colonnes dans le dataframe original non standardisé
df_complet["Prediction_Class_0"] = df_api['class_0_proba']  
df_complet["Prediction_Class_1"] = df_api['class_1_proba']
df_complet["Classe"] = df_api['target']


#############################################################################################################
# PREMIERE PARTIE DE LA PAGE PRINCIPALE
#############################################################################################################

# Division en 3 colonnes de taille inégale
col1, col2, col3 = st.columns([1,2,1])

###############################################
# Décision et demande d'une donnée spécifique
###############################################

# Prédiction spécifique du client sélectionné
if selected_client:
    client_data = df_complet[df_complet['SK_ID_CURR'] == selected_client]
    client_prediction = client_data['Classe'].values[0]
    credit_status = is_credit_accepted(client_prediction)

    # Définition de la taille du texte en utilisant du CSS
    style = "font-size: 24px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: -340px; margin-top:-525px"
    st.markdown(f"<p style='{style}'>Crédit : {credit_status}</p>", unsafe_allow_html=True)
    
    with col1:    
        # Définition de la taille de l'emoji en utilisant du CSS
        emoji_style = "font-size: 48px; margin-left: -340px;"

        # Affichage d'un pouce en l'air jaune si le crédit est accepté
        if credit_status == "Accepté":
            st.markdown(f"<p style='{emoji_style}'>👍</p>", unsafe_allow_html=True)
        # Affichage d'un pouce en bas si le crédit est refusé
        else:
            st.markdown(f"<p style='{emoji_style}'>👎</p>", unsafe_allow_html=True)

    # Accessibilité des données du client sous la forme d'une selectbox présente dans la barre latérale
        style_1bis = "font-size: 24px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: -340px; margin-top:0px"
        st.markdown(f"<p style='{style_1bis}'>Vos données</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.write("<p style='font-size: 14px; font-weight: bold; margin: 0;'>Sélectionnez une donnée personnelle:</p>", unsafe_allow_html=True)  
        columns_to_remove = ["SK_ID_CURR", "SK_ID_CURR", "Prediction_Class_0", "Prediction_Class_1","Classe"]
        personal_data = st.selectbox("", client_data.drop(columns=columns_to_remove).columns)
    
    with col1:   
        # Donnée demandée
        style_1ter = "font-size: 20px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: -340px; margin-top:0px"
        st.markdown(f"<p style='{style_1ter}'>Voici la donnée demandée</p>", unsafe_allow_html=True)
        style_1q = "font-size: 18px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: -340px; margin-top:0px"
        st.write(f"<p style='{style_1q}'>{personal_data} : {client_data[personal_data].values[0]}</p>", unsafe_allow_html=True)
    

#######################################
# Les valeurs de SHAPLEY locales
#######################################

    # Définition de la taille du texte en utilisant du CSS
    style_2 = "font-size: 24px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: +50px; margin-top:-545px"

    # Affichage de l'explication
    explanation = "Principaux facteurs contribuant à la décision"
    st.markdown(f"<p style='{style_2}'> {explanation}</p>", unsafe_allow_html=True)

    with col2:
        # Calcul des valeurs SHAP à partir du dataFrame original standardisé
        shap_df = df_complet_scaled[df_complet_scaled['SK_ID_CURR'] == selected_client].drop(["SK_ID_CURR", "Prediction_Class_0", "Prediction_Class_1", "Classe"], axis=1)
        explainer = shap.TreeExplainer(model)

        # Prédiction pour le client sélectionné
        shap_df_client = client_data.drop(["SK_ID_CURR", "Prediction_Class_0", "Prediction_Class_1", "Classe"], axis=1)
        shap_values = explainer(shap_df_client, check_additivity=False)

        # Obtention de la prédiction pour le client
        prediction_shap = model.predict_proba(shap_df_client)[0]

        # Affichage des valeurs SHAP pour la prédiction locale
        fig_shap_client = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0].values[:, 0], shap_df_client.iloc[0], max_display=11)
    
        # Affichage du graphique
        st.pyplot(fig_shap_client)

#######################################################################
# Positionnement par rapport aux autres clients et autres informations
#######################################################################

with col3:
    
    # Utilisation du  modèle pour l'obtention des probabilités prédites
    probas = model.predict_proba(df_complet_scaled.drop(["SK_ID_CURR", "Prediction_Class_0", "Prediction_Class_1","Classe"], axis=1))

    # Obtention des probabilités associées à chaque classe
    class_0_probas = probas[:, 0]
    class_1_probas = probas[:, 1]

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

    # Obtention des probabilités pour le client sélectionné
    if selected_client:
        client_index = df_complet_scaled[df_complet_scaled['SK_ID_CURR'] == selected_client].index[0]
        class_0_prob = class_0_probas[client_index]
        class_1_prob = class_1_probas[client_index]

        # Création de la jauge pour le client sélectionné en passant les probabilités des classes prédites
        st.plotly_chart(create_prediction_gauge(class_0_prob,class_1_prob))

###################################################
# DEUXIEME PARTIE: LES VALEURS DE SHAPLEY GLOBALES
# #################################################

# Ligne de séparation avec des styles CSS personnalisés
line_style_1 = "height: 2px; background-color: #000; margin: -150px 0; width: 100%;"
st.markdown(f"<hr style='{line_style_1}'>", unsafe_allow_html=True)

style_3 = "font-size: 24px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: 140px; margin-top:-50px"
st.markdown(f"<p style='{style_3}'>L'importance des données</p>", unsafe_allow_html=True)

@st.cache_data  # Mise en cache de la fonction pour qu'elle ne soit pas recalculée à chaque rafraîchissement de la page
def calculate_shap_values(df_complet_scaled):
    # Calcul des valeurs SHAP globales à l'aide du TreeExplainer
    shap_values_global = explainer.shap_values(df_complet_scaled.drop(["SK_ID_CURR", "Prediction_Class_0", "Prediction_Class_1","Classe"], axis=1))
    
    # Obtention des valeurs SHAP moyennes pour chaque variable
    mean_shap_values = np.abs(shap_values_global[0]).mean(axis=0)
    
    # Tri des variables par ordre décroissant en fonction de leur valeur SHAP moyenne
    sorted_indices = mean_shap_values.argsort()[-30:]
    
    return mean_shap_values, sorted_indices

# Utilisation de la fonction mise en cache pour calculer les valeurs SHAP
mean_shap_values, sorted_indices = calculate_shap_values(df_complet_scaled)

# Création du graphique barh pour afficher les valeurs SHAP globales des 30 variables
fig, ax = plt.subplots(figsize=(60, 60))
ax.barh(range(30), mean_shap_values[sorted_indices], color='dodgerblue')
ax.set_yticks(range(30))
ax.set_yticklabels(df_complet_scaled.drop(["SK_ID_CURR", "Prediction_Class_0", "Prediction_Class_1","Classe"], axis=1).columns[sorted_indices], fontsize=40)
st.pyplot(fig)

########################################################
# TROISIEME PARTIE: VISUALISATION DE CERTAINES DONNES
# ######################################################

# Ligne de séparation avec des styles CSS personnalisés
line_style = "height: 2px; background-color: #000; margin-top: -0x 0; width: 100%;"
st.markdown(f"<hr style='{line_style}'>", unsafe_allow_html=True)

# Sélection du client via une selectbox
selected_client_id = selected_client  # L'ID du client sélectionné
selected_client_data = df_complet[df_complet['SK_ID_CURR'] == selected_client_id]

# Liste des noms de graphiques
graphiques_disponibles = ["Montant du crédit selon l'âge", "Montant du crédit selon le revenu", "Annuité selon le montant du crédit", "Nombre de crédits précédemment acceptés en fonction de l'âge"]

with st.sidebar:    
    st.write("<p style='font-size: 14px; font-weight: bold; margin: 0;'>Sélectionnez les graphiques à afficher:</p>", unsafe_allow_html=True)  
    selected_graphiques = st.multiselect("", graphiques_disponibles)

style_4 = "font-size: 24px; font-family: Verdana, sans-serif; font-weight: bold; margin-left: 40px; margin-top:10px"
st.markdown(f"<p style='{style_4}'>Visualisation interactive de certaines données</p>", unsafe_allow_html=True)

if "Montant du crédit selon l'âge" in selected_graphiques:

    # Création d'une colonne Décision comportant Refus ou Accord
    df_complet['Décision'] = df_complet['Classe'].apply(lambda x: 'Accord' if x == 0 else 'Refus')

    # Création du code couleur
    color_map = {'Accord': 'dodgerblue', 'Refus': 'red'}
    fig_age_credit = go.Figure()
    
    # Ajout de la dispersion pour "Refus" et "Accord"
    fig_age_credit.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Refus']['AGE'],
                                             y=df_complet[df_complet['Décision'] == 'Refus']['AMT_CREDIT'],
                                             mode='markers',
                                             marker=dict(size=5, color='red'),
                                             name="Refus"))
    
    fig_age_credit.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Accord']['AGE'],
                                             y=df_complet[df_complet['Décision'] == 'Accord']['AMT_CREDIT'],
                                             mode='markers',
                                             marker=dict(size=5, color='dodgerblue'),
                                             name="Accord"))

    # Obtention des valeurs pour le client sélectionné
    selected_client_age = selected_client_data["AGE"].values[0]
    selected_client_credit_amount = selected_client_data['AMT_CREDIT'].values[0]

    # Ajout de l'emplacement du client sélectionné
    fig_age_credit.add_trace(go.Scattergl(x=[selected_client_age], y=[selected_client_credit_amount],
                                         mode='markers',
                                         marker=dict(size=10, color='yellow', line=dict(width=2, color='black')),
                                         name="VOUS"))

    # Personnalisation des titres et de la légende
    fig_age_credit.update_layout(title="MONTANT DU CREDIT SELON L'AGE",
                                xaxis_title="AGE",
                                yaxis_title="MONTANT DU CREDIT", 
                                legend_title_text="DECISION")

    # Affichage du graphique interactif
    st.plotly_chart(fig_age_credit)

if "Montant du crédit selon le revenu" in selected_graphiques:

    # Création d'une colonne Décision comportant Refus ou Accord
    df_complet['Décision'] = df_complet['Classe'].apply(lambda x: 'Accord' if x == 0 else 'Refus')

    # Création du code couleur
    color_map= {'Accord': 'dodgerblue', 'Refus': 'red'}
    fig_income_credit = go.Figure()

    # Ajout de la dispersion pour "Refus" et "Accord"
    fig_income_credit.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Refus']['AMT_INCOME_TOTAL'],
                                             y=df_complet[df_complet['Décision'] == 'Refus']['AMT_CREDIT'],
                                             mode='markers',
                                             marker=dict(size=5, color='red'),
                                             name="Refus"))
    
    fig_income_credit.add_trace(go.Scattergl(x=df_complet[df_complet["Décision"]== 'Accord']['AMT_INCOME_TOTAL'],
                                             y=df_complet[df_complet["Décision"]== 'Accord']['AMT_CREDIT'],
                                             mode='markers',
                                             marker=dict(size=5, color="dodgerblue"),
                                             name="Accord"))

    # Obtention des valeurs pour le client sélectionné
    selected_client_income = selected_client_data['AMT_INCOME_TOTAL'].values[0]
    selected_client_credit_amount = selected_client_data['AMT_CREDIT'].values[0]

    # Ajout de l'emplacement du client sélectionné
    fig_income_credit.add_trace(go.Scattergl(x=[selected_client_income], y=[selected_client_credit_amount], 
                                            mode='markers',
                                            marker=dict(size=10, color='yellow', line=dict(width=2, color='black')),  
                                            name="VOUS"))
    
    # Personnalisation des titres et de la légende
    fig_income_credit.update_layout(title="MONTANT DU CREDIT SELON LE REVENU",
                                    xaxis_title="REVENU",
                                    yaxis_title="MONTANT DU CREDIT",
                                    legend_title_text="DECISION")

    # Affichage du graphique interactif
    st.plotly_chart(fig_income_credit)

if "Annuité selon le montant du crédit" in selected_graphiques:
    
    # Création d'une colonne Décision comportant Refus ou Accord
    df_complet['Décision'] = df_complet['Classe'].apply(lambda x: 'Accord' if x == 0 else 'Refus')

    # # Création du code couleur
    color_map = {'Accord': 'dodgerblue', 'Refus': 'red'}
    fig_annuity_credit = go.Figure()

    # Ajout de la dispersion pour "Refus" et "Accord"
    fig_annuity_credit.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Refus']['AMT_CREDIT'],
                                             y=df_complet[df_complet['Décision'] == 'Refus']['AMT_ANNUITY'],
                                             mode='markers',
                                             marker=dict(size=5, color='red'),
                                             name="Refus"))
    
    fig_annuity_credit.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Accord']['AMT_CREDIT'],
                                             y=df_complet[df_complet['Décision'] == 'Accord']['AMT_ANNUITY'],
                                             mode='markers',
                                             marker=dict(size=5, color='dodgerblue'),
                                             name="Accord"))
    
    # Obtention des valeurs pour le client sélectionné
    selected_client_annuity = selected_client_data["AMT_ANNUITY"].values[0]
    selected_client_credit_amount = selected_client_data['AMT_CREDIT'].values[0]

    # Ajout de l'emplacement du client sélectionné
    fig_annuity_credit.add_trace(go.Scattergl(x=[selected_client_credit_amount], y=[selected_client_annuity],
                                         mode='markers',
                                         marker=dict(size=10, color='yellow', line=dict(width=2, color='black')),
                                         name="VOUS"))

    # Personnalisation des titres et de la légende
    fig_annuity_credit.update_layout(title="ANNUITE SELON LE MONTANT DU CREDIT",
                                    xaxis_title="MONTANT DU CREDIT",
                                    yaxis_title="ANNUITE",
                                    legend_title_text="DECISION")

    # Affichage du graphique interactif
    st.plotly_chart(fig_annuity_credit)

    
if "Nombre de crédits précédemment acceptés en fonction de l'âge" in selected_graphiques:

    # Création de la colonne Décision comportant Refus ou Accord
    df_complet['Décision'] = df_complet["Classe"].apply(lambda x: 'Accord' if x == 0 else 'Refus')

    # Création du code couleur
    color_map= {'Accord': 'dodgerblue', 'Refus': 'red'}
    fig_loans_age = go.Figure()
    
    # Ajout de la dispersion pour "Refus" et "Accord"
    fig_loans_age.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Refus']["AGE"],
                                        y= df_complet[df_complet['Décision'] == 'Refus']["total_accepted_loans"],
                                        mode= "markers",
                                        marker=dict(size=5, color='red'),
                                        name="Refus"))
    
    fig_loans_age.add_trace(go.Scattergl(x=df_complet[df_complet['Décision'] == 'Accord']["AGE"],
                                        y= df_complet[df_complet['Décision'] == 'Accord']["total_accepted_loans"],
                                        mode= "markers",
                                        marker=dict(size=5, color='dodgerblue'),
                                        name="Accord"))

    # Obtention des valeurs pour le client sélectionné
    selected_client_age = selected_client_data["AGE"].values[0]
    selected_client_total_loans = selected_client_data["total_accepted_loans"].values[0]

    # Ajout de l'emplacement du client sélectionné
    fig_loans_age.add_trace(go.Scattergl(x=[selected_client_age], y=[selected_client_total_loans], 
                                       mode='markers',
                                       marker=dict(size=10, color='yellow', line=dict(width=2, color='black')),
                                       name="VOUS"))

    # Personnalisation des titres et de la légende
    fig_loans_age.update_layout(title="NOMBRE DE CREDITS PRECEDEMMENT ACCEPTES SELON L'AGE",
                                xaxis_title="AGE",
                                yaxis_title="NOMBRE DE CREDITS",
                                legend_title_text="DECISION")

    # Affichage du graphique interactif
    st.plotly_chart(fig_loans_age)