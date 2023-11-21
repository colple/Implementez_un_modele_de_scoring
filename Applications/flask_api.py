from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

def predict_perso(model, X, threshold=0.5):
    probas = model.predict_proba(X)
    predict_target = [1 if row[1] > threshold else 0 for row in probas]
    return predict_target, probas[:, 0], probas[:, 1]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lecture des données directement à partir d'un dataframe
        df_param = pd.DataFrame(request.json["dataframe_split"]["data"], columns=request.json["dataframe_split"]["columns"])

        # Chargement du modèle pickle
        model_path = "C:/Users/colin/Documents/Formation_Openclassrooms/Projet7_ImplémentezUnModèleDeScoring/basic_lgbmc.pkl"
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        # Chargement du scaler avec pickle (X du jeu de données 'application_train.csv' standardisés avec le MinMaxScaler)
        with open('minmax_scaler.pkl', 'rb') as file:
            loaded_minmax = pickle.load(file)

        # Retrait de la variable "SK_ID_CURR" si elle est présente
        if 'SK_ID_CURR' in df_param.columns:
            df_param.drop('SK_ID_CURR', axis=1, inplace=True)

        # Normalisation avec le MinMax Scaler du jeu d'entraînement
        df_param_normalized = pd.DataFrame(loaded_minmax.transform(df_param), columns=df_param.columns)

        # Prédictions
        results_target = loaded_model.predict(df_param_normalized)
        results_target_best = predict_perso(loaded_model, df_param_normalized, threshold=0.2222)
        results_proba = loaded_model.predict_proba(df_param_normalized)
        results_target_best, class_0_proba, class_1_proba = predict_perso(loaded_model, df_param_normalized, threshold=0.2222)

        # Ajout des colonnes prédites au Dataframe
        df_param['target'] = results_target_best
        df_param['class_0_proba'] = class_0_proba
        df_param['class_1_proba'] = class_1_proba

        # Ajout d'impressions pour le débogage
        print("Données après prédictions:")
        print(df_param)

        dictionnaire = {
            'target0.5': results_target.tolist(),
            'target': results_target_best,
            'proba': results_proba.tolist(),
            'class_0_proba': results_proba[:, 0].tolist(), 
            'class_1_proba': results_proba[:, 1].tolist(),
            'threshold_used': 0.2222  
        }

    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")
        dictionnaire = {'error': f"Une erreur s'est produite : {str(e)}"}

    return jsonify(dictionnaire)

if __name__ == "__main__":
    app.run(debug=True, port=5014)

