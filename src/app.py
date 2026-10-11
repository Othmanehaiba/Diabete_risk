import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction System")
st.write("Saisissez les données cliniques du patient pour évaluer son niveau de risque de diabète.")

st.markdown("---")

# Formulaire de saisie des données du patient
with st.form(key="patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Taux de Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=115.0)
        blood_pressure = st.number_input("Pression artérielle (mm Hg)", min_value=0.0, max_value=180.0, value=72.0)
        skin_thickness = st.number_input("Épaisseur du pli cutané (mm)", min_value=0.0, max_value=100.0, value=20.0)
        
    with col2:
        insulin = st.number_input("Niveau d'Insuline (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("Indice de Masse Corporelle (BMI)", min_value=0.0, max_value=70.0, value=26.5)
        pedigree = st.number_input("Fonction Pedigree Diabète", min_value=0.0, max_value=3.0, value=0.45, format="%.3f")
        age = st.number_input("Âge", min_value=1, max_value=120, value=33)

    submit_button = st.form_submit_button(label="Analyser le Risque 🚀")

# Action lors du clic sur le bouton
if submit_button:
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(pedigree),
        "Age": int(age)
    }

    # Appel vers l'API FastAPI
    API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

    with st.spinner("Analyse en cours via l'API..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                label = result["risk_label"]
                prob = result["probability"] * 100

                st.markdown("### Résultat de l'analyse")
                
                if label == "High Risk":
                    st.error(f"⚠️ **Risque Élevé détecté** (Probabilité : {prob:.1f}%)")
                    st.warning("Recommandation : Programmer un bilan médical complémentaire.")
                else:
                    st.success(f"✅ **Risque Faible** (Probabilité : {prob:.1f}%)")
                    st.info("Le profil clinique ne présente pas d'indicateurs de risque majeur.")
            else:
                st.error(f"Erreur API ({response.status_code}) : Impossible de récupérer la prédiction.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Connexion impossible à l'API. Assurez-vous que le serveur FastAPI tourne sur le port 8000.")