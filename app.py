import pandas as pd
import numpy as np
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk

# Charger les données depuis le fichier CSV
data = pd.read_csv('train.csv')

# Supposons que vous ayez déjà défini vos caractéristiques X et votre cible y
X = data[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'TotalBsmtSF',
          'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'Fireplaces',
          'GarageCars', 'GarageArea', 'PoolArea']]
y = data['SalePrice']  # le nom de la colonne cible

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le scaler et normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entraîner le modèle
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Sauvegarder le modèle et le scaler
joblib.dump(model, 'model_boston_housing.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Ajouter les styles globaux et le conteneur
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
        color: #1a2b4e;  /* Bleu nuit pour tout le texte */
    }

    /* Masquer l'élément container vide */
    div[data-testid="stMarkdownContainer"] .form-container:empty {
        display: none;
    }

    /* Style global pour le fond */
    .stApp, .main, div[data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        background-image: url('img/pp.png') !important;  /* Ajout de l'image en arrière-plan */
        background-size: cover;  /* Couvrir tout l'arrière-plan */
        background-position: center;  /* Centrer l'image */
        background-repeat: no-repeat;  /* Ne pas répéter l'image */
    }

    .title-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: -25px;
        background-color: #ffffff;
    }

    .house-icon {
        width: 60px;
        height: 60px;
        fill: #1a2b4e;  /* Bleu nuit pour l'icône */
    }

    .title-text {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 32px;
        margin: 0;
        color: #1a2b4e;  /* Bleu nuit pour le titre */
    }

    .form-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(26, 43, 78, 0.05);  /* Ombre légère en bleu nuit */
    }

    .form-header {
        font-family: 'Poppins', sans-serif;
       color: white;  /* Bleu nuit pour l'en-tête */
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 2px solid #1a2b4e;  /* Bordure en bleu nuit */
    }

    .prediction-result {
        font-family: 'Poppins', sans-serif;
        font-size: 28px;
        font-weight: 600;
        text-align: center;
        color: #1a2b4e;  /* Bleu nuit pour le résultat */
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 2px 6px rgba(26, 43, 78, 0.05);
        border: 1px solid #1a2b4e;  /* Bordure en bleu nuit */
    }
    </style>

    <div class="title-container">
        <svg class="house-icon" viewBox="0 0 24 24">
            <path d="M12 3L4 9v12h16V9l-8-6zm6 16h-3v-6H9v6H6v-9l6-4.5 6 4.5v9z"/>
        </svg>
        <h1 class="title-text">SmartPrix</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    Bienvenue sur SmartPrix ! Notre application utilise l'intelligence artificielle pour estimer 
    le prix de votre maison en fonction de ses caractéristiques. Commencez à remplir le formulaire ci-dessous.
""")

# Créer un bloc pour les entrées utilisateur
st.markdown('<div class="form-container">', unsafe_allow_html=True)
st.markdown('<h3 class="form-header">Entrées utilisateur</h3>', unsafe_allow_html=True)

# Organiser les champs en rangées de 3
col1, col2, col3 = st.columns(3)
with col1:
    LotArea = st.number_input("Superficie terrain (m²)", min_value=0)
with col2:
    OverallQual = st.number_input("Qualité globale (1-10)", min_value=1, max_value=10)
with col3:
    OverallCond = st.number_input("État général (1-10)", min_value=1, max_value=10)

col1, col2, col3 = st.columns(3)
with col1:
    YearBuilt = st.number_input("Année construction", min_value=1900, max_value=2023)
with col2:
    TotalBsmtSF = st.number_input("Surface sous-sol (m²)", min_value=0)
with col3:
    GrLivArea = st.number_input("Surface habitable (m²)", min_value=0)

col1, col2, col3 = st.columns(3)
with col1:
    BedroomAbvGr = st.number_input("Nombre de chambres", min_value=0)
with col2:
    FullBath = st.number_input("Salles de bain", min_value=0)
with col3:
    HalfBath = st.number_input("Toilettes séparées", min_value=0)

col1, col2, col3 = st.columns(3)
with col1:
    Fireplaces = st.number_input("Nombre cheminées", min_value=0)
with col2:
    GarageCars = st.number_input("Places de garage", min_value=0)
with col3:
    GarageArea = st.number_input("Surface garage (m²)", min_value=0)

# Dernière ligne centrée
col1, col2, col3 = st.columns([1.2, 0.6, 1.2])
with col2:
    PoolArea = st.number_input("Surface piscine (m²)", min_value=0)

# Bouton de prédiction
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Calculer l'estimation", use_container_width=True)

if predict_button:
    user_input = pd.DataFrame({
        'LotArea': [LotArea],
        'OverallQual': [OverallQual],
        'OverallCond': [OverallCond],
        'YearBuilt': [YearBuilt],
        'TotalBsmtSF': [TotalBsmtSF],
        'GrLivArea': [GrLivArea],
        'BedroomAbvGr': [BedroomAbvGr],
        'FullBath': [FullBath],
        'HalfBath': [HalfBath],
        'Fireplaces': [Fireplaces],
        'GarageCars': [GarageCars],
        'GarageArea': [GarageArea],
        'PoolArea': [PoolArea]
    })

    # Normaliser les données et prédire
    user_input_scaled = scaler.transform(user_input)
    predicted_price = model.predict(user_input_scaled)

    # Afficher le résultat
    st.markdown(f"<div class='prediction-result'>Prix estimé : {predicted_price[0]:,.2f} €</div>",
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
