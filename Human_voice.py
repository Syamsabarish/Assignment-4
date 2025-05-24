import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline

# Dataset
DATA_PATH = 'C:/Users/SYAMNARAYANAN/OneDrive/Desktop/visual code/Guvi_project_1/Guvi_project_2/vocal_gender_features_new.csv'

feature_names = [
    'mfcc_5_mean', 'mean_spectral_contrast', 'mfcc_2_mean', 'mfcc_3_std',
    'std_spectral_bandwidth', 'mfcc_12_mean', 'mfcc_1_mean', 'mfcc_10_mean',
    'mfcc_2_std', 'rms_energy'
]

@st.cache_resource
def create_knn_model():
    df = pd.read_csv(DATA_PATH)
    X = df[feature_names]
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

model = create_knn_model()

# Navigation Bar
st.markdown("<h1 style='text-align:center; color:#1E90FF;'>🎧 Human Voice Classification </h1>", unsafe_allow_html=True)

page = st.radio(" ", ["🏡 Home", "🔍 Gender Prediction", "🧠 Develop Info"], horizontal=True)

# Page 1

if page == "🏡 Home":
    st.markdown("""
    <div style='background-color:#f0f8ff; padding: 15px; border-radius: 10px;'>
        <h3 style='color:#2E8B57;'>🔍 Project Overview</h3>
        <p style='font-size:17px; color:#333'>
        This interactive machine learning project aims to predict a speaker's gender based on extracted audio features.
        <br><br>
        The model is trained on a curated dataset with vocal attributes
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#fff0f5; padding: 15px; border-radius: 10px;'>
        <h3 style='color:#DC143C;'>🚀This Project About</h3>
        <ul style='font-size:17px; color:#444'>
            <li>Practice audio-based machine learning</li>
            <li>Understand feature engineering using MFCC</li>
            <li>Build and deploy an end-to-end Streamlit app</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# Page 3
elif page == "🧠 Develop Info":
    st.title("🧠 Model & Training Details")
    st.markdown("""
        <div style='color:#00BFFF; font-size: 17px'>
        <h4>Model: K-Nearest Neighbors Classifier (KNN)</h4>
        <ul>
            <li><b>Training Information()</b> – Split: 80% training / 20% testing</li>
            <li><b>StandardScaler()</b> – Normalize features before training</li>
            <li><b>SMOTE()-</b> – Synthetic Minority Over-sampling to balance class distribution</li>
            <li><b>KNN()</b> – Predicts based on the majority of closest neighbors</li>
            <li>KNeighborsClassifier - <b> Accuracy_score : 0.998</li>
                    
         </ul>
         </div>
    """, unsafe_allow_html=True)

# Page 2
elif page == "🔍 Gender Prediction":
    st.markdown("<h2 style='color:#FF1493'>🎛️ Gender Prediction Interface</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#00BFFF'>Use the sliders below to enter values for each feature:</h4>", unsafe_allow_html=True)

    input_values = []
    for feature in feature_names:
        st.markdown(f"<span style='color:#3333cc; font-weight:bold'>{feature}</span>", unsafe_allow_html=True)
        val = st.slider("", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key=feature)
        input_values.append(val)

    # --- Custom Button Style ---
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #FFA07A;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
            border: none;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #FF7F50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🔍 Predict Gender"):
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)
        gender_map = {0: "Female", 1: "Male"}
        predicted_gender = gender_map.get(prediction[0], "Unknown")
        
        if predicted_gender == "Female":
            color = "#FF69B4"  
        else:
             color = "#1E90FF"
        st.markdown(
            f"<h3 style='color:{color}'>🎯 Predicted Gender: {predicted_gender}</h3>",
            unsafe_allow_html=True
        )
