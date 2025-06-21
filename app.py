import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import socket

# Debugging info
st.set_page_config(page_title="Plant Doctor 🌱", layout="wide")
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
st.sidebar.write(f"Running on: {local_ip}")

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("plant_disease_data.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Preprocess
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Symptom Description'])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Disease Name'])
    
    model = MultinomialNB()
    model.fit(X, y)
    
    # Predict function
    def predict_disease(user_input):
        try:
            input_vec = vectorizer.transform([user_input])
            pred = model.predict(input_vec)[0]
            disease = label_encoder.inverse_transform([pred])[0]
            solution = df[df['Disease Name'] == disease]['Solution'].values[0]
            return disease, solution
        except Exception as e:
            return f"Error: {str(e)}", "Please try different symptoms"
    
    # UI
    st.title("🌿 Plant Disease Diagnosis")
    st.write("Describe your plant's symptoms below:")
    
    user_input = st.text_area("Examples: 'yellow leaves', 'white spots', 'wilting stems'")
    
    if st.button("Diagnose"):
        if user_input:
            with st.spinner("Analyzing symptoms..."):
                disease, solution = predict_disease(user_input)
                st.success(f"🔍 **Diagnosis:** {disease}")
                st.info(f"💊 **Treatment:** {solution}")
        else:
            st.warning("Please describe your plant's symptoms")
else:
    st.error("Failed to load disease database. Please check your data file.")
