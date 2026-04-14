import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="EmoSense", 
    page_icon="🧠", 
    layout="centered"
)

@st.cache_resource
def load_models():
    model = joblib.load('../emotion_model.pkl')
    vectorizer = joblib.load('../tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run trainmodel.py first.")
    st.stop()

def scale_score(raw_score, raw_min=2.4, raw_max=3.6, target_min=1.0, target_max=5.0):
    clamped_score = max(min(raw_score, raw_max), raw_min)
    scaled_score = target_min + (clamped_score - raw_min) * (target_max - target_min) / (raw_max - raw_min)
    
    return float(scaled_score)

st.title("🧠 EmoSense")
st.markdown("Analyze the **Valence** (Positivity) and **Arousal** (Intensity) of any given text. Scores are on a scale from 1.0 to 5.0.")

user_input = st.text_area(
    "Enter your sentence below:", 
    height=100, 
    placeholder="E.g., I am absolutely thrilled about the upcoming vacation! It's going to be amazing."
)

if st.button("Predict Emotion 🚀"):
    # Error handling for empty inputs
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]
            raw_valence = prediction[0]
            raw_arousal = prediction[1]

            valence = scale_score(raw_valence)
            arousal = scale_score(raw_arousal)

        st.divider()
        st.subheader("📊 Emotion Analysis Results")
        
        # Use columns for a side-by-side dashboard look
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Valence Score:** {valence:.2f} / 5.0")
            st.progress(float((valence - 1) / 4)) 
            
            if valence >= 3.5:
                st.write("**Vibe: Positive / Pleasant**")
            elif valence <= 2.5:
                st.write("**Vibe: Negative / Unpleasant**")
            else:
                st.write("**Vibe: Neutral**")

        with col2:
            st.warning(f"**Arousal Score:** {arousal:.2f} / 5.0")
            st.progress(float((arousal - 1) / 4))
            
            if arousal >= 3.5:
                st.write("**Energy: High / Excited**")
            elif arousal <= 2.5:
                st.write("**Energy: Low / Calm**")
            else:
                st.write("**Energy: Moderate**")
                