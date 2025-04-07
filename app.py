
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model and tokenizer
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 500

def preprocess(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def predict_news(text):
    cleaned = preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    return "🟢 Real News" if pred > 0.5 else "🔴 Fake News"

# UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector (Deep Learning)")
user_input = st.text_area("Paste a news article below:")

if st.button("Predict"):
    if user_input.strip():
        result = predict_news(user_input)
        st.success(f"**Prediction:** {result}")
    else:
        st.warning("⚠️ Please enter some news text.")
