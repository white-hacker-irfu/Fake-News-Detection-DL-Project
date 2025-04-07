# 📰 Fake News Detection using Deep Learning

This project uses a deep learning model to detect whether a news article is **fake** or **real** based on its text content. The project is built with **TensorFlow**, **Keras**, and **Streamlit** for a simple and interactive web-based user interface.

---

## 🚀 Features

- Binary classification: Detect **Fake (0)** or **Real (1)** news
- Trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Uses LSTM-based neural network
- Deployable via **Streamlit Cloud** or **Flask API**

---

## 🧠 Model Architecture

- Embedding layer
- Bidirectional LSTM
- Dense layers
- Trained for binary classification

Achieved **~99% Accuracy** on validation data!

---

## 📁 Files in this Repo

| File            | Description                                |
|-----------------|--------------------------------------------|
| `app.py`        | Streamlit frontend to test news text input |
| `model.h5`      | Trained deep learning model                |
| `tokenizer.pkl` | Tokenizer object for text preprocessing    |
| `requirements.txt` | Python dependencies                    |
| `README.md`     | Project info and instructions              |

---

## ✅ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fake-news-detection-dl.git
   cd fake-news-detection-dl
