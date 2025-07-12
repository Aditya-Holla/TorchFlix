import streamlit as st
import torch
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn

# --- Load model class ---
class TorchFlix(nn.Module):
    def __init__(self, input_shape, hidden_units, output_units):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_units // 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, output_units)
        )

    def forward(self, x):
        return self.layer_stack(x)

# --- Load encoders and model ---
genre_mlb = joblib.load("genre_mlb.pkl")
actor_mlb = joblib.load("actor_mlb.pkl")
keyword_mlb = joblib.load("keyword_mlb.pkl")

input_dim = len(genre_mlb.classes_) + len(actor_mlb.classes_) + len(keyword_mlb.classes_)
model = TorchFlix(input_shape=input_dim, hidden_units=256, output_units=1)
model.load_state_dict(torch.load("torchflix_weights.pth", map_location=torch.device("cpu")))
model.eval()

# --- Streamlit UI ---
st.title(" TorchFlix: Will You Like This Movie?")

st.markdown("Enter the traits of a movie and I'll tell you if you'd like it... maybe.")

genre_input = st.text_input("Genres (comma-separated)", "Action, Comedy")
actor_input = st.text_input("Actors (comma-separated)", "Will Smith")
keyword_input = st.text_input("Keywords (comma-separated)", "explosion, friendship")

if st.button("Predict if I'll Like It"):
    # Preprocess input
    genres = [g.strip() for g in genre_input.split(",") if g.strip()]
    actors = [a.strip() for a in actor_input.split(",") if a.strip()]
    keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]

    genre_vec = genre_mlb.transform([genres])
    actor_vec = actor_mlb.transform([actors])
    keyword_vec = keyword_mlb.transform([keywords])

    movie_vec = hstack([genre_vec, actor_vec, keyword_vec])
    movie_tensor = torch.tensor(movie_vec.toarray(), dtype=torch.float32)

    # Get prediction
    with torch.no_grad():
        logits = model(movie_tensor)
        prob = torch.sigmoid(logits).item()

    st.write(f" Prediction Probability: `{prob:.2f}`")
    if prob >= 0.5:
        st.success(" You’ll probably like this movie!")
    else:
        st.warning(" Probably a skip.")
