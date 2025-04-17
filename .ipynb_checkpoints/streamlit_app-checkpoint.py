import streamlit as st
st.set_page_config(page_title="Attack Classifier", layout="wide")

import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re
import pandas as pd
from datetime import datetime
import os

# Load model and assets
clf = joblib.load("attack_classifier_25.pkl")
attack_category_decoder = joblib.load("attack_category_decoder25.pkl")
tokenizer = AutoTokenizer.from_pretrained("tiny_bert_suchetmodelsave25")
model = AutoModel.from_pretrained("tiny_bert_suchetmodelsave25")
model.eval()

attack_descriptions = {
    "Analysis": "Traffic patterns indicate suspicious analysis activity, often used for reconnaissance.",
    "Backdoor": "Backdoor access suspected – attacker may have unauthorized persistent access.",
    "DoS": "Denial of Service attack detected – overwhelming system resources.",
    "Exploits": "Exploit attempt detected – attacker trying to use known vulnerabilities.",
    "Fuzzers": "Fuzzing activity – sending random data to find vulnerabilities.",
    "Generic": "Generic attack signature – behavior matching broad threat patterns.",
    "Reconnaissance": "Reconnaissance activity – scanning or probing network/system.",
    "Shellcode": "Shellcode attack – small code payloads used to gain control of system.",
    "Worms": "Worm activity – self-replicating malware attempting to spread.",
}

# Utility Functions
def parse_input(input_text):
    pattern = r"Protocol:\s*(\S+),\s*Source:\s*([\d.]+:\d+),\s*Destination:\s*([\d.]+:\d+),\s*Attack Name:\s*(.+)"
    match = re.match(pattern, input_text.strip())
    if match:
        protocol = match.group(1)
        source = match.group(2)
        destination = match.group(3)
        attack_info = match.group(4)
        return f"{protocol} {source} {destination} {attack_info}"
    return None

def predict_attack(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    pred_code = clf.predict(embedding)[0]
    prediction_label = attack_category_decoder[pred_code]
    description = attack_descriptions.get(prediction_label, "No description available.")
    return prediction_label, description

def log_prediction(user_input, parsed, prediction, description):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "Timestamp": now,
        "Input": user_input,
        "Parsed": parsed,
        "Prediction": prediction,
        "Description": description
    }
    df = pd.DataFrame([entry])
    if os.path.exists("logs.csv"):
        df.to_csv("logs.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("logs.csv", index=False)

def clear_logs():
    if os.path.exists("logs.csv"):
        os.remove("logs.csv")

# ========== STATE INIT ========== #
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "description" not in st.session_state:
    st.session_state.description = ""

# ========== SIDEBAR UI ========== #
st.sidebar.title("🔧 Controls")
st.sidebar.markdown("Enter traffic input in this format:")
st.sidebar.code("Protocol: TCP, Source: 192.168.1.2:1234, Destination: 10.0.0.1:80, Attack Name: FuzzingAttempt")

st.session_state.user_input = st.sidebar.text_area("📝 Input Traffic Data", value=st.session_state.user_input, height=180)

# Button actions
with st.sidebar:
    col1, col2 = st.columns(2)
    predict_btn = col1.button("🚀 Predict")
    clear_input_btn = col2.button("🧹 Clear Input")

    col3, col4 = st.columns(2)
    clear_output_btn = col3.button("🗑️ Clear Output")

# Handle Clear Input
if clear_input_btn:
    st.session_state.user_input = ""
    st.session_state.prediction = ""
    st.session_state.description = ""
    st.rerun()

# Handle Clear Logs
if clear_output_btn:
    clear_logs()
    st.rerun()

# Handle Predict
if predict_btn:
    if st.session_state.user_input.strip():
        parsed = parse_input(st.session_state.user_input)
        if parsed:
            prediction, description = predict_attack(parsed)
            st.session_state.prediction = prediction
            st.session_state.description = description
            log_prediction(st.session_state.user_input, parsed, prediction, description)
        else:
            st.session_state.prediction = "⚠️ Format error"
            st.session_state.description = "Please follow the input format."

# ========== MAIN PAGE ========== #

st.markdown("<h1 style='text-align: center;'>🔍 Encrypted Traffic Attack Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

if st.session_state.prediction:
    if "⚠️" in st.session_state.prediction:
        st.warning(st.session_state.description)
    else:
        st.success(f"🎯 Predicted Category: {st.session_state.prediction}")
        st.info(f"🧾 Description: {st.session_state.description}")

        result_df = pd.DataFrame({
            "Input": [st.session_state.user_input],
            "Prediction": [st.session_state.prediction],
            "Description": [st.session_state.description]
        })
        st.download_button("📥 Download Result", data=result_df.to_csv(index=False), file_name="prediction_result.csv", mime="text/csv")

# ========== LOGS & CHARTS ========== #
st.markdown("---")
st.markdown("## 📊 Recent Predictions")

if os.path.exists("logs.csv") and os.path.getsize("logs.csv") > 0:
    logs_df = pd.read_csv("logs.csv")
    if not logs_df.empty:
        st.dataframe(logs_df.tail(5), use_container_width=True)
        chart_data = logs_df["Prediction"].value_counts().reset_index()
        chart_data.columns = ["Category", "Count"]
        st.bar_chart(chart_data.set_index("Category"))

        # ✅ Download button moved here (after chart)
        st.download_button("📥 Download Logs", data=logs_df.to_csv(index=False), file_name="logs.csv", mime="text/csv")
    else:
        st.info("No recent predictions.")
else:
    st.info("No prediction logs yet.")
