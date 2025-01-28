import streamlit as st
import requests

API_URL = "https://api.xyz.com/v1/gpt4o-mini"  # przykładowy endpoint
API_KEY = "twoj_klucz_api"

st.title("GPT-4o-mini z LangChain w Streamlit - wersja API")

user_input = st.text_area("Twój prompt")

if st.button("Generuj"):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"prompt": user_input, "max_tokens": 200}
    response = requests.post(API_URL, json=payload, headers=headers)
    data = response.json()
    st.write(data.get("generated_text", "Brak tekstu"))

