import numpy as np
import json
import streamlit as st
import openai

# Funkcje do ładowania danych z cachowaniem
@st.cache_data
def load_embeddings(file_path):
    return np.load(file_path)

@st.cache_data
def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Załaduj embeddingi i teksty
rank_1_embeddings = load_embeddings('rank_1_embeddings.npy')
other_embeddings = load_embeddings('other_embeddings.npy')
rank_1_texts = load_texts('rank_1_texts.json')
other_texts = load_texts('other_texts.json')

# Konfiguracja strony
st.set_page_config(page_title="Amazon.de - generator opisów", page_icon="🎉")

# Sprawdzanie autoryzacji
if 'authorized' not in st.session_state:
    st.session_state['authorized'] = False

if not st.session_state['authorized']:
    password = st.text_input("Wprowadź hasło:", type="password")
    login_button = st.button("Zaloguj się")

    if login_button:
        if password == st.secrets["bot_secrets"]["password"]:
            st.session_state['authorized'] = True
            st.success("Hasło poprawne!")
            st.experimental_rerun()
        else:
            st.error("Błędne hasło. Spróbuj ponownie.")
            st.stop()
    else:
        st.stop()

# Ustawienia API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_description(keywords):
    prompt = f"Użyj poniższych słów kluczowych do stworzenia atrakcyjnego opisu produktu na Amazon.de:\n{', '.join(keywords)}."
    try:
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        st.error(f"Wystąpił błąd podczas generowania opisu: {e}")
        return ""

# Interfejs użytkownika
st.title("Generator Opisów Produktów na Amazon.de")

# Dynamiczny wybór kategorii
categories = ["Kategoria 1", "Kategoria 2", "Kategoria 3"]
selected_category = st.selectbox("Wybierz kategorię produktu", categories)

# Pobierz odpowiednie teksty
if selected_category == "Kategoria 1":
    keywords = rank_1_texts
    embeddings = rank_1_embeddings  # Jeśli nie używasz embeddingów, możesz usunąć tę linię
else:
    keywords = other_texts
    embeddings = other_embeddings  # Jeśli nie używasz embeddingów, możesz usunąć tę linię

# Generowanie opisu
if st.button("Generuj Opis"):
    if not keywords:
        st.error("Brak dostępnych słów kluczowych dla wybranej kategorii.")
    else:
        description = generate_description(keywords)
        if description:
            st.write(description)

