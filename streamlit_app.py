import numpy as np
import json
import streamlit as st
import openai
import numpy as np

# Załaduj embeddingi
rank_1_embeddings = np.load('rank_1_embeddings.npy')
other_embeddings = np.load('other_embeddings.npy')

# Załaduj odpowiadające teksty
with open('rank_1_texts.json', 'r', encoding='utf-8') as f:
    rank_1_texts = json.load(f)

with open('other_texts.json', 'r', encoding='utf-8') as f:
    other_texts = json.load(f)
    
# Website config
st.set_page_config(page_title="Amazon.de - generator opisów", page_icon="🎉")

# Authorization status tracking
if 'authorized' not in st.session_state:
    st.session_state['authorized'] = False

# Password input if not authorized
if not st.session_state['authorized']:
    password = st.text_input("Type your password:", type="password")
    login_button = st.button("Log-in")

    if login_button:
        if password == st.secrets["bot_secrets"]["password"]:
            st.session_state['authorized'] = True
            st.success("Password correct!")
            st.experimental_rerun()  
        else:
            st.error("Error. Try again later.")
            st.stop()  
    else:
        st.stop()  
# Ustawienia API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_description(keywords, embeddings):
    prompt = f"Użyj poniższych słów kluczowych do stworzenia atrakcyjnego opisu produktu na Amazon.de:\n{', '.join(keywords)}.\n\nEmbeddingi: {embeddings.tolist()}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Przykład użycia w Streamlit
st.title("Generator Opisów Produktów na Amazon.de")

# Wybierz kategorię lub inne kryteria
selected_category = st.selectbox("Wybierz kategorię produktu", ["Kategoria 1", "Kategoria 2", "Kategoria 3"])

# Pobierz odpowiednie embeddingi i teksty
if selected_category == "Kategoria 1":
    keywords = rank_1_texts
    embeddings = rank_1_embeddings
else:
    keywords = other_texts
    embeddings = other_embeddings

# Generowanie opisu
if st.button("Generuj Opis"):
    description = generate_description(keywords, embeddings)
    st.write(description)
