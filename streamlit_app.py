import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAIError, RateLimitError
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Konfiguracja strony
st.set_page_config(page_title="Amazon.de - Generator Opisów", page_icon="🎉")

# Sprawdzanie autoryzacji
if 'authorized' not in st.session_state:
    st.session_state['authorized'] = False

if not st.session_state['authorized']:
    password = st.text_input("🔑 Wprowadź hasło:", type="password")
    login_button = st.button("🔓 Zaloguj się")

    if login_button:
        try:
            if password == st.secrets["bot_secrets"]["password"]:
                st.session_state['authorized'] = True
                st.success("✅ Hasło poprawne!")
                st.rerun()  # Poprawiona wersja zamiast st.experimental_rerun()
            else:
                st.error("❌ Błędne hasło. Spróbuj ponownie.")
        except KeyError:
            st.error("⚠️ Brak skonfigurowanego hasła w sekretach.")
    st.stop()

# Ustawienia API OpenAI
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ Brak klucza API OpenAI w sekretach.")
    st.stop()

# Konfiguracja LangChain z ChatOpenAI
llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)

prompt_template = """
Przetłumacz poniższy opis produktu z języka angielskiego lub polskiego na profesjonalny opis w języku niemieckim w formie czterech punktów (bulletów):

{user_input}
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Funkcja generująca opis
def generate_description(user_input):
    try:
        description = chain.run(user_input)
        return description.strip()
    except RateLimitError:
        st.error("⏳ Przekroczono limit zapytań do OpenAI. Spróbuj ponownie później.")
        return ""
    except OpenAIError as e:
        st.error(f"❌ Wystąpił błąd podczas generowania opisu: {e}")
        return ""
    except Exception as e:
        st.error(f"⚠️ Nieoczekiwany błąd: {e}")
        return ""

# Ładowanie embedingów i słów kluczowych
@st.cache_resource
def load_embeddings():
    try:
        embeddings = np.load("rank_1_embeddings.npy")
        # Zakładam, że masz plik 'keywords_list.npy' z listą słów kluczowych
        keywords = np.load("keywords_list.npy")
        return embeddings, keywords
    except FileNotFoundError:
        st.error("⚠️ Plik z embeddingami lub słowami kluczowymi nie został znaleziony.")
        st.stop()

embeddings, keywords = load_embeddings()

# Funkcja generująca słowa kluczowe na podstawie opisu
def generate_keywords(user_input, embeddings, keywords, top_n=5):
    # Tutaj należy zaimplementować sposób przetwarzania user_input na embedding
    # Zakładam, że używasz OpenAI do tego celu
    try:
        user_embedding = llm.embed(user_input)  # Upewnij się, że metoda embed istnieje
    except AttributeError:
        st.error("❌ Metoda embed nie jest dostępna w używanym modelu.")
        return []
    
    # Oblicz podobieństwo kosinusowe
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    
    # Znajdź indeksy top_n najbardziej podobnych
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Pobierz odpowiadające słowa kluczowe
    suggested_keywords = [keywords[i] for i in top_indices]
    
    return suggested_keywords

# Interfejs użytkownika
st.title("📦 Generator Opisów Produktów na Amazon.de")

st.markdown(
    """
    ✍️ **Wprowadź opis swojego produktu w języku angielskim lub polskim**, a system przetworzy go na profesjonalny opis w języku niemieckim w formie **czterech punktów (bullet points)** oraz zasugeruje **słowa kluczowe**.
    """
)

# Dodanie rozwijanej listy kategorii pod instrukcją
category = st.selectbox(
    "📂 Wybierz kategorię:",
    ["Meble - Kuchnia, Dom i Mieszkanie"]
)

# Pole tekstowe dla użytkownika
user_description = st.text_area(
    "📝 Wprowadź opis produktu:",
    height=200,
    placeholder="Napisz tutaj opis swojego produktu w języku angielskim lub polskim..."
)

# Generowanie opisu i słów kluczowych
if st.button("🚀 Generuj Opis"):
    if not user_description.strip():
        st.error("⚠️ Proszę wprowadzić opis produktu przed wygenerowaniem.")
    else:
        with st.spinner("⏳ Generowanie opisu..."):
            description = generate_description(user_description)
            if description:
                # Formatowanie na cztery punkty
                bullets = description.split("\n")
                formatted_bullets = "\n".join([f"• {bullet.strip()}" for bullet in bullets if bullet.strip()])
                
                st.markdown("### 📌 Opis Produktu (Niemiecki)")
                st.code(formatted_bullets, language="markdown")
                
                # Generowanie słów kluczowych
                keywords = generate_keywords(user_description, embeddings, keywords)
                if keywords:
                    st.markdown("### 🔑 Sugerowane Słowa Kluczowe")
                    st.markdown(", ".join(keywords))
