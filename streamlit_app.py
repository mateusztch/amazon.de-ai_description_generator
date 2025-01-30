import streamlit as st
import numpy as np
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
                st.rerun() 
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
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)  # Upewnij się, że model jest prawidłowy

prompt_template = """
Przetłumacz poniższy opis produktu z języka angielskiego lub polskiego na profesjonalny opis w języku niemieckim w formie czterech punktów (bulletów). Użyj następujących słów kluczowych w opisie: {keywords}.

Opis:
{user_input}
"""

prompt = PromptTemplate(
    input_variables=["user_input", "keywords"],
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Funkcja generująca opis
def generate_description(user_input, keywords):
    try:
        # Przekazujemy zarówno user_input, jak i keywords do łańcucha
        description = chain.run(user_input=user_input, keywords=", ".join(keywords))
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

# Ładowanie słów kluczowych
@st.cache_resource
def load_keywords():
    try:
        keywords = np.load("keywords.npy", allow_pickle=True)
        return keywords
    except FileNotFoundError:
        st.error("⚠️ Plik ze słowami kluczowymi nie został znaleziony.")
        st.stop()
    except ValueError as ve:
        st.error(f"❌ Błąd podczas ładowania pliku Numpy: {ve}")
        st.stop()

keywords = load_keywords()

# Interfejs użytkownika
st.title("📦 Generator Opisów Produktów na Amazon.de")

st.markdown(
    """
    ✍️ **Wprowadź opis swojego produktu w języku angielskim lub polskim**, a system przetworzy go na profesjonalny opis w języku niemieckim w formie **czterech punktów (bullet points)** oraz wykorzysta **istniejące słowa kluczowe**.
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

# Generowanie opisu i wyświetlanie słów kluczowych
if st.button("🚀 Generuj Opis"):
    if not user_description.strip():
        st.error("⚠️ Proszę wprowadzić opis produktu przed wygenerowaniem.")
    else:
        with st.spinner("⏳ Generowanie opisu..."):
            description = generate_description(user_description, keywords)
            if description:
                # Formatowanie na cztery punkty
                bullets = description.split("\n")
                formatted_bullets = "\n".join([f"{bullet.strip()}" for bullet in bullets if bullet.strip()])
                
                st.markdown("### 📌 Opis Produktu")
                st.code(formatted_bullets, language="markdown")
                


