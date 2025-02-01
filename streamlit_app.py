import streamlit as st
import numpy as np
import time  # potrzebne do sleep
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
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# Zmodyfikowany prompt – dodajemy zmienną {rank_embeddings}
prompt_template = """
Przetłumacz poniższy opis produktu z języka angielskiego lub polskiego na profesjonalny opis w języku niemieckim w formie pięciu punktów (bulletów).
Użyj następujących słów kluczowych w opisie: {keywords}.
Dodatkowo, weź pod uwagę następujący kontekst embeddingów opisów z najlepszego rankingu: {rank_embeddings}.
Użyj od 1200 do 1300 słów.

Opis:
{user_input}
"""

# Rozszerzamy input_variables o "rank_embeddings"
prompt = PromptTemplate(
    input_variables=["user_input", "keywords", "rank_embeddings"],
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt)

def summarize_embeddings(embeddings, num=5):
    """
    Funkcja wybiera pierwsze num embeddingów i zwraca ich reprezentację jako string.
    Dzięki temu prompt nie będzie zawierał całego (prawdopodobnie bardzo dużego) zbioru embeddingów.
    """
    try:
        # Jeśli embeddings jest tablicą numpy, wybieramy pierwsze num wierszy
        summarized = embeddings[:num]
        return str(summarized.tolist())
    except Exception as e:
        st.error(f"⚠️ Błąd podczas przetwarzania embeddingów: {e}")
        return "Brak danych embeddingów"

# Funkcja generująca opis z mechanizmem retry dla RateLimitError
def generate_description(user_input, keywords, rank_embeddings):
    # Używamy skróconej wersji embeddingów
    rank_emb_str = summarize_embeddings(rank_embeddings)
    
    max_retries = 3
    retry_delay = 1  # początkowe opóźnienie w sekundach

    for attempt in range(max_retries):
        try:
            description = chain.run(
                user_input=user_input, 
                keywords=", ".join(keywords),
                rank_embeddings=rank_emb_str
            )
            return description.strip()
        except RateLimitError:
            st.error("⏳ Przekroczono limit zapytań do OpenAI. Próba ponownego połączenia...")
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff
        except OpenAIError as e:
            st.error(f"❌ Wystąpił błąd podczas generowania opisu: {e}")
            return ""
        except Exception as e:
            st.error(f"⚠️ Nieoczekiwany błąd: {e}")
            return ""
    st.error("⏳ Przekroczono limit zapytań do OpenAI po kilku próbach. Spróbuj ponownie później.")
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

# Ładowanie embeddingów (rank_1_embeddings)
@st.cache_resource
def load_rank_embeddings():
    try:
        embeddings = np.load("rank_1_embeddings.npy", allow_pickle=True)
        return embeddings
    except FileNotFoundError:
        st.error("⚠️ Plik z embeddingami nie został znaleziony.")
        st.stop()
    except ValueError as ve:
        st.error(f"❌ Błąd podczas ładowania pliku Numpy: {ve}")
        st.stop()

rank_embeddings = load_rank_embeddings()

# Interfejs użytkownika
st.title("📦 Generator Opisów Produktów na Amazon.de")

st.markdown(
    """
    ✍️ **Wprowadź opis swojego produktu w języku angielskim lub polskim**, a system przetworzy go na profesjonalny opis w języku niemieckim w formie **pięciu punktów (bullet points)** oraz wykorzysta **istniejące słowa kluczowe** i **embeddingi najlepszych opisów**.
    """
)

# Rozwijana lista kategorii
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

# Generowanie opisu i wyświetlanie
if st.button("🚀 Generuj Opis"):
    if not user_description.strip():
        st.error("⚠️ Proszę wprowadzić opis produktu przed wygenerowaniem.")
    else:
        with st.spinner("⏳ Generowanie opisu..."):
            description = generate_description(user_description, keywords, rank_embeddings)
            if description:
                # Formatowanie – zakładamy, że punkty oddzielone są nowymi liniami
                bullets = description.split("\n")
                formatted_bullets = "\n".join([bullet.strip() for bullet in bullets if bullet.strip()])
                
                st.markdown("### 📌 Opis Produktu")
                st.code(formatted_bullets, language="markdown")



