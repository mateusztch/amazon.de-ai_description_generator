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
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=openai_api_key)

# Modyfikowany prompt – dodajemy zmienną {rank_embeddings}
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

# Funkcja generująca opis – przekazujemy również embeddingi
def generate_description(user_input, keywords, rank_embeddings):
    try:
        # Konwertujemy embeddingi do formy czytelnego stringa (można to modyfikować, np. wybierając tylko kilka wartości)
        if hasattr(rank_embeddings, "tolist"):
            rank_emb_str = str(rank_embeddings.tolist())
        else:
            rank_emb_str = str(rank_embeddings)
        description = chain.run(
            user_input=user_input, 
            keywords=", ".join(keywords),
            rank_embeddings=rank_emb_str
        )
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

# Generowanie opisu i wyświetlanie słów kluczowych oraz embeddingów
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



