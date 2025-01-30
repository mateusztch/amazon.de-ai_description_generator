import streamlit as st
import openai


from langchain.chat_models import ChatOpenAI


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from openai.types.error import OpenAIError  # Alternatywa dla nowszych wersji OpenAI


# Konfiguracja strony
st.set_page_config(page_title="Amazon.de - Generator Opisów", page_icon="🎉")

# Sprawdzanie autoryzacji
if 'authorized' not in st.session_state:
    st.session_state['authorized'] = False

if not st.session_state['authorized']:
    password = st.text_input("Wprowadź hasło:", type="password")
    login_button = st.button("Zaloguj się")

    if login_button:
        try:
            if password == st.secrets["bot_secrets"]["password"]:
                st.session_state['authorized'] = True
                st.success("Hasło poprawne!")
                st.experimental_rerun()
            else:
                st.error("Błędne hasło. Spróbuj ponownie.")
        except KeyError:
            st.error("Brak skonfigurowanego hasła w sekretach.")
    st.stop()

# Ustawienia API OpenAI
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Brak klucza API OpenAI w sekretach.")
    st.stop()

# Konfiguracja LangChain z ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

prompt_template = """
Przetłumacz poniższy opis produktu z języka angielskiego lub polskiego na profesjonalny opis w języku niemieckim w formie czterech punktów (bulletów):

{user_input}
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=prompt_template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_description(user_input):
    try:
        description = chain.run(user_input)
        return description.strip()
    except OpenAIError as e:
        st.error(f"Wystąpił błąd podczas generowania opisu: {e}")
        return ""
    except Exception as e:
        st.error(f"Nieoczekiwany błąd: {e}")
        return ""

# Interfejs użytkownika
st.title("Generator Opisów Produktów na Amazon.de")

st.markdown(
    """
    Wprowadź opis swojego produktu w języku angielskim lub polskim, a system przetworzy go na profesjonalny opis w języku niemieckim w formie czterech punktów.
    """
)

# Pole tekstowe dla użytkownika
user_description = st.text_area(
    "Wprowadź opis produktu:",
    height=200,
    placeholder="Napisz tutaj opis swojego produktu w języku angielskim lub polskim..."
)

# Generowanie opisu
if st.button("Generuj Opis"):
    if not user_description.strip():
        st.error("Proszę wprowadzić opis produktu przed wygenerowaniem.")
    else:
        with st.spinner("Generowanie opisu..."):
            description = generate_description(user_description)
            if description:
                # Formatowanie na cztery punkty
                bullets = description.split('\n')
                formatted_bullets = '\n'.join([f"- {bullet.strip()}" for bullet in bullets if bullet.strip()])
                st.markdown("### Opis Produktu (Niemiecki)")
                st.markdown(formatted_bullets)




