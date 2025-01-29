import json
import streamlit as st
import openai
from openai.error import OpenAIError

# Konfiguracja strony
st.set_page_config(page_title="Amazon.de - Generator Opisów", page_icon="🎉")

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

def generate_description(user_input):
    prompt = (
        "Przetłumacz poniższy opis produktu z języka angielskiego lub polskiego na "
        "profesjonalny opis w języku niemieckim w formie czterech punktów (bulletów):\n\n"
        f"{user_input}"
    )
    try:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
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
                st.markdown("### Opis Produktu (Niemiecki)")
                st.markdown(f"- {description.replace('\n', '\n- ')}")


