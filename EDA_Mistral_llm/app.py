import streamlit as st
from nltk.tokenize import WhitespaceTokenizer

st.set_page_config(page_title="NLP Tokenizer App")

st.title("NLP Tokenizer App")
st.subheader("Whitespace Tokenizer using NLTK and Streamlit")

text = st.text_area("Enter some text here:")

if st.button("Tokenize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        tokenizer = WhitespaceTokenizer()
        tokens = tokenizer.tokenize(text)
        st.subheader("Here are the tokens:")
        st.write(tokens)

