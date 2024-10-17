from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import openai
import shutil
import streamlit as st

# Lade die Umgebungsvariablen
load_dotenv("settings.env")

# OpenAI API Key laden
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

CHROMA_PATH = "chroma"
# Angepasster Pfad zu deinen deutschen Dokumenten
DATA_PATH = "data/german_docs"

# Funktion zur Datenbankerstellung (nur wenn sie noch nicht existiert)
def generate_data_store():
    # Überprüfe, ob die Datenbank bereits existiert
    if os.path.exists(CHROMA_PATH):
        print("Datenbank existiert bereits. Lade vorhandene Datenbank.")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    else:
        print("Erstelle eine neue Datenbank.")
        documents = load_documents()
        chunks = split_text(documents)
        db = save_to_chroma(chunks)
    
    return db

# Dokumente laden
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    #loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

# Text splitten
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Chroma-Datenbank speichern
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    return db

# Hauptfunktion für die App
def main():
    st.title("Textdatenbank-Erstellung und Abfrage mit Langchain")
    
    # Datenbank nur einmal erstellen oder laden
    db = generate_data_store()
    
    # Eingabe für die Frage
    query = st.text_input("Gib deine Frage ein:")

    if st.button("Frage beantworten"):
        if query:  # Überprüfe, ob eine Frage eingegeben wurde
            with st.spinner("Verarbeite deine Anfrage..."):
                # Suche ähnliche Dokumente in der bestehenden Chroma-Datenbank
                results = db.similarity_search(query)
                
                if results:
                    st.write("Ergebnisse:")
                    for result in results:
                        st.write(result.page_content)
                else:
                    st.write("Keine ähnlichen Dokumente gefunden.")
        else:
            st.write("Bitte gib eine Frage ein.")

if __name__ == "__main__":
    main()