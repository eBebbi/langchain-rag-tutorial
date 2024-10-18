from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import openai
import shutil
import streamlit as st


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

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

def check_text_start(text,text2check):
    for txt in text2check:
        if text.startswith(txt):
            return True
            break
    return False


# Hauptfunktion für die App
def main():
    # Logo hinzufügen
    st.image("swisslogo.png", width=300)  # Passe den Pfad und die Größe an
    st.title("FAQ Microsoft Office für armasuisse Immobilien")

    # Datenbank nur einmal erstellen oder laden
    db = generate_data_store()
    
    # Eingabe für die Frage
    query = st.text_input("Gib deine Frage ein:")

    if st.button("Frage beantworten"):
        if query:  # Überprüfe, ob eine Frage eingegeben wurde
            with st.spinner("Verarbeite deine Anfrage..."):
                # Suche ähnliche Dokumente in der bestehenden Chroma-Datenbank

                # neue verbesserte Version
                # Search the DB.
                results = db.similarity_search_with_relevance_scores(query, k=3)
                if len(results) == 0 or results[0][1] < 0.7:
                    print(f"Unable to find matching results.")
                    aiAnswer = f"Tut mir leid, aber ich kann die Frage '{query}' nicht beantworten.\nÄndere bitte die Frage und versuche es erneut."
                    st.write(aiAnswer)
                    return
                else:

                    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    prompt = prompt_template.format(context=context_text, question=query)
                    print(prompt)

                    model = ChatOpenAI()
                    response_text = model.predict(prompt)

                    print(f"********* {response_text} ***********")

                    noMatchText = []
                    noMatchText.append("Sorry, I cannot provide an answer to the question.")
                    noMatchText.append("Unable to find matching results.")
                    if check_text_start(response_text,noMatchText):
                        aiAnswer = f"Tut mir leid, aber ich kann die Frage '{query}' nicht beantworten.\nÄndere bitte die Frage und versuche es erneut."
                    else:
                        aiAnswer = response_text

                    sources = [doc.metadata.get("source", None) for doc, _score in results]
                    #formatted_response = f"Response: {response_text}\nSources: {sources}"
                    #myResponse = f"\n\n\n\nFrage: {query}\nAntwort:{response_text}"
                    finalResponse = f"{aiAnswer}"
                    # print(formatted_response)
                    print(finalResponse)
                    st.write(finalResponse)
                    return
        else:
            st.write("Bitte gib eine Frage ein.")

if __name__ == "__main__":
    main()