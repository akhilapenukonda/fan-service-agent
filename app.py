import streamlit as st
import os
from dotenv import load_dotenv
import anthropic
import chromadb
from chromadb.utils import embedding_functions

# This MUST be the first Streamlit command in the script
st.set_page_config(page_title="Fan Services Assistant", page_icon="🏟️")

load_dotenv()

# --- Setup (runs once, cached so it doesn't reload on every question) ---
@st.cache_resource
def load_agent():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="eval/chroma_store")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    collection = chroma_client.get_collection(
        name="fan_services_kb",
        embedding_function=embedding_fn
    )
    return client, collection

client, collection = load_agent()

def ask_agent(question, n_chunks=3):
    results = collection.query(query_texts=[question], n_results=n_chunks)
    retrieved_chunks = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]
    context = "\n\n".join(retrieved_chunks)

    system_prompt = """You are a fan services assistant for a sports team. Answer the fan's question using ONLY the information in the provided context below.
If the context doesn't contain enough information to answer confidently, say you don't have that information rather than guessing.
Be concise and friendly."""

    user_message = f"""Context:
{context}

Fan's question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text, sources

# --- UI ---
st.title("🏟️ Fan Services Assistant")
st.caption("Ask about tickets, refunds, bag policy, parking, or stadium conduct.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            st.caption(f"📄 Sources: {', '.join(set(msg['sources']))}")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_agent(question)
        st.write(answer)
        st.caption(f"📄 Sources: {', '.join(set(sources))}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})