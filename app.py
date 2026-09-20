import streamlit as st
from src.rag import RAGChatbot

st.set_page_config(
    page_title="Agent-as-a-Judge RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agent-as-a-Judge RAG Chatbot")
st.caption("Ask questions about the uploaded research paper.")

try:
    chatbot = RAGChatbot(model_name="llama3.2:3b")

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run: python src/ingest.py")
    st.stop()

except Exception as e:
    st.error(f"Failed to load chatbot: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about the PDF...")


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Retrieving PDF chunks and generating answer..."):

            try:
                answer, sources = chatbot.answer(
                    question,
                    top_k=4
                )

                st.markdown(answer)

                with st.expander("📚 Retrieved chunks used by RAG"):

                    for i, item in enumerate(sources, start=1):

                        st.markdown(
                            f"**Chunk {i} — PDF page "
                            f"{item['page']} — similarity "
                            f"{item['score']:.3f}**"
                        )

                        st.write(item["text"])

                        st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(f"Error generating answer: {e}")