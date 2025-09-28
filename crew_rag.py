import streamlit as st
from crewai import Agent, Task, Crew, Process
import PyPDF2
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

st.title("Agentic AI Crew: Multi-Agent PDF RAG Chat")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question about your PDF:")

if uploaded_file:
    st.success("PDF uploaded! Extracting text...")
    text = extract_text_from_pdf(uploaded_file)
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
    vectordb = Chroma(
        persist_directory="pdf_vectordb",
        embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    )
    vectordb.add_texts(chunks)
    st.success("Vector DB ready!")

    summarizer = Agent(
        role="Summarizer",
        goal="Summarize major topics and context from the PDF.",
        backstory="Quickly highlights what matters in the doc.",
        verbose=True
    )
    retriever = Agent(
        role="Retriever",
        goal="Find the most relevant sections for any user query.",
        backstory="Expert locator, pinpointing the best passages.",
        verbose=True
    )
    explainer = Agent(
        role="Explainer",
        goal="Break down answers to simple, clear explanations.",
        backstory="Makes complex info accessible to anyone.",
        verbose=True
    )
    relevance_task = Task(
        description=f"Find passages in the PDF related to: {query}",
        expected_output="List of most relevant paragraphs.",
        agent=retriever
    )
    summary_task = Task(
        description="Create a concise summary of the PDF's major topics.",
        expected_output="Section-by-section PDF summary.",
        agent=summarizer
    )
    explain_task = Task(
        description=f"Take the most relevant passages for question '{query}' and explain them simply.",
        expected_output="A detailed, user-friendly answer for the question.",
        agent=explainer
    )
    crew = Crew(
        agents=[summarizer, retriever, explainer],
        tasks=[summary_task, relevance_task, explain_task],
        process=Process.sequential,
        verbose=True
    )

    st.success("Agents ready!")

    if query:
        st.info("Agents are collaborating...")
        results = crew.kickoff(inputs={'user_input': query})
        st.write("PDF Summary:")
        st.write(results.get("section-by-section PDF summary", "No summary"))
        st.write("Relevant Passages:")
        st.write(results.get("List of most relevant paragraphs", "No relevant passages"))
        st.write("Explained Answer:")
        st.write(results.get("A detailed, user-friendly answer for the question", "No explanation"))

