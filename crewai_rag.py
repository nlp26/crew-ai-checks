from crewai import LLM, Agent, Crew, Task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# 1. Setup LLM (Ollama must be running with the correct models pulled!)
llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.0
)

# 2. PDF source (adjust filename/path if needed)
knowledge_source = PDFKnowledgeSource(file_paths=["agentic_ai.pdf"])

# 3. Agent (backstory REQUIRED)
agent = Agent(
    role="Domain Analyst",
    goal="Answer grounded questions using provided knowledge.",
    backstory="A careful, detail-oriented technical analyst focused on extracting the most relevant, accurate insights from any documentation relating to Agentic AI.",
    llm=llm,
    knowledge_sources=[knowledge_source],
)

# 4. Task (expected_output REQUIRED)
task = Task(
    description="Answer the following question: {question}",
    expected_output="A precise, well-organized answer based strictly on information found in the PDF.",
    agent=agent
)

# 5. Crew with embedder config (Ollama for embeddings)
crew = Crew(
    agents=[agent],
    tasks=[task],
    embedder={
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large",
            "url": "http://localhost:11434/api/embeddings"
        }
    }
)

# 6. Run question-answering
result = crew.kickoff(inputs={"question": "Key takeaways?"})
print(result)