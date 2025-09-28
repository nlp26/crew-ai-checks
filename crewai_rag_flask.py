from flask import Flask, request, render_template_string
from crewai import LLM, Agent, Crew, Task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.0
)

app = Flask(__name__)
PDF_PATH = "agentic_ai.pdf"  # use file upload logic for user selection if you want

CHAT_TEMPLATE = """
<!doctype html>
<title>Agentic AI PDF Chat</title>
<h1>Ask a question about your PDF</h1>
<form method=post>
  <input name=question style="width: 60%;" autofocus>
  <input type=submit value="Send">
</form>
{% if answer %}
  <div style="margin-top: 2em;">
    <strong>You:</strong> {{ question }}<br>
    <strong>Agent:</strong> {{ answer }}
  </div>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = question = None
    if request.method == "POST":
        question = request.form["question"]
        knowledge_source = PDFKnowledgeSource(file_paths=[PDF_PATH])
        agent = Agent(
            role="Domain Analyst",
            goal="Answer grounded questions using provided knowledge.",
            backstory="A careful, detail-oriented technical analyst focused on extracting the most relevant, accurate insights from Agentic AI documentation.",
            llm=llm,
            knowledge_sources=[knowledge_source],
        )
        task = Task(
            description=f"Answer the following question: {question}",
            expected_output="A precise, well-organized answer based strictly on information found in the PDF.",
            agent=agent
        )
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
        answer = crew.kickoff(inputs={"question": question})
    return render_template_string(CHAT_TEMPLATE, question=question, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
