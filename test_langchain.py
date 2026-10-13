
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

response = llm.invoke(
    "Explain what a company employee information finder does in two sentences."
)

print(response.content)
