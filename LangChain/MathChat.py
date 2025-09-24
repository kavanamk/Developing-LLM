from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGroq(model="llama3-8b-8192")

messages = [
  SystemMessage(content="You are a math tutor who provides answers with a bit of sarcasm."),
  HumanMessage(content="What is the square of 2?"),
]

response = llm.invoke(messages)
print(response.content)
