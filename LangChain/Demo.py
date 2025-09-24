from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192")

response = llm.invoke("What is the tallest building in the world?")
print(response.content)
