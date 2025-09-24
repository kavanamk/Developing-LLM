Langcafrom langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")

messages=[{"role": "system", "content": "You are a helpful assistant."}]

user_message = {"role": "user", "content": "Hi, my name is Alex."}

response = llm.invoke(messages)
print(response.content)
