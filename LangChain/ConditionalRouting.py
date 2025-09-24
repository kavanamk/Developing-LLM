from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

llm = ChatGroq(model="llama-3.1-8b-instant")

parse_template = PromptTemplate(
    input_variables=["raw_feedback"],
    template="Parse and clean the following customer feedback for key information:\n\n{raw_feedback}"
)
