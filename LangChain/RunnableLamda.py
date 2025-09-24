from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

llm = ChatGroq(model="llama-3.1-8b-instant")

parse_template = PromptTemplate(
    input_variables=["raw_feedback"],
    template="Parse and clean the following customer feedback for key information:\n\n{raw_feedback}"
)

summary_template = PromptTemplate(
    input_variables=["parsed_feedback"],
    template="Summarize this customer feedback in one concise sentence:\n\n{parsed_feedback}"
)

sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either 'Positive', 'Neutral', or 'Negative':\n\n{feedback}"
)

format_parsed_output = RunnableLambda(lambda output: {"parsed_feedback": output})
format_summary_output = RunnableLambda(lambda output: {"feedback": output})

# Neutral
user_feedback = "The delivery was late, and the product was damaged when it arrived. However, the customer support team was very helpful in resolving the issue quickly."

# Postive
# user_feedback = "The customer service was fantastic. The representative was friendly, knowledgeable, and resolved my issue quickly."

# Negative
# user_feedback = "I was extremely disappointed with the customer service. The representative was unhelpful and rude."

chain = parse_template | llm | format_parsed_output | summary_template | llm | format_summary_output | sentiment_template| llm | StrOutputParser()
feedback_sentiment = chain.invoke({"raw_feedback": user_feedback})

print(feedback_sentiment)
