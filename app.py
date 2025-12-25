from flask import Flask, render_template, jsonify, request
from src.helper import downlad_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

embedding = downlad_embedding()
index_name = "medical-chatbot"

doc_search =PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# now retrieve the most similar from these vectorDB
retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 5})

model= ChatOpenAI(
    name="gpt-4o"
)

prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

qa_chain= create_stuff_documents_chain(llm=model, prompt=prompt)
rag_chain = create_retrieval_chain(retriever,qa_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print ("Response:" , response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)