import os
import pandas as pd
import numpy as np
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load the Excel file
df = pd.read_excel("MasterDataSUV2019.xlsx", sheet_name="Sheet1")

# Convert rows into documents
documents = [
    Document(page_content=row.to_json(), metadata={"row_index": i})
    for i, row in df.iterrows()
]

# Set up the embedding model (make sure your API key is set in the environment)
embedding = OpenAIEmbeddings()

# Create the FAISS vector store
vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("faiss_suv_index")

# Load the retriever and LLM
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Example query
query = "Which SUV has the best economy and performance?"
response = rag_chain.run(query)

print("Query:", query)
print("Response:", response)

