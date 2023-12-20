



import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_TPHa4Eo92SL3pp9N8iaS3RUeaTSb02z169tVr" #GET API KEY AT : https://replicate.com/account/api-tokens

# Initialize Pinecone
pinecone.init(api_key='4245f119-1ab7-4b5d-9b0f-3c5e15aa7b92', environment='gcp-starter') #NAME OF ENVIRONMENT GET API KEY AT: https://app.pinecone.io/organizations/-NlbmMRDDwQwYcwODAQJ/projects/gcp-starter:crkzytq/keys

#from google.colab import files
#uploaded = files.upload()



# Load and preprocess the PDF document
loader = PyPDFLoader('/home/yosi/Downloads/instructions.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()

# Set up the Pinecone vector database
index_name = "vecs"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.1, "max_length": 3000}
)

"""To create a dynamic and interactive chatbot, we construct the ConversationalRetrievalChain by combining Llama2 LLM and the Pinecone vector database.

This chain enables the chatbot to retrieve relevant responses based on user queries and the chat history.
"""

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)



# Start chatting with the chatbot
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))




