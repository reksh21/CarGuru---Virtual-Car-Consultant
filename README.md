# CarGuru---Virtual-Car-Consultant

Overview

Document Genie is a user-friendly Streamlit application that empowers you to gain insights from your car-related PDF documents using Google's cutting-edge Generative AI model, Gemini-PRO. It leverages the Retrieval-Augmented Generation (RAG) framework to deliver accurate and contextually relevant answers to your questions.

Technical Details

Framework: Streamlit
PDF Processing: PyPDF2
Text Splitting: langchain.text_splitter.RecursiveCharacterTextSplitter
Embedding Model: langchain_google_genai.GoogleGenerativeAIEmbeddings
Vector Store: langchain_community.vectorstores.faiss.FAISS
Generative AI Model: langchain_google_genai.ChatGoogleGenerativeAI (Gemini-PRO)
Question Answering Chain: langchain.chains.question_answering.load_qa_chain
Prompt Template: langchain.prompts.PromptTemplate

Running the Application

1) Install Dependencies:
   
