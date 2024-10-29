# CarGuru---Your AI-powered Car Information Assistant

Overview

Car Guru is a user-friendly Streamlit application that helps you unlock insights from your car-related PDF documents using Google's cutting-edge Generative AI model, Gemini-PRO.  Leveraging the Retrieval-Augmented Generation (RAG) framework, it delivers accurate and contextually relevant answers to your questions about the car details in your PDFs.

How to Use Document Genie

1) Obtain Google API Key: Create a free Google Cloud project and enable the Generative AI API to get an API key. Visit https://makersuite.google.com/app/apikey for instructions.
2) Upload Your Car Documents: Select and upload the PDF files containing car information you want to analyze. Document Genie accepts multiple files for comprehensive insights. Click "Submit & Process": Initiate the processing step to build a searchable knowledge base from the uploaded documents.
3) Ask Questions Related to Cars: Once processing is complete, interact with the chatbot by typing your questions about the car details mentioned in the documents. Document Genie will provide answers that are directly relevant to your queries.

Technical Details

1) Framework: Streamlit

2) PDF Processing: PyPDF2

3) Text Splitting: langchain.text_splitter.RecursiveCharacterTextSplitter
4) Embedding Model: langchain_google_genai.GoogleGenerativeAIEmbeddings
5) Vector Store: langchain_community.vectorstores.faiss.FAISS
6) Generative AI Model: langchain_google_genai.ChatGoogleGenerativeAI (Gemini-PRO)
7) Question Answering Chain: langchain.chains.question_answering.load_qa_chain
8) Prompt Template: langchain.prompts.PromptTemplate

Running the Application

1) Install Dependencies:
   pip install -r requirements.txt
2) Obtain Google API Key (as mentioned in "How to Use")
3) Run the script:
   streamlit run Streamlit_app.py

<img width="1500" alt="Screenshot 2024-10-29 at 9 33 35 AM" src="https://github.com/user-attachments/assets/51c7949f-8776-472f-93a8-7fc0898b4384">


   
