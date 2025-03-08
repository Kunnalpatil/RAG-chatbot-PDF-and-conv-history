# PDFChat: Conversational AI for PDFs  

Try the app :https://rag-chatbot-pdf-and-conv-history.streamlit.app/

PDFChat is an AI-powered web application that lets you upload PDF files and ask questions about their content in natural language. Powered by Retrieval Augmented Generation (RAG), it delivers accurate, contextually relevant answers while preserving chat history for seamless conversations.  

Built with Streamlit, LangChain, ChromaDB, Hugging Face, and Groq, PDFChat is your go-to PDF assistant—ideal for seeking quick insights from documents.  

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square) ![Chroma](https://img.shields.io/badge/Chroma-00C78C?style=flat-square) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FDEE21?style=flat-square&logo=HuggingFace) ![Groq](https://img.shields.io/badge/Groq-FF6A00?style=flat-square)  

---

## 🚀 Features  

- **PDF Upload & Processing**: Upload multiple PDFs and extract text for instant querying.  
- **Conversational Interface**: Ask questions in natural language and get concise, accurate answers.  
- **Model Selection**: Choose between AI models like Gemma2-9b-it and Llama-3.1-8b-instant.  
- **Session Management**: Use session IDs to manage chat history and start new conversations.  
- **Efficient Retrieval**: Powered by ChromaDB for fast, vector-based search of PDF content.  

---

## 🛠️ Installation  

### Prerequisites  
- Python 3.12  
- Virtual environment (recommended)  

### Steps  
1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/Kunnalpatil/RAG-chatbot-PDF-and-conv-history
     ```
2. **Create and activate a virtual environment**:  
   ```bash
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**:  
- Create a .env file in the root directory.  
- Add your Hugging Face and Groq API keys: 
   ```
   HF_TOKEN=your_huggingface_token  
   GROQ_API_KEY=your_groq_api_key
   ```

🖥️ Usage
1. **Run the Streamlit app**:  
   ```bash
   streamlit run app.py
   ```
2. **Upload your PDF files using the file uploader**.  
3. **Select an AI model from the dropdown (e.g., Gemma2-9b-it)**.  
4. **Enter a session ID to start a new conversation or continue an existing one**.  
5. **Ask questions in the chat input field and receive answers based on the PDF content**.

🧠 Technical Stack
PDFChat is built using:  
- Streamlit: For the interactive web interface.  
- LangChain: To manage the RAG workflow, including text splitting, embedding, and retrieval.  
- Chroma: As the vector database for efficient storage and retrieval of text embeddings.  
- Hugging Face Embeddings: For generating text embeddings from PDFs.  
- Groq : For models 
  
🛠️ Overcoming Deployment Challenges
- During development, I faced an SQLite version error with Chroma, which requires SQLite >= 3.35.0. To resolve this without system upgrades, I used pysqlite3-binary, a pre-compiled SQLite package.
- Solution:  
Installed pysqlite3-binary via:  
  ```bash
     pip install pysqlite3-binary
  ```
Set the environment variable in the code:  
  ```python
     os.environ['CHROMA_SQLITE_IMPL'] = 'pysqlite3'
  ``` 
This fix is essential for deployment on platforms like Streamlit Cloud, where system-level changes aren’t possible. For more details, see Chroma’s troubleshooting guide.  
🌟 Try It Out
Explore PDFChat and see how it transforms your PDF interactions! I’d love your feedback.
