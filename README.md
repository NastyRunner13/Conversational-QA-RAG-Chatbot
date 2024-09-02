# Conversational RAG with PDF Uploads and Chat History

This application is a Conversational Retrieval-Augmented Generation (RAG) tool built using Streamlit and the LangChain framework. It allows users to upload PDF files, and chat with the content within them, while maintaining a chat history across sessions. The application is particularly useful for question-answering tasks over large documents, with the ability to remember the context of the conversation.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- LangChain
- Chroma
- Hugging Face Transformers
- Dotenv

## Installation

1. Clone the repository from GitHub:

```
git clone https://github.com/NastyRunner13/Conversational-QA-RAG-Chatbot
```

2. Change to the project directory:

```
Conversational-QA-RAG-Chatbot
```

3. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

5. Create a `.env` file in the project directory and add your Groq API key:

```
HF_TOKEN=your_hugging_face_token
GROQ_API_KEY=your_groq_api_key
```

## Running the Application

1. Start the Streamlit application:

```
streamlit run app.py
```

2. The application will open in your default web browser.

3. Upload one or more PDF files using the file uploader.

4. Enter a session ID (or use the default "default_session") and your Groq API key.

5. Ask a question in the input field, and the application will retrieve relevant information from the uploaded PDFs and provide an answer.

6. The chat history will be displayed below the input field.

## Features

- Upload multiple PDF files
- Persistent chat history across sessions
- Retrieval-Augmented Generation (RAG) model for answering questions
- Contextual understanding of questions based on chat history
- Leverages LangChain, Chroma, and Hugging Face Transformers libraries

## Future Enhancements

- Improved Error Handling: Enhancing the application's robustness with more comprehensive error handling.
- Multi-format Support: Expanding support for other document formats like Word, Excel, etc.
- Advanced Search Capabilities: Adding functionality for more complex queries and searches within the document content.

## Contributing

If you find any issues or have suggestions for improvements, feel free to create a new issue or submit a pull request on the GitHub repository.
