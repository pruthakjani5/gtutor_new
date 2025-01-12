import streamlit as st
import requests
import os
from typing import List, Dict
from dotenv import load_dotenv
import tempfile
import json
import uuid
import clipboard
import markdown
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Load environment variables
load_dotenv()

st.set_page_config(page_title="GTUtor", page_icon="🎓", layout="wide")

# Set up API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure genai with the API key
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Create base data folder structure if it doesn't exist
data_folders = {
    "gtutor_data": [
        "vector_stores",
        "chat_histories"
    ]
}

for parent_folder, sub_folders in data_folders.items():
    parent_path = os.path.join(os.getcwd(), parent_folder)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
        print(f"Created {parent_folder} directory")
    
    for sub_folder in sub_folders:
        sub_path = os.path.join(parent_path, sub_folder)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
            print(f"Created {sub_folder} subdirectory")

# Create directories for storing data
data_folder = os.path.join(os.getcwd(), "gtutor_data")
vector_stores_folder = os.path.join(data_folder, "vector_stores")
history_folder = os.path.join(data_folder, "chat_histories")
os.makedirs(vector_stores_folder, exist_ok=True)
os.makedirs(history_folder, exist_ok=True)

# File to store subject names
subjects_file = os.path.join(data_folder, "subjects.json")

def load_subjects():
    if os.path.exists(subjects_file):
        with open(subjects_file, 'r') as f:
            return json.load(f)
    return []

def save_subjects(subjects):
    with open(subjects_file, 'w') as f:
        json.dump(subjects, f)

# Initialize vector stores dictionary
vector_stores = {}

def get_embeddings():
    """Create a new embeddings object"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

def get_or_create_vectorstore(subject: str):
    """Get or create a FAISS vector store for a subject"""
    if subject not in vector_stores:
        vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
        if os.path.exists(vector_store_path):
            embeddings = get_embeddings()
            vector_stores[subject] = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            embeddings = get_embeddings()
            vector_stores[subject] = FAISS.from_texts(texts=["Initial text"], embedding=embeddings)
            save_vectorstore(subject)
    return vector_stores[subject]

def save_vectorstore(subject: str):
    """Save vector store to disk"""
    vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
    vector_stores[subject].save_local(vector_store_path)

def load_chat_history(subject: str) -> List[Dict]:
    """Load chat history for a subject"""
    history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def save_chat_history(subject: str, history: List[Dict]):
    """Save chat history for a subject"""
    history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f)

def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
        return None

def process_pdf(pdf_content: bytes) -> List[str]:
    """Process PDF content and return text chunks"""
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_file.write(pdf_content)
    pdf_file.close()

    reader = PdfReader(pdf_file.name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    os.unlink(pdf_file.name)
    return chunks

def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
    """Add document to vector store"""
    chunks = process_pdf(pdf_content)
    embeddings = get_embeddings()
    
    if subject not in vector_stores:
        vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
    else:
        vector_stores[subject].add_texts(chunks)
    
    save_vectorstore(subject)
    st.success(f"Successfully added {source} to the {subject} vector store.")

def get_relevant_passages(query: str, subject: str, k: int = 5) -> List[str]:
    """Get relevant passages from vector store"""
    vectorstore = get_or_create_vectorstore(subject)
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

@st.cache_data
def generate_answer(prompt: str) -> str:
    """Generate answer using Gemini Pro"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        result = model.generate_content(prompt, generation_config=generation_config)
        return result.text
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# def make_rag_prompt(query: str, relevant_passages: List[str], subject: str, chat_history: List[Dict]) -> str:
#     """Construct RAG prompt"""
#     passages_text = "\n".join(f"PASSAGE {i+1}: {p}" for i, p in enumerate(relevant_passages))
#     history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in chat_history[-5:]])
    
#     return f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU students.
# Use the provided passages and your knowledge to give comprehensive answers.
# If the passages don't contain relevant information, use your general knowledge.

# Chat History:
# {history_text}

# Reference Passages:
# {passages_text}

# QUESTION: '{query}'

# ANSWER:"""

def make_rag_prompt(query: str, relevant_passages: List[str], subject: str, chat_history: List[Dict]):
    escaped_passages = [p.replace("'", "").replace('"', "").replace("\n", " ") for p in relevant_passages]
    passages_text = "\n".join(f"PASSAGE {i+1}: {p}" for i, p in enumerate(escaped_passages))
    
    history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in chat_history[-5:]])
    
    prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students.
Your role is to:
1. First check if the provided reference passages contain relevant information for the question.
2. If they do, use that information as your primary source and combine it with your knowledge to provide a comprehensive answer.
3. If they don't contain relevant information, use your own knowledge to provide a detailed answer instead of saying you cannot answer.
4. When using information from Include all relevant information and specify the page numbers, line numbers, and PDF names where the information is found. If the answer requires additional knowledge beyond the provided context, provide relevant information or insights using your knowledge. Do not provide incorrect information.
5. Always maintain an academic and informative tone.

Remember: Maintain a formal and academic tone throughout your response which is also simple to understand and informative. Answer as per required depth and weightage to the topic in subject.
You should ALWAYS provide a helpful answer. If the passages don't contain relevant information, use your general knowledge instead of saying you cannot answer.

Chat History:
{history_text}

Reference Passages:
{passages_text}

QUESTION: '{query}'

ANSWER:"""
    return prompt


def generate_llm_answer(query: str, subject: str = None, chat_history: List[Dict] = None) -> str:
    """Generate answer using LLM's knowledge without RAG"""
    history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in (chat_history or [])[-5:]])
    
    if subject:
        prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students. 
You have in-depth knowledge about GTU's curriculum and courses related to {subject}.
Please provide a comprehensive and informative answer to the following question, using your specialized knowledge and considering the chat history:

Chat History:
{history_text}

QUESTION: {query}

ANSWER:"""
    else:
        prompt = f"""You are GTUtor, a helpful and informative AI assistant for GTU (Gujarat Technological University) students. 
You have general knowledge about GTU's curriculum and various courses.
Please provide a comprehensive and informative answer to the following question, using your knowledge and considering the chat history:

Chat History:
{history_text}

QUESTION: {query}

ANSWER:"""
    return generate_answer(prompt)

def delete_message(subject: str, index: int):
    """Delete a specific message from chat history"""
    if subject in st.session_state.chat_histories:
        del st.session_state.chat_histories[subject][index]
        save_chat_history(subject, st.session_state.chat_histories[subject])
        st.rerun()

# Initialize session state
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

# Load existing subjects
subjects = load_subjects()

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
.stTextArea textarea {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("GTUtor: Dynamic Multi-Subject Chat System")

# Subject selection
subject_option = st.selectbox("Select a subject or create a new one", [""] + subjects + ["Create New Subject"])

if subject_option == "Create New Subject":
    new_subject = st.text_input("Enter the name of the new subject")
    if new_subject and new_subject not in subjects:
        subjects.append(new_subject)
        save_subjects(subjects)
        st.success(f"New subject '{new_subject}' created successfully!")
        subject_option = new_subject

selected_subject = subject_option if subject_option != "Create New Subject" else new_subject

# Load chat history for selected subject
if selected_subject and selected_subject not in st.session_state.chat_histories:
    st.session_state.chat_histories[selected_subject] = load_chat_history(selected_subject)

# File upload and processing
if selected_subject:
    st.subheader(f"Add Documents to {selected_subject}")
    uploaded_file = st.file_uploader(f"Choose a PDF file for {selected_subject}", type="pdf")
    pdf_url = st.text_input(f"Or enter a PDF URL for {selected_subject}")

    if uploaded_file:
        pdf_content = uploaded_file.read()
        with st.spinner("Processing PDF..."):
            add_document_to_vectorstore(pdf_content, uploaded_file.name, selected_subject)

    elif pdf_url:
        with st.spinner("Downloading PDF..."):
            pdf_content = download_pdf(pdf_url)
            if pdf_content:
                with st.spinner("Processing PDF..."):
                    add_document_to_vectorstore(pdf_content, pdf_url, selected_subject)

# Display chat history
if selected_subject and selected_subject in st.session_state.chat_histories:
    for i, turn in enumerate(st.session_state.chat_histories[selected_subject]):
        # User message
        st.markdown(f'''
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/>
                </div>
                <div class="message">{turn["human"]}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Delete button for message
        cols = st.columns([0.85, 0.15])
        cols[1].button("🗑️", key=f"delete_msg_{i}", on_click=lambda idx=i: delete_message(selected_subject, idx))
        
        # Bot message
        bot_message_html = markdown.markdown(turn["ai"])
        st.markdown(f'''
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/>
                </div>
                <div class="message">{bot_message_html}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Copy buttons
        cols = st.columns(2)
        cols[0].button("Copy Question", key=f"copy_q_{i}", on_click=lambda q=turn["human"]: clipboard.copy(q))
        cols[1].button("Copy Answer", key=f"copy_a_{i}", on_click=lambda a=turn["ai"]: clipboard.copy(a))

# Query input and processing
query = st.text_input("Enter your question")

if query:
    with st.spinner("Generating answer..."):
        if selected_subject:
            try:
                # Try RAG-based answer first
                relevant_texts = get_relevant_passages(query, selected_subject)
                chat_history = st.session_state.chat_histories.get(selected_subject, [])
                
                rag_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
                answer = generate_answer(rag_prompt)
                
                # Fallback to general knowledge if RAG fails
                if not answer or "unable to answer" in answer.lower() or "do not contain" in answer.lower():
                    answer = generate_llm_answer(query, selected_subject, chat_history)
            except Exception as e:
                st.error(f"Error with RAG response: {str(e)}")
                answer = generate_llm_answer(query, selected_subject, chat_history)
        else:
            # Generate general knowledge answer when no subject is selected
            answer = generate_llm_answer(query)
        
        if answer:
        # Display the new conversation
            st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{query}</div></div>', unsafe_allow_html=True)
            answer_html = markdown.markdown(answer, extensions=['tables'])
            answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
            st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
            # Copy buttons for the new conversation
            cols = st.columns(2)
            cols[0].button("Copy Question", key="copy_current_q", on_click=lambda: clipboard.copy(query))
            cols[1].button("Copy Answer", key="copy_current_a", on_click=lambda: clipboard.copy(answer))
            
            # Update chat history
            if selected_subject:
                st.session_state.chat_histories.setdefault(selected_subject, []).append({
                    'human': query,
                    'ai': answer
                })
                save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])

st.sidebar.title("🎓 GTUtor: Your AI Study Companion")
st.sidebar.markdown("""
### Welcome to GTUtor! 🌟

GTUtor is an advanced AI-powered tutoring system specifically designed for Gujarat Technological University (GTU) students. It combines Google's cutting-edge Gemini Pro AI with a sophisticated document-based knowledge system to provide:

- 📚 **Multi-Subject Learning**: Create and manage separate subjects with dedicated knowledge bases
- 🔍 **Smart Document Integration**: Upload PDFs or add via URLs to enhance subject understanding
- 💡 **Intelligent Responses**: Context-aware answers combining document knowledge with AI capabilities
- 💬 **Interactive Chat**: Dynamic conversation system with history tracking
- 🎯 **GTU-Focused Content**: Tailored specifically for GTU curriculum and courses
- 📋 **Easy Sharing**: Copy and paste functionality for questions and answers

### How to Use
1. Select or create a subject
2. Upload relevant PDF documents
3. Start asking questions!

### Getting Started
Upload your first PDF document or select an existing subject to begin your learning journey.

Made with ❤️ for GTU students
""")


# Sidebar information and buttons
st.sidebar.title("GTUtor Controls")

if selected_subject:
    db = get_or_create_vectorstore(selected_subject)
    total_docs = len(db.index_to_docstore_id)
    st.sidebar.write(f"Total documents in {selected_subject} database: {total_docs}")

    # Clear database button
    if st.sidebar.button(f"Clear {selected_subject} Database"):
        db.delete(delete_all=True)
        st.session_state.chat_histories[selected_subject] = []
        save_chat_history(selected_subject, [])
        st.sidebar.success(f"{selected_subject} database and chat history cleared successfully.")
        st.rerun()

    # Delete subject button
    if st.sidebar.button(f"Delete {selected_subject} Subject"):
        # Remove from subjects list
        subjects.remove(selected_subject)
        save_subjects(subjects)
        
        # Delete database
        db_path = os.path.join(vector_stores_folder, selected_subject.lower().replace(" ", "_"))
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
        
        # Delete chat history
        if selected_subject in st.session_state.chat_histories:
            del st.session_state.chat_histories[selected_subject]
        history_file = os.path.join(history_folder, f"{selected_subject.lower().replace(' ', '_')}_history.json")
        if os.path.exists(history_file):
            os.remove(history_file)
        
        st.sidebar.success(f"{selected_subject} subject deleted successfully.")
        st.rerun()

# Option to start a new conversation
if st.sidebar.button("Start New Conversation"):
    if selected_subject:
        st.session_state.chat_histories[selected_subject] = []
        save_chat_history(selected_subject, [])
        st.success("New conversation started.")
        st.rerun()
    else:
        st.warning("Please select a subject before starting a new conversation.")

# Function to delete a specific message
def delete_message(subject, index):
    if subject in st.session_state.chat_histories:
        del st.session_state.chat_histories[subject][index]
        save_chat_history(subject, st.session_state.chat_histories[subject])
        st.rerun()

# Add custom CSS to improve readability
st.markdown("""
<style>
.stTextArea textarea {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)
