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
from datetime import datetime
import pytz
import base64

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

# def save_chat_history(subject: str, history: List[Dict]):
#     """Save chat history for a subject"""
#     history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
#     with open(history_file, 'w') as f:
#         json.dump(history, f)

def save_chat_history(subject: str, history: List[Dict]):
    """Save chat history for a subject with timestamps"""
    history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, default=str)

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

#  Add this function to get document count
def get_document_count(subject: str) -> int:
    """Get the number of documents in a subject's vector store"""
    try:
        vectorstore = get_or_create_vectorstore(subject)
        if hasattr(vectorstore, 'index_to_docstore_id'):
            return len(vectorstore.index_to_docstore_id)
        return 0
    except Exception:
        return 0

# def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
#     """Add document to vector store"""
#     chunks = process_pdf(pdf_content)
#     embeddings = get_embeddings()
    
#     if subject not in vector_stores:
#         vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     else:
#         vector_stores[subject].add_texts(chunks)
    
#     save_vectorstore(subject)
#     st.success(f"Successfully added {source} to the {subject} vector store.")

# Modify the add_document_to_vectorstore function to track document count
def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
    """Add document to vector store with document tracking"""
    chunks = process_pdf(pdf_content)
    embeddings = get_embeddings()
    
    if subject not in vector_stores:
        vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
    else:
        vector_stores[subject].add_texts(chunks)
    
    # Update document count in metadata
    save_vectorstore(subject)
    
    # Update the session state to reflect new document count
    if 'document_counts' not in st.session_state:
        st.session_state.document_counts = {}
    st.session_state.document_counts[subject] = get_document_count(subject)
    
    st.success(f"Successfully added {source} to the {subject} vector store.")
    
# Add this function to clear the database properly
def clear_subject_database(subject: str):
    """Clear all data for a subject"""
    try:
        # Clear vector store
        embeddings = get_embeddings()
        vector_stores[subject] = FAISS.from_texts(texts=["Initial text"], embedding=embeddings)
        save_vectorstore(subject)
        
        # Clear chat history
        if subject in st.session_state.chat_histories:
            st.session_state.chat_histories[subject] = []
        save_chat_history(subject, [])
        
        # Reset document count
        if 'document_counts' in st.session_state:
            st.session_state.document_counts[subject] = 0
            
        return True
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        return False

# Update the chat message display to include timestamps
def display_chat_message(turn: Dict, index: int, is_current: bool = False):
    """Display a chat message with timestamp"""
    timestamp = turn.get('timestamp', 'No timestamp')
    if isinstance(timestamp, str) and timestamp != 'No timestamp':
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%d-%m-%y %I:%M %p")
        except:
            formatted_time = timestamp
    else:
        formatted_time = timestamp
        
    # User message with timestamp
    message_html = markdown.markdown(turn["human"], extensions=['tables', 'fenced_code', 'codehilite'])
    st.markdown(f'''
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/>
            </div>
            <div class="message">
                <div class="timestamp" style="font-size: 0.9em;">{formatted_time}</div>
                <div class="content">{message_html}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

def get_chat_history_download_link(subject: str, chat_history: list):
    """Generate a download link for entire chat history"""
    if not chat_history:
        return None, None
    
    # Format chat history for export
    export_data = {
        'subject': subject,
        'exported_at': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
        'total_messages': len(chat_history),
        'messages': chat_history
    }
    
    # Convert to JSON string with proper formatting
    json_str = json.dumps(export_data, indent=2, default=str)
    b64 = base64.b64encode(json_str.encode()).decode()
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{subject.lower().replace(' ', '_')}_complete_history_{timestamp}.json"
    
    return json_str, filename

def display_copy_buttons(question: str, answer: str, i: int):
    # Create three equal columns for center alignment
    # cols = st.columns([1, 1, 1])
    
    # # Add empty column for centering
    # with cols[0]:
    #     if st.button("Copy Question", key=f"copy_q_{i}"):
    #         st.code(question, language=None)
    
    # with cols[1]:
    #     if st.button("Copy Answer", key=f"copy_a_{i}"):
    #         st.code(answer, language=None)
    cols = st.columns(3)
    
    # Display question in a hidden code block that can be copied
    if cols[0].button("Copy Question", key=f"copy_q_{i}"):
        st.code(question, language=None)
        
    # Display answer in a hidden code block that can be copied
    if cols[1].button("Copy Answer", key=f"copy_a_{i}"):
        st.code(answer, language=None)
    
    # Add download button for the entire chat history if this is the last message
    if selected_subject and i == len(st.session_state.chat_histories.get(selected_subject, [])) - 1:
        chat_history = st.session_state.chat_histories.get(selected_subject, [])
        json_data, filename = get_chat_history_download_link(selected_subject, chat_history)
        if json_data and filename:
            with cols[2]:
                st.download_button(
                    "⬇️ Download History",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    key=f"download_history_{i}"
                )

def get_relevant_passages(query: str, subject: str, k: int = 5) -> List[str]:
    """Get relevant passages from vector store"""
    vectorstore = get_or_create_vectorstore(subject)
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

@st.cache_data
def generate_answer(prompt: str) -> str:
    """Generate answer using Gemini Pro"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
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

Remember: Maintain a formal and academic tone throughout your response which is also simple to understand and informative. Answer as per required depth and weightage to the topic in subject.  If asked for any code or program then reply only code in txt format that can be directly copied.
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
You have in-depth knowledge about GTU's curriculum and courses related to {subject}. You should always provide an answer. If asked for any code or program then reply only code in txt format that can be directly copied.
Please provide a comprehensive and informative answer to the following question, using your specialized knowledge and considering the chat history:

Chat History:
{history_text}

QUESTION: {query}

ANSWER:"""
    else:
        prompt = f"""You are GTUtor, a helpful and informative AI assistant for GTU (Gujarat Technological University) students. 
You have general knowledge about GTU's curriculum and various courses.  If asked for any code or program then reply only code in txt format that can be directly copied.
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
if 'query' not in st.session_state:
    st.session_state.query = ""

def submit_query():
    st.session_state.query = st.session_state.query_input

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
# st.title("GTUtor: Dynamic Multi-Subject Chat System")
st.title("🎓 GTUtor: Your AI-Powered Academic Companion")
# Create an engaging subtitle
st.markdown("*Elevating Education with Intelligent Conversations*", unsafe_allow_html=True)

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
        # Use the display_chat_message function to show human question with timestamp
        display_chat_message(turn, i)
        
        # Delete button for message
        cols = st.columns([0.85, 0.15])
        cols[1].button("🗑️", key=f"delete_msg_{i}", on_click=lambda idx=i: delete_message(selected_subject, idx))
        
        # Bot message
        # Convert markdown to HTML with proper extensions and code formatting
        bot_message_html = markdown.markdown(turn["ai"], extensions=['tables'])
        
        # Clean up code block formatting
        bot_message_html = bot_message_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")


        st.markdown(f'''
            <div class="chat-message bot">
            <div class="avatar">
                <img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/>
            </div>
            <div class="message">
                <div class="content">{bot_message_html}</div>
            </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Copy buttons
        cols = st.columns(2)
        # cols[0].button("Copy Question", key=f"copy_q_{i}", on_click=lambda q=turn["human"]: clipboard.copy(q))
        # cols[1].button("Copy Answer", key=f"copy_a_{i}", on_click=lambda a=turn["ai"]: clipboard.copy(a))
        display_copy_buttons(turn["human"], turn["ai"], i)


# Query input and processing
# query = st.text_input("Enter your question")
query = st.text_input(
    "Enter your question",
    key="query_input",
    on_change=submit_query
)

# if query:
#     with st.spinner("Generating answer..."):
#         if selected_subject:
#             try:
#                 # Try RAG-based answer first
#                 relevant_texts = get_relevant_passages(query, selected_subject)
#                 chat_history = st.session_state.chat_histories.get(selected_subject, [])
#                 # Display chat history with proper rendering
#                 for i, turn in enumerate(chat_history):
#                     # Format the AI response with markdown and code block handling
#                     ai_response_html = markdown.markdown(turn['ai'], extensions=['tables'])
#                     ai_response_html = ai_response_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
                
#                 rag_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
#                 answer = generate_answer(rag_prompt)
                
#                 # Fallback to general knowledge if RAG fails
#                 if not answer or "unable to answer" in answer.lower() or "do not contain" in answer.lower():
#                     answer = generate_llm_answer(query, selected_subject, chat_history)
#             except Exception as e:
#                 st.error(f"Error with RAG response: {str(e)}")
#                 answer = generate_llm_answer(query, selected_subject, chat_history)
#         else:
#             # Generate general knowledge answer when no subject is selected
#             answer = generate_llm_answer(query)
        
#         if answer:
#         # Display the new conversation
#             st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{query}</div></div>', unsafe_allow_html=True)
#             answer_html = markdown.markdown(answer, extensions=['tables'])
#             answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
#             st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
#             # Copy buttons for the new conversation
#             cols = st.columns(2)
#             cols[0].button("Copy Question", key="copy_current_q", on_click=lambda: clipboard.copy(query))
#             cols[1].button("Copy Answer", key="copy_current_a", on_click=lambda: clipboard.copy(answer))
            
#             # Update chat history
#             if selected_subject:
#                 st.session_state.chat_histories.setdefault(selected_subject, []).append({
#                     'human': query,
#                     'ai': answer
#                 })
#                 save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])

# if query:
if st.session_state.query:
    with st.spinner("🤖 GTUtor is thinking..."):
        # Add timestamp to the new message
        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        if selected_subject:
            try:
                relevant_texts = get_relevant_passages(query, selected_subject)
                chat_history = st.session_state.chat_histories.get(selected_subject, [])
                rag_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
                answer = generate_answer(rag_prompt)
                
                if not answer or "unable to answer" in answer.lower() or "do not contain" in answer.lower() or "sorry" in answer.lower() or "i apologize" in answer.lower() or "cannot help" in answer.lower() or "no information" in answer.lower() or "insufficient data" in answer.lower() or "not enough context" in answer.lower():
                    answer = generate_llm_answer(query, selected_subject, chat_history)
            except Exception as e:
                st.error(f"Error with RAG response: {str(e)}")
                answer = generate_llm_answer(query, selected_subject, chat_history)
        else:
            answer = generate_llm_answer(query)
        
        if answer:
            # Create new message with timestamp
            new_message = {
                'human': query,
                'ai': answer,
                'timestamp': current_time.isoformat()
            }
            if answer:
                # Display the new conversation
                st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{query}</div></div>', unsafe_allow_html=True)
                answer_html = markdown.markdown(answer, extensions=['tables'])
                answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
                st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
                # Copy buttons for the new conversation
                cols = st.columns(2)
                # cols[0].button("Copy Question", key="copy_current_q", on_click=lambda: clipboard.copy(query))
                # cols[1].button("Copy Answer", key="copy_current_a", on_click=lambda: clipboard.copy(answer))
                display_copy_buttons(turn["human"], turn["ai"], 0)
                
            # Update chat history
            if selected_subject:
                st.session_state.chat_histories.setdefault(selected_subject, []).append(new_message)
                save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])
            # After processing, clear the query
            st.session_state.query = ""
            # Force a rerun to clear the input field
            # st.rerun()

# Add these UI enhancements
st.markdown("""
<style>
/* Enhanced chat message styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.chat-message:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.timestamp {
    font-size: 0.8em;
    color: #888;
    margin-bottom: 0.5rem;
}
/* Enhanced button styling */
.stButton>button {
    border-radius: 20px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
/* Enhanced sidebar styling */
.sidebar .sidebar-content {
    background-color: #f8f9fa;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)
# border-left: 4px solid #4CAF50;


# Update the sidebar information display
if selected_subject:
    st.sidebar.markdown(f"""
    ### 📊 Current Subject Stats
    - **Subject**: {selected_subject}
    - **Documents**: {get_document_count(selected_subject)}
    - **Messages**: {len(st.session_state.chat_histories.get(selected_subject, []))}
    """)


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
<div>
        <h3>🚀 Getting Started</h3>
        <ol style='font-size: 1.1rem;'>
            <li>Select or create a subject from the dropdown below</li>
            <li>Upload your study materials (PDF format)</li>
            <li>Start asking questions and learn interactively!</li>
        </ol>
</div>

Made with ❤️ for GTU students
""",unsafe_allow_html=True)


# Sidebar information and buttons
st.sidebar.title("GTUtor Controls")

if selected_subject:
    db = get_or_create_vectorstore(selected_subject)
    total_docs = len(db.index_to_docstore_id)
    st.sidebar.write(f"Total documents in {selected_subject} database: {total_docs}")

    # Clear database button
    if st.sidebar.button(f"Clear {selected_subject} Database"):
        # Create a new empty vectorstore
        embeddings = get_embeddings()
        vector_stores[selected_subject] = FAISS.from_texts(texts=["Initial text"], embedding=embeddings)
        save_vectorstore(selected_subject)
        # Clear chat history
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


# import streamlit as st
# import requests
# import os
# from typing import List, Dict
# from dotenv import load_dotenv
# import tempfile
# import json
# import uuid
# import clipboard
# import markdown
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# st.set_page_config(page_title="GTUtor", page_icon="🎓", layout="wide")

# # Set up API key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in .env file")

# # Configure genai with the API key
# genai.configure(api_key=GOOGLE_API_KEY)
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# # Create base data folder structure if it doesn't exist
# data_folders = {
#     "gtutor_data": [
#         "vector_stores",
#         "chat_histories"
#     ]
# }

# for parent_folder, sub_folders in data_folders.items():
#     parent_path = os.path.join(os.getcwd(), parent_folder)
#     if not os.path.exists(parent_path):
#         os.makedirs(parent_path)
#         print(f"Created {parent_folder} directory")
    
#     for sub_folder in sub_folders:
#         sub_path = os.path.join(parent_path, sub_folder)
#         if not os.path.exists(sub_path):
#             os.makedirs(sub_path)
#             print(f"Created {sub_folder} subdirectory")

# # Create directories for storing data
# data_folder = os.path.join(os.getcwd(), "gtutor_data")
# vector_stores_folder = os.path.join(data_folder, "vector_stores")
# history_folder = os.path.join(data_folder, "chat_histories")
# os.makedirs(vector_stores_folder, exist_ok=True)
# os.makedirs(history_folder, exist_ok=True)

# # File to store subject names
# subjects_file = os.path.join(data_folder, "subjects.json")

# def load_subjects():
#     if os.path.exists(subjects_file):
#         with open(subjects_file, 'r') as f:
#             return json.load(f)
#     return []

# def save_subjects(subjects):
#     with open(subjects_file, 'w') as f:
#         json.dump(subjects, f)

# # Initialize vector stores dictionary
# vector_stores = {}

# def get_embeddings():
#     """Create a new embeddings object"""
#     return GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=GOOGLE_API_KEY
#     )

# def get_or_create_vectorstore(subject: str):
#     """Get or create a FAISS vector store for a subject"""
#     if subject not in vector_stores:
#         vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
#         if os.path.exists(vector_store_path):
#             embeddings = get_embeddings()
#             vector_stores[subject] = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
#         else:
#             embeddings = get_embeddings()
#             vector_stores[subject] = FAISS.from_texts(texts=["Initial text"], embedding=embeddings)
#             save_vectorstore(subject)
#     return vector_stores[subject]

# def save_vectorstore(subject: str):
#     """Save vector store to disk"""
#     vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
#     vector_stores[subject].save_local(vector_store_path)

# def load_chat_history(subject: str) -> List[Dict]:
#     """Load chat history for a subject"""
#     history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
#     if os.path.exists(history_file):
#         with open(history_file, 'r') as f:
#             return json.load(f)
#     return []

# def save_chat_history(subject: str, history: List[Dict]):
#     """Save chat history for a subject"""
#     history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
#     with open(history_file, 'w') as f:
#         json.dump(history, f)

# def download_pdf(url: str) -> bytes:
#     """Download PDF from URL"""
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         return response.content
#     except requests.RequestException as e:
#         st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
#         return None

# def process_pdf(pdf_content: bytes) -> List[str]:
#     """Process PDF content and return text chunks"""
#     pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#     pdf_file.write(pdf_content)
#     pdf_file.close()

#     reader = PdfReader(pdf_file.name)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()

#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1500,
#         chunk_overlap=300,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)

#     os.unlink(pdf_file.name)
#     return chunks

# def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
#     """Add document to vector store"""
#     chunks = process_pdf(pdf_content)
#     embeddings = get_embeddings()
    
#     if subject not in vector_stores:
#         vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     else:
#         vector_stores[subject].add_texts(chunks)
    
#     save_vectorstore(subject)
#     st.success(f"Successfully added {source} to the {subject} vector store.")

# def get_relevant_passages(query: str, subject: str, k: int = 5) -> List[str]:
#     """Get relevant passages from vector store"""
#     vectorstore = get_or_create_vectorstore(subject)
#     results = vectorstore.similarity_search(query, k=k)
#     return [doc.page_content for doc in results]

# @st.cache_data
# def generate_answer(prompt: str) -> str:
#     """Generate answer using Gemini Pro"""
#     try:
#         model = genai.GenerativeModel('gemini-pro')
#         generation_config = {
#             "temperature": 0.7,
#             "top_p": 0.8,
#             "top_k": 40,
#             "max_output_tokens": 2048,
#         }
#         result = model.generate_content(prompt, generation_config=generation_config)
#         return result.text
#     except Exception as e:
#         st.error(f"Error generating answer: {str(e)}")
#         return None

# # def make_rag_prompt(query: str, relevant_passages: List[str], subject: str, chat_history: List[Dict]) -> str:
# #     """Construct RAG prompt"""
# #     passages_text = "\n".join(f"PASSAGE {i+1}: {p}" for i, p in enumerate(relevant_passages))
# #     history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in chat_history[-5:]])
    
# #     return f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU students.
# # Use the provided passages and your knowledge to give comprehensive answers.
# # If the passages don't contain relevant information, use your general knowledge.

# # Chat History:
# # {history_text}

# # Reference Passages:
# # {passages_text}

# # QUESTION: '{query}'

# # ANSWER:"""

# def make_rag_prompt(query: str, relevant_passages: List[str], subject: str, chat_history: List[Dict]):
#     escaped_passages = [p.replace("'", "").replace('"', "").replace("\n", " ") for p in relevant_passages]
#     passages_text = "\n".join(f"PASSAGE {i+1}: {p}" for i, p in enumerate(escaped_passages))
    
#     history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in chat_history[-5:]])
    
#     prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students.
# Your role is to:
# 1. First check if the provided reference passages contain relevant information for the question.
# 2. If they do, use that information as your primary source and combine it with your knowledge to provide a comprehensive answer.
# 3. If they don't contain relevant information, use your own knowledge to provide a detailed answer instead of saying you cannot answer.
# 4. When using information from Include all relevant information and specify the page numbers, line numbers, and PDF names where the information is found. If the answer requires additional knowledge beyond the provided context, provide relevant information or insights using your knowledge. Do not provide incorrect information.
# 5. Always maintain an academic and informative tone.

# Remember: Maintain a formal and academic tone throughout your response which is also simple to understand and informative. Answer as per required depth and weightage to the topic in subject.
# You should ALWAYS provide a helpful answer. If the passages don't contain relevant information, use your general knowledge instead of saying you cannot answer.

# Chat History:
# {history_text}

# Reference Passages:
# {passages_text}

# QUESTION: '{query}'

# ANSWER:"""
#     return prompt


# def generate_llm_answer(query: str, subject: str = None, chat_history: List[Dict] = None) -> str:
#     """Generate answer using LLM's knowledge without RAG"""
#     history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in (chat_history or [])[-5:]])
    
#     if subject:
#         prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students. 
# You have in-depth knowledge about GTU's curriculum and courses related to {subject}.
# Please provide a comprehensive and informative answer to the following question, using your specialized knowledge and considering the chat history:

# Chat History:
# {history_text}

# QUESTION: {query}

# ANSWER:"""
#     else:
#         prompt = f"""You are GTUtor, a helpful and informative AI assistant for GTU (Gujarat Technological University) students. 
# You have general knowledge about GTU's curriculum and various courses.
# Please provide a comprehensive and informative answer to the following question, using your knowledge and considering the chat history:

# Chat History:
# {history_text}

# QUESTION: {query}

# ANSWER:"""
#     return generate_answer(prompt)

# def delete_message(subject: str, index: int):
#     """Delete a specific message from chat history"""
#     if subject in st.session_state.chat_histories:
#         del st.session_state.chat_histories[subject][index]
#         save_chat_history(subject, st.session_state.chat_histories[subject])
#         st.rerun()

# # Initialize session state
# if 'chat_histories' not in st.session_state:
#     st.session_state.chat_histories = {}

# # Load existing subjects
# subjects = load_subjects()

# # Custom CSS
# st.markdown("""
# <style>
# .chat-message {
#     padding: 1.5rem;
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
#     display: flex
# }
# .chat-message.user {
#     background-color: #2b313e
# }
# .chat-message.bot {
#     background-color: #475063
# }
# .chat-message .avatar {
#     width: 20%;
# }
# .chat-message .avatar img {
#     max-width: 78px;
#     max-height: 78px;
#     border-radius: 50%;
#     object-fit: cover;
# }
# .chat-message .message {
#     width: 80%;
#     padding: 0 1.5rem;
#     color: #fff;
# }
# .stTextArea textarea {
#     font-size: 16px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # Main UI
# st.title("GTUtor: Dynamic Multi-Subject Chat System")

# # Subject selection
# subject_option = st.selectbox("Select a subject or create a new one", [""] + subjects + ["Create New Subject"])

# if subject_option == "Create New Subject":
#     new_subject = st.text_input("Enter the name of the new subject")
#     if new_subject and new_subject not in subjects:
#         subjects.append(new_subject)
#         save_subjects(subjects)
#         st.success(f"New subject '{new_subject}' created successfully!")
#         subject_option = new_subject

# selected_subject = subject_option if subject_option != "Create New Subject" else new_subject

# # Load chat history for selected subject
# if selected_subject and selected_subject not in st.session_state.chat_histories:
#     st.session_state.chat_histories[selected_subject] = load_chat_history(selected_subject)

# # File upload and processing
# if selected_subject:
#     st.subheader(f"Add Documents to {selected_subject}")
#     uploaded_file = st.file_uploader(f"Choose a PDF file for {selected_subject}", type="pdf")
#     pdf_url = st.text_input(f"Or enter a PDF URL for {selected_subject}")

#     if uploaded_file:
#         pdf_content = uploaded_file.read()
#         with st.spinner("Processing PDF..."):
#             add_document_to_vectorstore(pdf_content, uploaded_file.name, selected_subject)

#     elif pdf_url:
#         with st.spinner("Downloading PDF..."):
#             pdf_content = download_pdf(pdf_url)
#             if pdf_content:
#                 with st.spinner("Processing PDF..."):
#                     add_document_to_vectorstore(pdf_content, pdf_url, selected_subject)

# # Display chat history
# if selected_subject and selected_subject in st.session_state.chat_histories:
#     for i, turn in enumerate(st.session_state.chat_histories[selected_subject]):
#         # User message
#         st.markdown(f'''
#             <div class="chat-message user">
#                 <div class="avatar">
#                     <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/>
#                 </div>
#                 <div class="message">{turn["human"]}</div>
#             </div>
#         ''', unsafe_allow_html=True)
        
#         # Delete button for message
#         cols = st.columns([0.85, 0.15])
#         cols[1].button("🗑️", key=f"delete_msg_{i}", on_click=lambda idx=i: delete_message(selected_subject, idx))
        
#         # Bot message
#         bot_message_html = markdown.markdown(turn["ai"])
#         st.markdown(f'''
#             <div class="chat-message bot">
#                 <div class="avatar">
#                     <img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/>
#                 </div>
#                 <div class="message">{bot_message_html}</div>
#             </div>
#         ''', unsafe_allow_html=True)
        
#         # Copy buttons
#         cols = st.columns(2)
#         cols[0].button("Copy Question", key=f"copy_q_{i}", on_click=lambda q=turn["human"]: clipboard.copy(q))
#         cols[1].button("Copy Answer", key=f"copy_a_{i}", on_click=lambda a=turn["ai"]: clipboard.copy(a))

# # Query input and processing
# query = st.text_input("Enter your question")

# if query:
#     with st.spinner("Generating answer..."):
#         if selected_subject:
#             try:
#                 # Try RAG-based answer first
#                 relevant_texts = get_relevant_passages(query, selected_subject)
#                 chat_history = st.session_state.chat_histories.get(selected_subject, [])
                
#                 rag_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
#                 answer = generate_answer(rag_prompt)
                
#                 # Fallback to general knowledge if RAG fails
#                 if not answer or "unable to answer" in answer.lower() or "do not contain" in answer.lower():
#                     answer = generate_llm_answer(query, selected_subject, chat_history)
#             except Exception as e:
#                 st.error(f"Error with RAG response: {str(e)}")
#                 answer = generate_llm_answer(query, selected_subject, chat_history)
#         else:
#             # Generate general knowledge answer when no subject is selected
#             answer = generate_llm_answer(query)
        
#         if answer:
#         # Display the new conversation
#             st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{query}</div></div>', unsafe_allow_html=True)
#             answer_html = markdown.markdown(answer, extensions=['tables'])
#             answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
#             st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
#             # Copy buttons for the new conversation
#             cols = st.columns(2)
#             cols[0].button("Copy Question", key="copy_current_q", on_click=lambda: clipboard.copy(query))
#             cols[1].button("Copy Answer", key="copy_current_a", on_click=lambda: clipboard.copy(answer))
            
#             # Update chat history
#             if selected_subject:
#                 st.session_state.chat_histories.setdefault(selected_subject, []).append({
#                     'human': query,
#                     'ai': answer
#                 })
#                 save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])

# st.sidebar.title("🎓 GTUtor: Your AI Study Companion")
# st.sidebar.markdown("""
# ### Welcome to GTUtor! 🌟

# GTUtor is an advanced AI-powered tutoring system specifically designed for Gujarat Technological University (GTU) students. It combines Google's cutting-edge Gemini Pro AI with a sophisticated document-based knowledge system to provide:

# - 📚 **Multi-Subject Learning**: Create and manage separate subjects with dedicated knowledge bases
# - 🔍 **Smart Document Integration**: Upload PDFs or add via URLs to enhance subject understanding
# - 💡 **Intelligent Responses**: Context-aware answers combining document knowledge with AI capabilities
# - 💬 **Interactive Chat**: Dynamic conversation system with history tracking
# - 🎯 **GTU-Focused Content**: Tailored specifically for GTU curriculum and courses
# - 📋 **Easy Sharing**: Copy and paste functionality for questions and answers

# ### How to Use
# 1. Select or create a subject
# 2. Upload relevant PDF documents
# 3. Start asking questions!

# ### Getting Started
# Upload your first PDF document or select an existing subject to begin your learning journey.

# Made with ❤️ for GTU students
# """)


# # Sidebar information and buttons
# st.sidebar.title("GTUtor Controls")

# if selected_subject:
#     db = get_or_create_vectorstore(selected_subject)
#     total_docs = len(db.index_to_docstore_id)
#     st.sidebar.write(f"Total documents in {selected_subject} database: {total_docs}")

#     # Clear database button
#     if st.sidebar.button(f"Clear {selected_subject} Database"):
#         # db.delete(delete_all=True)
#         # Reinitialize the vector store with an empty text
#         embeddings = get_embeddings()
#         vector_stores[selected_subject] = FAISS.from_texts(texts=["Initial text"], embedding=embeddings)
#         save_vectorstore(selected_subject)
#         # Clear chat history
#         st.session_state.chat_histories[selected_subject] = []
#         save_chat_history(selected_subject, [])
#         st.sidebar.success(f"{selected_subject} database and chat history cleared successfully.")
#         st.rerun()
    
#     # Delete subject button
#     if st.sidebar.button(f"Delete {selected_subject} Subject"):
#         # Remove from subjects list
#         subjects.remove(selected_subject)
#         save_subjects(subjects)
        
#         # Delete database
#         db_path = os.path.join(vector_stores_folder, selected_subject.lower().replace(" ", "_"))
#         if os.path.exists(db_path):
#             import shutil
#             shutil.rmtree(db_path)
        
#         # Delete chat history
#         if selected_subject in st.session_state.chat_histories:
#             del st.session_state.chat_histories[selected_subject]
#         history_file = os.path.join(history_folder, f"{selected_subject.lower().replace(' ', '_')}_history.json")
#         if os.path.exists(history_file):
#             os.remove(history_file)
        
#         st.sidebar.success(f"{selected_subject} subject deleted successfully.")
#         st.rerun()

# # Option to start a new conversation
# if st.sidebar.button("Start New Conversation"):
#     if selected_subject:
#         st.session_state.chat_histories[selected_subject] = []
#         save_chat_history(selected_subject, [])
#         st.success("New conversation started.")
#         st.rerun()
#     else:
#         st.warning("Please select a subject before starting a new conversation.")

# # Function to delete a specific message
# def delete_message(subject, index):
#     if subject in st.session_state.chat_histories:
#         del st.session_state.chat_histories[subject][index]
#         save_chat_history(subject, st.session_state.chat_histories[subject])
#         st.rerun()

# # Add custom CSS to improve readability
# st.markdown("""
# <style>
# .stTextArea textarea {
#     font-size: 16px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # import streamlit as st
# # import requests
# # import os
# # from typing import List, Dict
# # from dotenv import load_dotenv
# # import tempfile
# # import json
# # import uuid
# # import clipboard
# # import markdown
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain_community.vectorstores import FAISS
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # import google.generativeai as genai
# # from google_auth_oauthlib.flow import Flow
# # from google.oauth2.credentials import Credentials
# # from googleapiclient.discovery import build
# # import pickle
# # from pathlib import Path

# # # Load environment variables
# # load_dotenv()

# # # OAuth 2.0 Configuration
# # GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
# # GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# # # Define multiple redirect URIs for different environments
# # # REDIRECT_URIS = [
# # #     "http://localhost:8501/",  # Local development
# # #     "http://localhost:8501/callback",  # Local with callback path
# # #     "https://gtutor.streamlit.app/",  # Streamlit Cloud URL
# # #     "https://gtutor.streamlit.app/_stcore/callback",  # Streamlit Cloud callback
# # # ]

# # # OAuth2 Scopes
# # SCOPES = [
# #     'https://www.googleapis.com/auth/userinfo.email',
# #     'https://www.googleapis.com/auth/userinfo.profile',
# # ]

# # # Initialize session state for authentication
# # if 'user' not in st.session_state:
# #     st.session_state.user = None

# # # First, modify the REDIRECT_URIS to ensure they exactly match what's configured in Google Cloud Console
# # REDIRECT_URIS = [
# #     "http://localhost:8501/",  # Local development
# #     "https://gtutor.streamlit.app/",  # Production URL
# #     "https://gtutor.streamlit.app/_stcore/callback"  # Production callback URL
# # ]

# # def is_production():
# #     """Check if the app is running in production using URL-based detection"""
# #     try:
# #         # Get the current URL from Streamlit's internal state
# #         current_url = st.runtime.get_instance().get_current_page_url()
# #         return "gtutor.streamlit.app" in current_url
# #     except:
# #         # Fallback detection methods
# #         return (
# #             'STREAMLIT_SHARING_PORT' in os.environ or  # Check for Streamlit Cloud
# #             'STREAMLIT_SERVER_PORT' in os.environ or   # Another production indicator
# #             'STREAMLIT_SERVER_HEADLESS' in os.environ  # Common in cloud deployments
# #         )

# # def get_redirect_uri():
# #     """Get the appropriate redirect URI based on the current environment"""
# #     if is_production():
# #         return "https://gtutor.streamlit.app/_stcore/callback"
# #     return "http://localhost:8501/"

# # def get_base_url():
# #     """Get the base URL for the current environment"""
# #     if is_production():
# #         return "https://gtutor.streamlit.app"
# #     return "http://localhost:8501"

# # def handle_oauth_callback():
# #     """Handle OAuth callback and user authentication"""
# #     # Add comprehensive debug information
# #     debug_info = {
# #         "Query Params": dict(st.query_params),
# #         "Is Production": is_production(),
# #         "Redirect URI": get_redirect_uri(),
# #         "Base URL": get_base_url(),
# #         "Environment Vars": {k: v for k, v in os.environ.items() if k.startswith('STREAMLIT')},
# #         "Current URL": st.runtime.get_instance().get_current_page_url() if hasattr(st.runtime, 'get_instance') else None
# #     }
    
# #     with st.sidebar.expander("Debug Info"):
# #         st.write(debug_info)
    
# #     # Get query parameters
# #     query_params = st.query_params
    
# #     if 'code' in query_params:
# #         try:
# #             flow = create_oauth_flow()
            
# #             # Build the full authorization response URL
# #             base_url = get_base_url()
# #             query_string = '&'.join([f"{k}={v}" for k, v in query_params.items()])
# #             authorization_response = f"{base_url}/?{query_string}"
            
# #             # Log the authorization URL for debugging
# #             st.sidebar.write("Authorization Response URL:", authorization_response)
            
# #             # Exchange the authorization code for tokens
# #             flow.fetch_token(
# #                 authorization_response=authorization_response
# #             )
            
# #             # Get credentials and user info
# #             credentials = flow.credentials
# #             user_info = get_user_info(credentials)
            
# #             if user_info:
# #                 st.session_state.user = user_info
# #                 st.session_state.credentials = credentials_to_dict(credentials)
                
# #                 # Clear URL parameters and refresh
# #                 st.query_params.clear()
# #                 st.rerun()
# #             else:
# #                 st.error("Failed to get user info after authentication")
            
# #         except Exception as e:
# #             st.error(f"Authentication failed: {str(e)}")
# #             st.error("Detailed error information for debugging:", str(e.__class__.__name__))

# # def create_oauth_flow():
# #     """Create OAuth 2.0 flow instance with environment-aware configuration"""
# #     if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
# #         raise ValueError("Missing Google OAuth credentials. Please check your environment variables.")
    
# #     client_config = {
# #         "web": {
# #             "client_id": GOOGLE_CLIENT_ID,
# #             "client_secret": GOOGLE_CLIENT_SECRET,
# #             "auth_uri": "https://accounts.google.com/o/oauth2/auth",
# #             "token_uri": "https://oauth2.googleapis.com/token",
# #             "redirect_uris": [
# #                 "http://localhost:8501/",
# #                 "https://gtutor.streamlit.app/",
# #                 "https://gtutor.streamlit.app/_stcore/callback"
# #             ],
# #             "javascript_origins": [
# #                 "http://localhost:8501",
# #                 "https://gtutor.streamlit.app"
# #             ]
# #         }
# #     }
    
# #     # Create flow with the appropriate redirect URI
# #     flow = Flow.from_client_config(
# #         client_config,
# #         scopes=SCOPES,
# #         redirect_uri=get_redirect_uri()
# #     )
# #     return flow

# # def login():
# #     """Initiate Google OAuth login flow with environment awareness"""
# #     try:
# #         flow = create_oauth_flow()
# #         authorization_url, state = flow.authorization_url(
# #             access_type='offline',
# #             include_granted_scopes='true',
# #             prompt='consent'
# #         )
        
# #         # Store state in session
# #         st.session_state.oauth_state = state
        
# #         # Create the sign-in button with dynamic URL
# #         st.markdown('''
# #             <style>
# #             .google-btn {
# #                 background-color: #4285F4;
# #                 color: white;
# #                 padding: 10px 20px;
# #                 border: none;
# #                 border-radius: 5px;
# #                 cursor: pointer;
# #                 font-family: 'Google Sans', sans-serif;
# #                 font-size: 14px;
# #                 display: inline-flex;
# #                 align-items: center;
# #                 box-shadow: 0 2px 4px 0 rgba(0,0,0,.25);
# #             }
# #             .google-btn:hover {
# #                 background-color: #357ABD;
# #                 box-shadow: 0 0 3px 3px rgba(66,133,244,.3);
# #             }
# #             .google-btn img {
# #                 margin-right: 10px;
# #                 height: 18px;
# #             }
# #             </style>
# #         ''', unsafe_allow_html=True)
        
# #         # Add debug information to the authorization URL
# #         debug_params = {
# #             'redirect_uri': get_redirect_uri(),
# #             'is_production': str(is_production()).lower(),
# #             'base_url': get_base_url()
# #         }
# #         authorization_url += '&' + '&'.join([f"debug_{k}={v}" for k, v in debug_params.items()])
        
# #         st.markdown(f'''
# #             <a href="{authorization_url}" target="_self">
# #                 <button class="google-btn">
# #                     <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg"/>
# #                     Sign in with Google
# #                 </button>
# #             </a>
# #         ''', unsafe_allow_html=True)
        
# #         # Display current environment info
# #         st.sidebar.write("Current Environment:", "Production" if is_production() else "Local")
# #         st.sidebar.write("Redirect URI:", get_redirect_uri())
        
# #     except Exception as e:
# #         st.error(f"Failed to create authorization URL: {str(e)}")
# #         st.error("Detailed error information:", str(e.__class__.__name__))

# # def get_user_info(credentials):
# #     """Get user information from Google"""
# #     try:
# #         service = build('oauth2', 'v2', credentials=credentials)
# #         user_info = service.userinfo().get().execute()
# #         return user_info
# #     except Exception as e:
# #         st.error(f"Failed to get user info: {str(e)}")
# #         return None

# # def credentials_to_dict(credentials):
# #     """Convert credentials to dictionary for storage"""
# #     return {
# #         'token': credentials.token,
# #         'refresh_token': credentials.refresh_token,
# #         'token_uri': credentials.token_uri,
# #         'client_id': credentials.client_id,
# #         'client_secret': credentials.client_secret,
# #         'scopes': credentials.scopes
# #     }

# # def auth_required(func):
# #     """Decorator to require authentication for protected routes"""
# #     def wrapper(*args, **kwargs):
# #         if st.session_state.user is None:
# #             st.warning("Please sign in to access this feature.")
# #             login()
# #             return None
# #         return func(*args, **kwargs)
# #     return wrapper

# # # Modify your main Streamlit UI to include authentication
# # def main():
# #     st.set_page_config(page_title="GTUtor", page_icon="🎓", layout="wide")
    
# #     # Handle OAuth callback
# #     handle_oauth_callback()
    
# #     if not st.session_state.get('user'):
# #         st.write("Please sign in to use GTUtor")
# #         login()
# #         return
    
# #     # Authentication status
# #     if st.session_state.user:
# #         with st.sidebar:
# #             col1, col2 = st.columns([1, 3])
# #             with col1:
# #                 st.image(st.session_state.user['picture'], width=50)
# #             with col2:
# #                 st.write(f"Welcome, {st.session_state.user['name']}!")
# #                 st.write(f"Email: {st.session_state.user['email']}")
            
# #             if st.button("Sign Out", type="primary"):
# #                 st.session_state.user = None
# #                 st.session_state.credentials = None
# #                 st.rerun()
# #     else:
# #         st.sidebar.write("Please sign in to use GTUtor")
# #         login()
# #         return

# #     # Your existing GTUtor code here, wrapped with @auth_required decorator
# #     if st.session_state.user:
# #         show_gtutor_interface()

# # # Load environment variables
# # # load_dotenv()

# # # Set up API key
# # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# # if not GOOGLE_API_KEY:
# #     raise ValueError("GOOGLE_API_KEY not found in .env file")

# # # Configure genai with the API key
# # genai.configure(api_key=GOOGLE_API_KEY)
# # os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# # # Create base data folder structure if it doesn't exist
# # data_folders = {
# #     "gtutor_data": [
# #         "vector_stores",
# #         "chat_histories"
# #     ]
# # }

# # for parent_folder, sub_folders in data_folders.items():
# #     parent_path = os.path.join(os.getcwd(), parent_folder)
# #     if not os.path.exists(parent_path):
# #         os.makedirs(parent_path)
# #         print(f"Created {parent_folder} directory")
    
# #     for sub_folder in sub_folders:
# #         sub_path = os.path.join(parent_path, sub_folder)
# #         if not os.path.exists(sub_path):
# #             os.makedirs(sub_path)
# #             print(f"Created {sub_folder} subdirectory")

# # # Create directories for storing data
# # data_folder = os.path.join(os.getcwd(), "gtutor_data")
# # vector_stores_folder = os.path.join(data_folder, "vector_stores")
# # history_folder = os.path.join(data_folder, "chat_histories")
# # os.makedirs(vector_stores_folder, exist_ok=True)
# # os.makedirs(history_folder, exist_ok=True)

# # # File to store subject names
# # subjects_file = os.path.join(data_folder, "subjects.json")

# # def load_subjects():
# #     if os.path.exists(subjects_file):
# #         with open(subjects_file, 'r') as f:
# #             return json.load(f)
# #     return []

# # def save_subjects(subjects):
# #     with open(subjects_file, 'w') as f:
# #         json.dump(subjects, f)

# # # Initialize vector stores dictionary
# # vector_stores = {}

# # def get_embeddings():
# #     """Create a new embeddings object"""
# #     return GoogleGenerativeAIEmbeddings(
# #         model="models/embedding-001",
# #         google_api_key=GOOGLE_API_KEY
# #     )

# # def get_or_create_vectorstore(subject: str):
# #     """Get or create a FAISS vector store for a subject"""
# #     if subject not in vector_stores:
# #         vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
# #         if os.path.exists(vector_store_path):
# #             embeddings = get_embeddings()
# #             vector_stores[subject] = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
# #         else:
# #             embeddings = get_embeddings()
# #             vector_stores[subject] = FAISS.from_texts(texts=["Initial text"], embedding=embeddings)
# #             save_vectorstore(subject)
# #     return vector_stores[subject]

# # def save_vectorstore(subject: str):
# #     """Save vector store to disk"""
# #     vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
# #     vector_stores[subject].save_local(vector_store_path)

# # def load_chat_history(subject: str) -> List[Dict]:
# #     """Load chat history for a subject"""
# #     history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
# #     if os.path.exists(history_file):
# #         with open(history_file, 'r') as f:
# #             return json.load(f)
# #     return []

# # def save_chat_history(subject: str, history: List[Dict]):
# #     """Save chat history for a subject"""
# #     history_file = os.path.join(history_folder, f"{subject.lower().replace(' ', '_')}_history.json")
# #     with open(history_file, 'w') as f:
# #         json.dump(history, f)

# # def download_pdf(url: str) -> bytes:
# #     """Download PDF from URL"""
# #     try:
# #         response = requests.get(url, timeout=10)
# #         response.raise_for_status()
# #         return response.content
# #     except requests.RequestException as e:
# #         st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
# #         return None

# # def process_pdf(pdf_content: bytes) -> List[str]:
# #     """Process PDF content and return text chunks"""
# #     pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
# #     pdf_file.write(pdf_content)
# #     pdf_file.close()

# #     reader = PdfReader(pdf_file.name)
# #     text = ""
# #     for page in reader.pages:
# #         text += page.extract_text()

# #     text_splitter = CharacterTextSplitter(
# #         separator="\n",
# #         chunk_size=1500,
# #         chunk_overlap=300,
# #         length_function=len
# #     )
# #     chunks = text_splitter.split_text(text)

# #     os.unlink(pdf_file.name)
# #     return chunks

# # def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
# #     """Add document to vector store"""
# #     chunks = process_pdf(pdf_content)
# #     embeddings = get_embeddings()
    
# #     if subject not in vector_stores:
# #         vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
# #     else:
# #         vector_stores[subject].add_texts(chunks)
    
# #     save_vectorstore(subject)
# #     st.success(f"Successfully added {source} to the {subject} vector store.")

# # def get_relevant_passages(query: str, subject: str, k: int = 5) -> List[str]:
# #     """Get relevant passages from vector store"""
# #     vectorstore = get_or_create_vectorstore(subject)
# #     results = vectorstore.similarity_search(query, k=k)
# #     return [doc.page_content for doc in results]

# # @st.cache_data
# # def generate_answer(prompt: str) -> str:
# #     """Generate answer using Gemini Pro"""
# #     try:
# #         model = genai.GenerativeModel('gemini-pro')
# #         generation_config = {
# #             "temperature": 0.7,
# #             "top_p": 0.8,
# #             "top_k": 40,
# #             "max_output_tokens": 2048,
# #         }
# #         result = model.generate_content(prompt, generation_config=generation_config)
# #         return result.text
# #     except Exception as e:
# #         st.error(f"Error generating answer: {str(e)}")
# #         return None

# # def make_rag_prompt(query: str, relevant_passages: List[str], subject: str, chat_history: List[Dict]):
# #     escaped_passages = [p.replace("'", "").replace('"', "").replace("\n", " ") for p in relevant_passages]
# #     passages_text = "\n".join(f"PASSAGE {i+1}: {p}" for i, p in enumerate(escaped_passages))
    
# #     history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in chat_history[-5:]])
    
# #     prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students.
# # Your role is to:
# # 1. First check if the provided reference passages contain relevant information for the question.
# # 2. If they do, use that information as your primary source and combine it with your knowledge to provide a comprehensive answer.
# # 3. If they don't contain relevant information, use your own knowledge to provide a detailed answer instead of saying you cannot answer.
# # 4. When using information from Include all relevant information and specify the page numbers, line numbers, and PDF names where the information is found. If the answer requires additional knowledge beyond the provided context, provide relevant information or insights using your knowledge. Do not provide incorrect information.
# # 5. Always maintain an academic and informative tone.

# # Remember: Maintain a formal and academic tone throughout your response which is also simple to understand and informative. Answer as per required depth and weightage to the topic in subject.
# # You should ALWAYS provide a helpful answer. If the passages don't contain relevant information, use your general knowledge instead of saying you cannot answer.

# # Chat History:
# # {history_text}

# # Reference Passages:
# # {passages_text}

# # QUESTION: '{query}'

# # ANSWER:"""
# #     return prompt


# # def generate_llm_answer(query: str, subject: str = None, chat_history: List[Dict] = None) -> str:
# #     """Generate answer using LLM's knowledge without RAG"""
# #     history_text = "\n".join([f"Human: {turn['human']}\nAssistant: {turn['ai']}" for turn in (chat_history or [])[-5:]])
    
# #     if subject:
# #         prompt = f"""You are GTUtor, a helpful and informative AI assistant specializing in {subject} for GTU (Gujarat Technological University) students. 
# # You have in-depth knowledge about GTU's curriculum and courses related to {subject}.
# # Please provide a comprehensive and informative answer to the following question, using your specialized knowledge and considering the chat history:

# # Chat History:
# # {history_text}

# # QUESTION: {query}

# # ANSWER:"""
# #     else:
# #         prompt = f"""You are GTUtor, a helpful and informative AI assistant for GTU (Gujarat Technological University) students. 
# # You have general knowledge about GTU's curriculum and various courses.
# # Please provide a comprehensive and informative answer to the following question, using your knowledge and considering the chat history:

# # Chat History:
# # {history_text}

# # QUESTION: {query}

# # ANSWER:"""
# #     return generate_answer(prompt)

# # # Function to delete a specific message
# # def delete_message(subject, index):
# #     if subject in st.session_state.chat_histories:
# #         del st.session_state.chat_histories[subject][index]
# #         save_chat_history(subject, st.session_state.chat_histories[subject])
# #         st.rerun()

# # @auth_required
# # def show_gtutor_interface():
# #     """Main GTUtor interface - only shown to authenticated users"""
# #     # Your existing GTUtor code goes here
# #     # (Copy your existing main interface code here)
# #     if 'chat_histories' not in st.session_state:
# #         st.session_state.chat_histories = {}

# #     # Load existing subjects
# #     subjects = load_subjects()

# #     # Custom CSS
# #     st.markdown("""
# #     <style>
# #     .chat-message {
# #         padding: 1.5rem;
# #         border-radius: 0.5rem;
# #         margin-bottom: 1rem;
# #         display: flex
# #     }
# #     .chat-message.user {
# #         background-color: #2b313e
# #     }
# #     .chat-message.bot {
# #         background-color: #475063
# #     }
# #     .chat-message .avatar {
# #         width: 20%;
# #     }
# #     .chat-message .avatar img {
# #         max-width: 78px;
# #         max-height: 78px;
# #         border-radius: 50%;
# #         object-fit: cover;
# #     }
# #     .chat-message .message {
# #         width: 80%;
# #         padding: 0 1.5rem;
# #         color: #fff;
# #     }
# #     .stTextArea textarea {
# #         font-size: 16px !important;
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)

# #     # Main UI
# #     st.title("GTUtor: Dynamic Multi-Subject Chat System")

# #     # Subject selection
# #     subject_option = st.selectbox("Select a subject or create a new one", [""] + subjects + ["Create New Subject"])

# #     if subject_option == "Create New Subject":
# #         new_subject = st.text_input("Enter the name of the new subject")
# #         if new_subject and new_subject not in subjects:
# #             subjects.append(new_subject)
# #             save_subjects(subjects)
# #             st.success(f"New subject '{new_subject}' created successfully!")
# #             subject_option = new_subject

# #     selected_subject = subject_option if subject_option != "Create New Subject" else new_subject

# #     # Load chat history for selected subject
# #     if selected_subject and selected_subject not in st.session_state.chat_histories:
# #         st.session_state.chat_histories[selected_subject] = load_chat_history(selected_subject)

# #     # File upload and processing
# #     if selected_subject:
# #         st.subheader(f"Add Documents to {selected_subject}")
# #         uploaded_file = st.file_uploader(f"Choose a PDF file for {selected_subject}", type="pdf")
# #         pdf_url = st.text_input(f"Or enter a PDF URL for {selected_subject}")

# #         if uploaded_file:
# #             pdf_content = uploaded_file.read()
# #             with st.spinner("Processing PDF..."):
# #                 add_document_to_vectorstore(pdf_content, uploaded_file.name, selected_subject)

# #         elif pdf_url:
# #             with st.spinner("Downloading PDF..."):
# #                 pdf_content = download_pdf(pdf_url)
# #                 if pdf_content:
# #                     with st.spinner("Processing PDF..."):
# #                         add_document_to_vectorstore(pdf_content, pdf_url, selected_subject)

# #     # Display chat history
# #     if selected_subject and selected_subject in st.session_state.chat_histories:
# #         for i, turn in enumerate(st.session_state.chat_histories[selected_subject]):
# #             # User message
# #             st.markdown(f'''
# #                 <div class="chat-message user">
# #                     <div class="avatar">
# #                         <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/>
# #                     </div>
# #                     <div class="message">{turn["human"]}</div>
# #                 </div>
# #             ''', unsafe_allow_html=True)
            
# #             # Delete button for message
# #             cols = st.columns([0.85, 0.15])
# #             cols[1].button("🗑️", key=f"delete_msg_{i}", on_click=lambda idx=i: delete_message(selected_subject, idx))
            
# #             # Bot message
# #             bot_message_html = markdown.markdown(turn["ai"])
# #             st.markdown(f'''
# #                 <div class="chat-message bot">
# #                     <div class="avatar">
# #                         <img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/>
# #                     </div>
# #                     <div class="message">{bot_message_html}</div>
# #                 </div>
# #             ''', unsafe_allow_html=True)
            
# #             # Copy buttons
# #             cols = st.columns(2)
# #             cols[0].button("Copy Question", key=f"copy_q_{i}", on_click=lambda q=turn["human"]: clipboard.copy(q))
# #             cols[1].button("Copy Answer", key=f"copy_a_{i}", on_click=lambda a=turn["ai"]: clipboard.copy(a))

# #     # Query input and processing
# #     query = st.text_input("Enter your question")

# #     if query:
# #         with st.spinner("Generating answer..."):
# #             if selected_subject:
# #                 try:
# #                     # Try RAG-based answer first
# #                     relevant_texts = get_relevant_passages(query, selected_subject)
# #                     chat_history = st.session_state.chat_histories.get(selected_subject, [])
                    
# #                     rag_prompt = make_rag_prompt(query, relevant_texts, selected_subject, chat_history)
# #                     answer = generate_answer(rag_prompt)
                    
# #                     # Fallback to general knowledge if RAG fails
# #                     if not answer or "unable to answer" in answer.lower() or "do not contain" in answer.lower():
# #                         answer = generate_llm_answer(query, selected_subject, chat_history)
# #                 except Exception as e:
# #                     st.error(f"Error with RAG response: {str(e)}")
# #                     answer = generate_llm_answer(query, selected_subject, chat_history)
# #             else:
# #                 # Generate general knowledge answer when no subject is selected
# #                 answer = generate_llm_answer(query)
            
# #             if answer:
# #             # Display the new conversation
# #                 st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{query}</div></div>', unsafe_allow_html=True)
# #                 answer_html = markdown.markdown(answer, extensions=['tables'])
# #                 answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
# #                 st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
# #                 # Copy buttons for the new conversation
# #                 cols = st.columns(2)
# #                 cols[0].button("Copy Question", key="copy_current_q", on_click=lambda: clipboard.copy(query))
# #                 cols[1].button("Copy Answer", key="copy_current_a", on_click=lambda: clipboard.copy(answer))
                
# #                 # Update chat history
# #                 if selected_subject:
# #                     st.session_state.chat_histories.setdefault(selected_subject, []).append({
# #                         'human': query,
# #                         'ai': answer
# #                     })
# #                     save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])

# #     st.sidebar.title("🎓 GTUtor: Your AI Study Companion")
# #     st.sidebar.markdown("""
# #     ### Welcome to GTUtor! 🌟

# #     GTUtor is an advanced AI-powered tutoring system specifically designed for Gujarat Technological University (GTU) students. It combines Google's cutting-edge Gemini Pro AI with a sophisticated document-based knowledge system to provide:

# #     - 📚 **Multi-Subject Learning**: Create and manage separate subjects with dedicated knowledge bases
# #     - 🔍 **Smart Document Integration**: Upload PDFs or add via URLs to enhance subject understanding
# #     - 💡 **Intelligent Responses**: Context-aware answers combining document knowledge with AI capabilities
# #     - 💬 **Interactive Chat**: Dynamic conversation system with history tracking
# #     - 🎯 **GTU-Focused Content**: Tailored specifically for GTU curriculum and courses
# #     - 📋 **Easy Sharing**: Copy and paste functionality for questions and answers

# #     ### How to Use
# #     1. Select or create a subject
# #     2. Upload relevant PDF documents
# #     3. Start asking questions!

# #     ### Getting Started
# #     Upload your first PDF document or select an existing subject to begin your learning journey.

# #     Made with ❤️ for GTU students
# #     """)


# #     # Sidebar information and buttons
# #     st.sidebar.title("GTUtor Controls")

# #     if selected_subject:
# #         db = get_or_create_vectorstore(selected_subject)
# #         total_docs = len(db.index_to_docstore_id)
# #         st.sidebar.write(f"Total documents in {selected_subject} database: {total_docs}")

# #         # Clear database button
# #         if st.sidebar.button(f"Clear {selected_subject} Database"):
# #             db.delete(delete_all=True)
# #             st.session_state.chat_histories[selected_subject] = []
# #             save_chat_history(selected_subject, [])
# #             st.sidebar.success(f"{selected_subject} database and chat history cleared successfully.")
# #             st.rerun()

# #         # Delete subject button
# #         if st.sidebar.button(f"Delete {selected_subject} Subject"):
# #             # Remove from subjects list
# #             subjects.remove(selected_subject)
# #             save_subjects(subjects)
            
# #             # Delete database
# #             db_path = os.path.join(vector_stores_folder, selected_subject.lower().replace(" ", "_"))
# #             if os.path.exists(db_path):
# #                 import shutil
# #                 shutil.rmtree(db_path)
            
# #             # Delete chat history
# #             if selected_subject in st.session_state.chat_histories:
# #                 del st.session_state.chat_histories[selected_subject]
# #             history_file = os.path.join(history_folder, f"{selected_subject.lower().replace(' ', '_')}_history.json")
# #             if os.path.exists(history_file):
# #                 os.remove(history_file)
            
# #             st.sidebar.success(f"{selected_subject} subject deleted successfully.")
# #             st.rerun()

# #     # Option to start a new conversation
# #     if st.sidebar.button("Start New Conversation"):
# #         if selected_subject:
# #             st.session_state.chat_histories[selected_subject] = []
# #             save_chat_history(selected_subject, [])
# #             st.success("New conversation started.")
# #             st.rerun()
# #         else:
# #             st.warning("Please select a subject before starting a new conversation.")
# #     # Add custom CSS to improve readability
# #     st.markdown("""
# #     <style>
# #     .stTextArea textarea {
# #         font-size: 16px !important;
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)

# # if __name__ == "__main__":
# #     main()
