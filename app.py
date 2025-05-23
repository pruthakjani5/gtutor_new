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
import re
import time
import shutil
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
        try:
            os.makedirs(parent_path)
            print(f"Created {parent_folder} directory")
        except FileExistsError:
            # Directory already exists (possible race condition)
            print(f"{parent_folder} directory already exists")
    
    for sub_folder in sub_folders:
        sub_path = os.path.join(parent_path, sub_folder)
        if not os.path.exists(sub_path):
            try:
                os.makedirs(sub_path)
                print(f"Created {sub_folder} subdirectory")
            except FileExistsError:
                # Subdirectory already exists
                print(f"{sub_folder} subdirectory already exists")

# Create directories for storing data - redundant with the code above, but keeping for safety
data_folder = os.path.join(os.getcwd(), "gtutor_data")
vector_stores_folder = os.path.join(data_folder, "vector_stores")
history_folder = os.path.join(data_folder, "chat_histories")
try:
    os.makedirs(vector_stores_folder, exist_ok=True)
    os.makedirs(history_folder, exist_ok=True)
except Exception as e:
    print(f"Note: {e}")

# # Create base data folder structure if it doesn't exist
# for parent_folder, sub_folders in data_folders.items():
#     parent_path = os.path.join(os.getcwd(), parent_folder)
#     os.makedirs(parent_path, exist_ok=True)
    
#     for sub_folder in sub_folders:
#         sub_path = os.path.join(parent_path, sub_folder)
#         os.makedirs(sub_path, exist_ok=True)

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

# def download_pdf(url: str) -> bytes:
#     """Download PDF from URL"""
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         return response.content
#     except requests.RequestException as e:
#         st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
#         return None

def download_pdf(url: str) -> bytes:
    """Download PDF from URL with improved error handling for restricted sites"""
    try:
        # Add custom headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://gtu.ac.in/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Check if the response contains a PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            return response.content
        else:
            st.error(f"The URL doesn't point to a PDF file. Content-Type: {content_type}")
            return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            error_message = """
            ### Access Denied (403 Forbidden)
            
            The website is blocking direct access to this PDF. Try these alternatives:
            1. Download the PDF manually from the GTU website
            2. Then upload it using the file uploader instead
            3. Or try a different PDF link that allows direct access
            """
            st.error(error_message)
        else:
            st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
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
# def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
#     """Add document to vector store with document tracking"""
#     chunks = process_pdf(pdf_content)
#     embeddings = get_embeddings()
    
#     if subject not in vector_stores:
#         vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     else:
#         vector_stores[subject].add_texts(chunks)
    
#     # Update document count in metadata
#     save_vectorstore(subject)
    
#     # Update the session state to reflect new document count
#     if 'document_counts' not in st.session_state:
#         st.session_state.document_counts = {}
#     st.session_state.document_counts[subject] = get_document_count(subject)
    
#     st.success(f"Successfully added {source} to the {subject} vector store.")

def add_document_to_vectorstore(pdf_content: bytes, source: str, subject: str):
    """Add document to vector store with document tracking"""
    chunks = process_pdf(pdf_content)
    
    if not chunks:
        st.warning(f"No text content extracted from {source}. The PDF might be scanned or contain only images.")
        return
    
    embeddings = get_embeddings()
    
    # Get existing vector store or create a new one
    if subject in vector_stores:
        # Use add_texts to append to the existing vector store
        vector_stores[subject].add_texts(chunks)
    else:
        # Create a new vector store if none exists
        vector_store_path = os.path.join(vector_stores_folder, f"{subject.lower().replace(' ', '_')}.pkl")
        if os.path.exists(vector_store_path):
            # Load existing vector store from disk
            vector_stores[subject] = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            # Then add new texts
            vector_stores[subject].add_texts(chunks)
        else:
            # Create completely new vector store
            vector_stores[subject] = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    # Save the updated vector store
    save_vectorstore(subject)
    
    # Update the session state to reflect new document count
    if 'document_counts' not in st.session_state:
        st.session_state.document_counts = {}
    st.session_state.document_counts[subject] = get_document_count(subject)
    
    st.success(f"Successfully added {source} to the {subject} vector store. Total documents: {get_document_count(subject)}")

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
        # model = genai.GenerativeModel('gemini-1.5-pro-latest')
        model = genai.GenerativeModel('gemini-2.0-flash')
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
Always give short, concise, to-the-point answer when asked questions like "Define", "Explain", "What is", "How to", "What are", "List", "Write", "Show", "Give" etc. unless asked in "detail" or "depth".
You have in-depth knowledge about GTU's curriculum and courses related to {subject}.
You should always provide an answer. If asked for any code or program then reply only code in txt format that can be directly copied.
Please provide a comprehensive and informative answer to the following question, using the provided passages and your knowledge.
Your role is to:
1. First check if the provided reference passages contain relevant information for the question (DO NOT MAKE RANDOM PASSAGES NAME YOURSELF).
2.If they do, use that information as your primary source and combine it with your knowledge to provide a comprehensive answer.
3. If they don't contain relevant information, use your own knowledge to provide a detailed answer instead of saying you cannot answer.
4. When using information from Include all relevant information and specify the page numbers, line numbers, and PDF names where the information is found. If the answer requires additional knowledge beyond the provided context, provide relevant information or insights using your knowledge. Do not provide incorrect information.
5. Always maintain an academic and informative tone.
6. Initially always give concise, to the point and short answer unless Student asks for more in-depth detailed answers.

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
Always give short, concise, to-the-point answer when asked questions like "Define", "Explain", "What is", "How to", "What are", "List", "Write", "Show", "Give" etc. unless asked in "detail" or "depth".
Please provide a comprehensive and informative answer to the following question, using your specialized knowledge and considering the chat history:

Chat History:
{history_text}

QUESTION: {query}

ANSWER:"""
    else:
        prompt = f"""You are GTUtor, a helpful and informative AI assistant for GTU (Gujarat Technological University) students. 
You have general knowledge about GTU's curriculum and various courses.  If asked for any code or program then reply only code in txt format that can be directly copied.
Always give short, concise, to-the-point answer when asked questions like "Define", "Explain", "What is", "How to", "What are", "List", "Write", "Show", "Give" etc. unless asked in "detail" or "depth".
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

def generate_study_plan(
    subject: str,
    days_left: int,
    student_level: str = "Intermediate",
    difficulty: str = "Medium",
    target_score: int = 70,
    exam_type: str = "End-Sem",
    remaining_topics: List[str] = [],
    important_topics: List[str] = [],
    full_syllabus: List[str] = [],
    hours_per_day: int = 5
):
    """
    Generate a comprehensive study plan with detailed topics, important questions,
    study tips, and key concepts for each day.
    
    Args:
        subject: The subject name
        days_left: Number of days until exam
        student_level: Knowledge level of student
        difficulty: Subject difficulty level
        target_score: Target score percentage
        exam_type: Type of exam (Mid-Sem, End-Sem, etc.)
        remaining_topics: List of topics still to be covered
        important_topics: List of important topics based on previous exams
        full_syllabus: Complete syllabus list
        hours_per_day: Study hours available per day
        
    Returns:
        A detailed JSON-formatted study plan
    """
    # Format lists for better prompt readability
    remaining = "\n".join(f"- {topic}" for topic in remaining_topics) if remaining_topics else "Not specified"
    important = "\n".join(f"- {topic}" for topic in important_topics) if important_topics else "Not specified"
    syllabus = "\n".join(f"- {topic}" for topic in full_syllabus) if full_syllabus else "Not provided"

    # Calculate daily distribution based on topic importance and days available
    total_topics = len(remaining_topics) if remaining_topics else (len(full_syllabus) if full_syllabus else 5)
    topics_per_day = max(1, round(total_topics / max(1, days_left - 2)))  # Reserve 2 days for revision
    
    # Add vectorstore context if available for this subject
    context = ""
    if subject in vector_stores:
        try:
            relevant_passages = get_relevant_passages(f"{subject} key concepts important questions", subject, k=3)
            if relevant_passages:
                context = "Reference material from your knowledge base:\n" + "\n".join(relevant_passages)
        except:
            pass

    prompt = f"""
You are GTUtor — an expert study planner for GTU students specialized in {subject}.
You are Student's Teacher, Guide, Tutor, Counsellor, Friend and Mentor. If output json becomes too large (detailed) as per you write it smaller so that it will always end in correct order.
Make the plan manageable, easy to follow such that not only pass exams but also understand the subject {subject} in depth clearly.
Create a comprehensive, personalized {days_left}-day study plan for: **{subject}**.

🧠 Student Profile:
- Knowledge Level: {student_level}
- Target Score: {target_score}%
- Available Study Time: {hours_per_day} hours/day

📚 Exam Information:
- Type: {exam_type}
- Days Remaining: {days_left}
- Subject Difficulty: {difficulty}

📑 Full Syllabus:
{syllabus}

📌 Remaining Topics:
{remaining}

📜 Important (PYQ-based) Topics:
{important}

{context}

---

👉 Requirements:
1. Create a detailed day-by-day plan covering all topics with approximately {topics_per_day} topic(s) per day
2. Prioritize important topics in the first {min(days_left // 3, 5)} days
3. For EACH topic include:
   - Study time allocation (in hours)
   - 2-3 key concepts to master
   - 1-2 important practice questions or problem types
   - Specific study activities (reading, problem-solving, note-making, etc.)
   - Study tips tailored to this specific topic along with some examples.
4. Include strategic revision days (at least {max(1, days_left // 5)} days)
5. Add a mock test day before final revision if time is there
6. For the last day, include quick-review strategies, revision and exam tips

Format the plan as a complete detailed JSON object (IF THIS OUTPUT BECOMES TOO BIG WRITE/MAKE IT SMALLER SO THAT I CAN PARSE IT):
{{
    "Day 1": [
        {{
            "topic": "Topic Name",
            "time": "X hrs",
            "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
            "important_questions": ["Question/Problem type 1", "Question/Problem type 2"],
            "activities": ["Activity 1", "Activity 2"],
            "tips": "Topic-specific study tip"
        }},
        ...more topics for day 1...
    ],
    "Day 2": [...],
    ...and so on...
}}

Remember to balance the workload across days based on topic difficulty and importance.
    """

    try:
        # Add caching for faster repeat generations with same parameters
        return generate_study_plan_with_cache(prompt, subject, days_left, student_level, target_score)
    except Exception as e:
        st.error(f"Error generating study plan: {str(e)}")
        # Fallback to basic generation if advanced version fails
        return generate_answer(prompt)

# @st.cache_data(ttl=3600, show_spinner=False)
# def generate_study_plan_with_cache(prompt, subject, days_left, student_level, target_score):
#     """Cached version of the plan generator to improve performance for repeated requests"""
#     return generate_answer(prompt)
# Update the generate_study_plan_with_cache function to use a shorter format
@st.cache_data(ttl=3600, show_spinner=False)
def generate_study_plan_with_cache(prompt, subject, days_left, student_level, target_score):
    """Cached version of the plan generator with improved handling for large responses"""
    try:
        # Use a more compact prompt to avoid truncation
        shorter_prompt = prompt + "\n\nImportant: Keep your response concise. For longer plans (more than 10 days), limit to 1-2 key concepts and 1 important question per topic."
        
        # Generate response
        response = generate_answer(shorter_prompt)
        
        # Check if response appears to be truncated
        if "Day " + str(days_left) not in response and days_left > 5:
            # If truncated, regenerate with even more compact format
            compact_prompt = prompt + "\n\nImportant: Due to length constraints, provide a highly condensed plan with at most 1 key concept and 1 important question per topic."
            return generate_answer(compact_prompt)
        return response
    except Exception as e:
        st.error(f"Error in study plan generation: {str(e)}")
        return "Error generating study plan. Please try with fewer days or topics."

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
    # Create tabs for Chat and Study Plan
    chat_tab, study_plan_tab = st.tabs(["💬 Chat", "📝 Study Plan Generator"])
    
    with chat_tab:
        # Document upload section
        with st.expander("📚 Add Documents", expanded=False):
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
        
        # Display chat history only in the chat tab
        if selected_subject and selected_subject in st.session_state.chat_histories:
            for i, turn in enumerate(st.session_state.chat_histories[selected_subject]):
                # User message with timestamp
                display_chat_message(turn, i)
                
                # Delete button for message
                cols = st.columns([0.85, 0.15])
                cols[1].button("🗑️", key=f"delete_msg_{i}", on_click=lambda idx=i: delete_message(selected_subject, idx))
                
                # Bot message with formatting
                bot_message_html = markdown.markdown(turn["ai"], extensions=['tables'])
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
                display_copy_buttons(turn["human"], turn["ai"], i)
        
        # Query input and processing - only in chat tab
        query = st.text_input(
            "Enter your question",
            key="query_input",
            on_change=submit_query
        )
        
        if st.session_state.query:
            with st.spinner("🤖 GTUtor is thinking..."):
                # Add timestamp to the new message
                current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                
                if selected_subject:
                    try:
                        relevant_texts = get_relevant_passages(st.session_state.query, selected_subject)
                        chat_history = st.session_state.chat_histories.get(selected_subject, [])
                        rag_prompt = make_rag_prompt(st.session_state.query, relevant_texts, selected_subject, chat_history)
                        answer = generate_answer(rag_prompt)
                        
                        if not answer or "unable to answer" in answer.lower() or "do not contain" in answer.lower() or "sorry" in answer.lower() or "i apologize" in answer.lower() or "cannot help" in answer.lower() or "no information" in answer.lower() or "insufficient data" in answer.lower() or "not enough context" in answer.lower():
                            answer = generate_llm_answer(st.session_state.query, selected_subject, chat_history)
                    except Exception as e:
                        st.error(f"Error with RAG response: {str(e)}")
                        answer = generate_llm_answer(st.session_state.query, selected_subject, chat_history)
                else:
                    answer = generate_llm_answer(st.session_state.query)
                
                if answer:
                    # Create new message with timestamp
                    new_message = {
                        'human': st.session_state.query,
                        'ai': answer,
                        'timestamp': current_time.isoformat()
                    }
                    
                    # Display the new conversation
                    st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message">{st.session_state.query}</div></div>', unsafe_allow_html=True)
                    answer_html = markdown.markdown(answer, extensions=['tables'])
                    answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
                    st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message">{answer_html}</div></div>', unsafe_allow_html=True)
                    
                    # Copy buttons for the new conversation
                    unique_key = f"new_msg_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    
                    # Create two columns for copy buttons
                    copy_cols = st.columns(2)
                    
                    # Copy question button
                    if copy_cols[0].button("Copy Question", key=f"copy_q_{unique_key}"):
                        st.code(st.session_state.query, language=None)
                        st.toast("Question copied!")
                    
                    # Copy answer button
                    if copy_cols[1].button("Copy Answer", key=f"copy_a_{unique_key}"):
                        st.code(answer, language=None)
                        st.toast("Answer copied!")
                    
                    # Update chat history
                    if selected_subject:
                        st.session_state.chat_histories.setdefault(selected_subject, []).append(new_message)
                        save_chat_history(selected_subject, st.session_state.chat_histories[selected_subject])
                    
                    # Clear the query
                    st.session_state.query = ""
    
    with study_plan_tab:
        st.subheader(f"Generate Study Plan for {selected_subject}")
        
        col1, col2 = st.columns(2)
        with col1:
            col1_days, col2_days = st.columns(2)
            with col1_days:
                exam_date = st.date_input("Exam date")
            with col2_days:
                # Calculate days left automatically based on selected date
                today = datetime.now().date()
                days_difference = (exam_date - today).days
                days_left = max(1, days_difference)  # Ensure minimum 1 day
                st.info(f"{days_left} days left until exam")
                # Allow manual override
                days_left = st.number_input("Or manually set days left", min_value=1, max_value=60, value=days_left)
            
            student_level = st.selectbox("Your knowledge level", ["Beginner", "Intermediate", "Advanced", "Expert"], index=1)
            difficulty = st.selectbox("Subject difficulty", ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"], index=2)
        
        with col2:
            target_score = st.slider("Target score (%)", min_value=40, max_value=100, value=70, step=5)
            exam_type = st.selectbox("Exam type", ["Mid-Sem", "End-Sem", "Internal", "External", "Viva", "Submission"], index=1)
            hours_per_day = st.slider("Study hours per day", min_value=1, max_value=12, value=5)
                
        st.subheader("Topics Information")
        remaining_topics_text = st.text_area("Remaining topics to study (separate with commas)", height=100)
        remaining_topics = [topic.strip() for topic in remaining_topics_text.split(",") if topic.strip()]
        
        # Important topics - allow manual entry or AI extraction
        st.subheader("Important Topics")
        topic_input_method = st.radio("How would you like to add important topics?", 
                 ["Manual Entry", "Upload PYQ File", "Download From URL"])
        
        if topic_input_method == "Manual Entry":
            important_topics_text = st.text_area("Important topics (PYQ-based) (separate with commas)", height=100)
            important_topics = [topic.strip() for topic in important_topics_text.split(",") if topic.strip()]
        elif topic_input_method == "Upload PYQ File":
            pyq_file = st.file_uploader("Upload Previous Year Question Paper (PDF)", type="pdf", key="pyq_uploader")
            important_topics = []
            
            if pyq_file:
                with st.spinner("Analyzing question paper..."):
                    pdf_content = pyq_file.read()
                    try:
                        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf_file.write(pdf_content)
                        pdf_file.close()
                        
                        reader = PdfReader(pdf_file.name)
                        pyq_text = ""
                        for page in reader.pages:
                            pyq_text += page.extract_text()
                        
                        extraction_prompt = f"""
                        You are an expert academic analyst. Extract the most important topics from this GTU previous year question paper for {selected_subject}.
                        
                        Question Paper Content:
                        {pyq_text}
                        
                        Please identify 5-10 key topics that appear frequently or carry high marks in this paper.
                        Return ONLY a comma-separated list of topics, with no additional text or explanations.
                        """
                        
                        with st.spinner("Extracting important topics..."):
                            extracted_topics = generate_answer(extraction_prompt)
                            extracted_topics = extracted_topics.strip().strip('[]')
                            important_topics = [topic.strip() for topic in extracted_topics.split(",") if topic.strip()]
                            
                        os.unlink(pdf_file.name)
                        st.success(f"Successfully extracted {len(important_topics)} important topics!")
                        
                    except Exception as e:
                        st.error(f"Error extracting topics: {str(e)}")
        else:  # Download from URL
            pyq_url = st.text_input("Enter URL to Previous Year Question Paper")
            important_topics = []
            
            if pyq_url and st.button("Get Topics from URL", key="pyq_url_button"):
                with st.spinner("Downloading PDF from URL..."):
                    pdf_content = download_pdf(pyq_url)
                    if pdf_content:
                        with st.spinner("Analyzing question paper..."):
                            try:
                                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                                pdf_file.write(pdf_content)
                                pdf_file.close()
                                
                                reader = PdfReader(pdf_file.name)
                                pyq_text = ""
                                for page in reader.pages:
                                    pyq_text += page.extract_text()
                                
                                extraction_prompt = f"""
                                You are an expert academic analyst. Extract the most important topics from this GTU previous year question paper for {selected_subject}.
                                
                                Question Paper Content:
                                {pyq_text}
                                
                                Please identify 5-10 key topics that appear frequently or carry high marks in this paper.
                                Return ONLY a comma-separated list of topics, with no additional text or explanations.
                                """
                                
                                with st.spinner("Extracting important topics..."):
                                    extracted_topics = generate_answer(extraction_prompt)
                                    extracted_topics = extracted_topics.strip().strip('[]')
                                    important_topics = [topic.strip() for topic in extracted_topics.split(",") if topic.strip()]
                                    
                                os.unlink(pdf_file.name)
                                st.success(f"Successfully extracted {len(important_topics)} important topics!")
                                
                            except Exception as e:
                                st.error(f"Error extracting topics: {str(e)}")
                    else:
                        st.error("Failed to download PDF from the URL")
        
        # Display the important topics regardless of input method
        if important_topics:
            st.write("Important Topics:")
            st.info(", ".join(important_topics))
            
            # Allow editing the extracted topics
            edit_topics = st.checkbox("Edit important topics")
            if edit_topics:
                topics_text = ", ".join(important_topics)
                edited_topics = st.text_area("Edit topics:", value=topics_text, height=100)
                important_topics = [topic.strip() for topic in edited_topics.split(",") if topic.strip()]
        
        # Full syllabus section with multiple input options
        st.subheader("Full Syllabus")
        syllabus_input_method = st.radio("How would you like to add full syllabus?", 
                 ["Manual Entry", "Upload Syllabus File", "Download From URL"])
        
        if syllabus_input_method == "Manual Entry":
            full_syllabus_text = st.text_area("Full syllabus (separate with commas)", height=100)
            full_syllabus = [topic.strip() for topic in full_syllabus_text.split(",") if topic.strip()]
        elif syllabus_input_method == "Upload Syllabus File":
            syllabus_file = st.file_uploader("Upload Syllabus PDF", type="pdf", key="syllabus_uploader")
            if syllabus_file:
                with st.spinner("Processing syllabus..."):
                    try:
                        pdf_content = syllabus_file.read()
                        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf_file.write(pdf_content)
                        pdf_file.close()
                        
                        reader = PdfReader(pdf_file.name)
                        syllabus_text = ""
                        for page in reader.pages:
                            syllabus_text += page.extract_text()
                        
                        extraction_prompt = f"""
                        Extract the full syllabus topics from this document for {selected_subject}.
                        Return ONLY a comma-separated list of topics, with no additional text.
                        Make sure to include all chapters, units, and individual topics.
                        """
                        
                        with st.spinner("Extracting syllabus topics..."):
                            extracted_syllabus = generate_answer(extraction_prompt + "\n\n" + syllabus_text)
                            # Clean up the response
                            extracted_syllabus = extracted_syllabus.strip().strip('[]')
                            full_syllabus = [topic.strip() for topic in extracted_syllabus.split(",") if topic.strip()]
                            
                        os.unlink(pdf_file.name)
                        st.success(f"Successfully extracted {len(full_syllabus)} syllabus topics!")
                    except Exception as e:
                        st.error(f"Error extracting syllabus: {str(e)}")
                        full_syllabus = []
            else:
                full_syllabus = []
        else:  # Download from URL
            syllabus_url = st.text_input("Enter URL to Syllabus PDF")
            if syllabus_url and st.button("Get Syllabus from URL", key="syllabus_url_button"):
                with st.spinner("Downloading syllabus PDF..."):
                    pdf_content = download_pdf(syllabus_url)
                    if pdf_content:
                        with st.spinner("Processing syllabus..."):
                            try:
                                pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                                pdf_file.write(pdf_content)
                                pdf_file.close()
                                
                                reader = PdfReader(pdf_file.name)
                                syllabus_text = ""
                                for page in reader.pages:
                                    syllabus_text += page.extract_text()
                                
                                extraction_prompt = f"""
                                Extract the full syllabus topics from this document for {selected_subject}.
                                Return ONLY a comma-separated list of topics, with no additional text.
                                Make sure to include all chapters, units, and individual topics.
                                """
                                
                                with st.spinner("Extracting syllabus topics..."):
                                    extracted_syllabus = generate_answer(extraction_prompt + "\n\n" + syllabus_text)
                                    # Clean up the response
                                    extracted_syllabus = extracted_syllabus.strip().strip('[]')
                                    full_syllabus = [topic.strip() for topic in extracted_syllabus.split(",") if topic.strip()]
                                    
                                os.unlink(pdf_file.name)
                                st.success(f"Successfully extracted {len(full_syllabus)} syllabus topics!")
                            except Exception as e:
                                st.error(f"Error extracting syllabus: {str(e)}")
                                full_syllabus = []
                    else:
                        st.error("Failed to download PDF from the URL")
                        full_syllabus = []
            else:
                full_syllabus = []
            
        # Display the extracted syllabus topics and allow editing
        if full_syllabus:
            st.write("Full Syllabus Topics:")
            syllabus_text = ", ".join(full_syllabus)
            edited_syllabus = st.text_area("Edit extracted syllabus if needed:", value=syllabus_text, height=100)
            full_syllabus = [topic.strip() for topic in edited_syllabus.split(",") if topic.strip()]
        
        if st.button("Generate Study Plan"):
            with st.spinner("Creating your personalized study plan..."):
                plan_response = generate_study_plan(
                    subject=selected_subject,
                    days_left=days_left,
                    student_level=student_level,
                    difficulty=difficulty,
                    target_score=target_score,
                    exam_type=exam_type,
                    remaining_topics=remaining_topics,
                    important_topics=important_topics,
                    full_syllabus=full_syllabus,
                    hours_per_day=hours_per_day
                )
                
            try:
                # Try to parse as JSON
                import re
                json_match = re.search(r'({[\s\S]*})', plan_response)
                if json_match:
                    json_str = json_match.group(1)
                    
                    # Check if JSON appears truncated
                    if json_str.count('{') > json_str.count('}'):
                        st.warning("The study plan appears to be truncated. Displaying as text instead.")
                        st.markdown("### Your Study Plan:")
                        st.markdown(plan_response)
                    else:
                        try:
                            plan_json = json.loads(json_str)
                            
                            st.success("✅ Your personalized study plan is ready!")
                            
                            # Display summary information
                            st.markdown(f"""
                            ### 📊 Study Plan Summary
                            - **Subject:** {selected_subject}
                            - **Days until exam:** {days_left}
                            - **Total study hours planned:** {days_left * hours_per_day}
                            - **Target score:** {target_score}%
                            """)
                            
                            # Display the plan in a more visual and detailed way
                            for day, topics in plan_json.items():
                                st.markdown(f"## {day}")
                                for topic in topics:
                                    st.markdown(f"### 📘 {topic['topic']} ({topic['time']})")
                                    
                                    # Display key concepts with better error handling
                                    if 'key_concepts' in topic and topic['key_concepts']:
                                        st.markdown("#### 🔑 Key Concepts:")
                                        for concept in topic['key_concepts']:
                                            st.markdown(f"- {concept}")
                                    
                                    # Display important questions
                                    if 'important_questions' in topic and topic['important_questions']:
                                        st.markdown("#### ❓ Important Questions/Problems:")
                                        for question in topic['important_questions']:
                                            st.markdown(f"- {question}")
                                    
                                    # Display study activities
                                    if 'activities' in topic and topic['activities']:
                                        st.markdown("#### 🛠️ Study Activities:")
                                        for activity in topic['activities']:
                                            st.markdown(f"- {activity}")
                                    
                                    # Display tips
                                    if 'tips' in topic and topic['tips']:
                                        st.markdown(f"#### 💡 Study Tip: *{topic['tips']}*")
                                    
                                    st.markdown("---")

                            # Then provide the raw JSON view separately
                            with st.expander("View Raw Study Plan JSON"):
                                st.json(plan_json)
                                
                            # Download buttons for the study plan in different formats
                            col1, col2 = st.columns(2)
                            
                            # JSON download
                            json_str = json.dumps(plan_json, indent=2)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            json_filename = f"{selected_subject.lower().replace(' ', '_')}_study_plan_{timestamp}.json"
                            
                            with col1:
                                st.download_button(
                                    label="⬇️ Download JSON Study Plan",
                                    data=json_str,
                                    file_name=json_filename,
                                    mime="application/json",
                                    key=f"download_study_plan_json_{timestamp}"
                                )
                            
                            # Markdown download (more human-readable)
                            markdown_content = f"# Study Plan for {selected_subject}\n\n"
                            markdown_content += f"- **Days until exam:** {days_left}\n"
                            markdown_content += f"- **Daily study hours:** {hours_per_day}\n"
                            markdown_content += f"- **Target score:** {target_score}%\n\n"
                            
                            for day, topics in plan_json.items():
                                markdown_content += f"## {day}\n\n"
                                for topic in topics:
                                    markdown_content += f"### {topic['topic']} ({topic['time']})\n\n"
                                    
                                    if 'key_concepts' in topic and topic['key_concepts']:
                                        markdown_content += "#### Key Concepts:\n"
                                        for concept in topic['key_concepts']:
                                            markdown_content += f"- {concept}\n"
                                        markdown_content += "\n"
                                    
                                    if 'important_questions' in topic and topic['important_questions']:
                                        markdown_content += "#### Important Questions/Problems:\n"
                                        for question in topic['important_questions']:
                                            markdown_content += f"- {question}\n"
                                        markdown_content += "\n"
                                    
                                    if 'activities' in topic and topic['activities']:
                                        markdown_content += "#### Study Activities:\n"
                                        for activity in topic['activities']:
                                            markdown_content += f"- {activity}\n"
                                        markdown_content += "\n"
                                    
                                    if 'tips' in topic and topic['tips']:
                                        markdown_content += f"#### Study Tip: *{topic['tips']}*\n\n"
                                        
                                    markdown_content += "---\n\n"
                            
                            md_filename = f"{selected_subject.lower().replace(' ', '_')}_study_plan_{timestamp}.md"
                            with col2:
                                st.download_button(
                                    label="⬇️ Download Markdown Study Plan",
                                    data=markdown_content,
                                    file_name=md_filename,
                                    mime="text/markdown",
                                    key=f"download_study_plan_md_{timestamp}"  # Fixed unique key
                                )
                        except json.JSONDecodeError:
                            # If JSON parsing fails, display as text
                            st.warning("Couldn't parse the plan as JSON. Displaying raw response:")
                            st.markdown(plan_response)
                else:
                    # If no JSON match found, display as text
                    st.markdown("### Your Study Plan:")
                    st.markdown(plan_response)
            except Exception as e:
                st.error(f"Error processing study plan: {str(e)}")
                st.markdown(plan_response)
else:
    # When no subject is selected, show welcome message
    st.info("👋 Welcome to GTUtor! Please select or create a subject to get started.")
    st.markdown("""
    ### Quick Start Guide:
    1. Select a subject from the dropdown above or create a new one
    2. Upload your study materials (PDF format)
    3. Start asking questions in the chat tab
    4. Generate personalized study plans in the study plan tab
    """)
    st.markdown("### 💬 General Chat")
    st.markdown("Ask any question without selecting a specific subject")
    
    # Query input for general chat
    general_query = st.text_input("Enter your question", key="general_query_input")
    
    if general_query:
        with st.spinner("🤖 GTUtor is thinking..."):
            # Generate answer without using any specific subject knowledge
            answer = generate_llm_answer(general_query)
            
            if answer:
                # Add timestamp
                current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                formatted_time = current_time.strftime("%d-%m-%y %I:%M %p")
                
                # Display the conversation
                st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhCtDRFGo8W5fLw1wg12N0zHKONLsTXgY3Ko1MDaYBc2INdt3-EU1MGJR9Thaq9lzC730&usqp=CAU"/></div><div class="message"><div class="timestamp" style="font-size: 0.9em;">{formatted_time}</div><div class="content">{general_query}</div></div></div>', unsafe_allow_html=True)
                
                answer_html = markdown.markdown(answer, extensions=['tables'])
                answer_html = answer_html.replace("```python", "```").replace("```PowerShell", "```").replace("```javascript", "```").replace("```java", "```").replace("```sql", "```").replace("```css", "```").replace("```html", "```").replace("```"," ")
                st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://img.freepik.com/premium-vector/ai-logo-template-vector-with-white-background_1023984-15069.jpg"/></div><div class="message"><div class="content">{answer_html}</div></div></div>', unsafe_allow_html=True)
                
                # Copy buttons for the conversation
                unique_key = f"general_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                
                # Create columns for copy buttons
                copy_cols = st.columns(2)
                
                # Copy question button
                if copy_cols[0].button("Copy Question", key=f"copy_q_{unique_key}"):
                    st.code(general_query, language=None)
                    st.toast("Question copied!")
                
                # Copy answer button
                if copy_cols[1].button("Copy Answer", key=f"copy_a_{unique_key}"):
                    st.code(answer, language=None)
                    st.toast("Answer copied!")

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

# Update the sidebar information display
if selected_subject:
    st.sidebar.markdown(f"""
    ### 📊 Current Subject Stats
    - **Subject**: {selected_subject}
    - **Documents**: {get_document_count(selected_subject)}
    - **Messages**: {len(st.session_state.chat_histories.get(selected_subject, []))}
    """)

# Add sidebar controls and information
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
- 📅 **Study Planning**: Generate personalized study plans based on your needs and timeline
<div>
        <h3>🚀 Getting Started</h3>
        <ol style='font-size: 1.1rem;'>
            <li>Select or create a subject from the dropdown below</li>
            <li>Upload your study materials (PDF format)</li>
            <li>Start asking questions and learn interactively!</li>
        </ol>
</div>

Made with ❤️ for GTU students
""", unsafe_allow_html=True)

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
        db_path = os.path.join(vector_stores_folder, f"{selected_subject.lower().replace(' ', '_')}.pkl")
        if os.path.exists(db_path):
            os.remove(db_path)
        
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
        st.session_state.chat_histories[selected_subject] = []
        save_chat_history(selected_subject, [])
        st.success("New conversation started.")
        st.rerun()
else:
    st.sidebar.info("Select or create a subject to see available controls.")

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
# Add disclaimer in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Disclaimer")
st.sidebar.markdown("""
- Uploaded PDFs are processed and stored as text embeddings only, not in their original PDF form.
- Please do not upload documents containing highly sensitive or confidential information.
- GTUtor may occasionally make mistakes or provide incorrect information. Always verify important information from reliable sources.
- Report any issues or bugs to improve the system.
""")
