import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langdetect import detect
from typing import List, Tuple, Dict
import numpy as np
import json
import tenacity
import os
import re
import tempfile
from typing import List, Tuple
import requests
from deep_translator import GoogleTranslator
from llama_parse import LlamaParse

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "jqfl7ja6Y6MdryHqcpgcLCNL7z19n2I3"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Set up LlamaParse
LLAMA_CLOUD_API_KEY = "llx-k52AkEbC7j1x91hHOGVh5UfF9phdJ7OMMj62BHRKWbkRFfbU"
with open('language_mappings.json', 'r') as file:
    LANGUAGE_CODES = json.load(file)

documents = []
uploaded_file_names = []
translated_documents = {}
vector_store = None
embeddings = None
latest_response = ""
current_language = 'en'
translator = GoogleTranslator()
llama_parse = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")

@st.cache_data
def extract_text_from_pdf(file_path: str) -> List[Document]:
    try:
        document = llama_parse.load_data(file_path)
        doc_list = []
        for page_number, page in enumerate(document, start=1):
            doc = Document(page_content=page.text, metadata={"page_number": page_number})
            doc_list.append(doc)
        return doc_list
    except Exception as e:
        st.error(f"Error extracting text from PDF using LlamaParse: {e}")
        return []

def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = []
    for document in documents:
        splits = text_splitter.split_documents([document])
        for split in splits:
            split.metadata.update(document.metadata)  # Propagate metadata
            split.metadata['page_number'] = document.metadata.get('page_number', 'unknown')
            split.metadata['document_name'] = document.metadata.get('document_name', 'unknown')
        all_splits.extend(splits)
    return all_splits

@st.cache_data(show_spinner=False)
def serialize_text_chunks(_text_chunks):
    """Serialize text chunks to a hashable format (e.g., JSON string)."""
    return json.dumps([{
        'page_content': chunk.page_content,
        'metadata': chunk.metadata
    } for chunk in _text_chunks], ensure_ascii=False)

@st.cache_resource(show_spinner=False)
def create_faiss_index(_serialized_text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="faiss_index"):
    text_chunks = json.loads(_serialized_text_chunks)
    if not text_chunks:
        raise ValueError("No text chunks to index.")
    documents = []
    for chunk in text_chunks:
        doc = Document(page_content=chunk['page_content'], metadata={
            'document_name': chunk['metadata'].get('document_name', 'unknown'),
            'page_number': chunk['metadata'].get('page_number', 'unknown')
        })
        documents.append(doc)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(index_path)
    return vector_store, embeddings

@st.cache_resource(show_spinner=False)
def load_faiss_index(index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.load_local(index_path, embeddings)
    return vector_store, embeddings

@st.cache_data(show_spinner=False)
def detect_language(text: str) -> str:
    try:
        detected_language = detect(text)
        return detected_language
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        return 'en'

@st.cache_data(show_spinner=False)
def translate_text(_documents: List[Document], source_lang: str = 'auto', target_lang: str = 'en') -> List[str]:
    translated_pages = []
    for document in _documents:
        if document.metadata['document_name'] not in st.session_state.translated_documents:
            pages = []
            for page in _documents:
                if page.metadata['document_name'] == document.metadata['document_name']:
                    pages.append(page.page_content)
            translated_document = []
            for page in pages:
                max_chunk_size = 5000
                text_chunks = [page[i:i + max_chunk_size] for i in range(0, len(page), max_chunk_size)]
                translated_chunks = []
                for chunk in text_chunks:
                    translated_chunk = GoogleTranslator(source=source_lang, target=target_lang).translate(chunk)
                    translated_chunks.append(translated_chunk)
                translated_document.append(' '.join(translated_chunks))
            translated_pages.append('\n\n'.join(translated_document))
            st.session_state.translated_documents[document.metadata['document_name']] = '\n\n'.join(translated_document)
        else:
            translated_pages.append(st.session_state.translated_documents[document.metadata['document_name']])
    return translated_pages

def retrieve_chunks(query, num_results=5):
    if vector_store is None:
        raise ValueError("FAISS index not loaded. Please ensure the FAISS index is created and loaded before querying.")
    
    response = vector_store.similarity_search(query, k=num_results)
    documents = [doc for doc in response if isinstance(doc, Document)]
    return documents

def clean_text(text):
    if isinstance(text, list):
        return ' '.join([re.sub(r'\s+', ' ', t) for t in text])
    return re.sub(r'\s+', ' ', text) 

def generate_pdf_links(sources_info):
    links = []
    for source in sources_info:
        file_name = source['document_name']
        page_number = source['page_number']
        temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
        url = f"{temp_file_path}#page={page_number}"
        link = f"[{file_name} (Page {page_number})]({url})"
        links.append(link)
    return links

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_if_exception_type(Exception))
def query_mistral(messages):
    data = {
        "model": "mistral-small-latest",
        "messages": messages,
        "max_tokens": 512
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    elif response.status_code == 429:
        raise Exception("Rate limit exceeded")
    else:
        return f"Error: {response.status_code}, {response.text}"

def extract_answer_from_chain(context, question):
    system_prompt = """You are a well-informed and proficient assistant. Your responses must be clear, detailed, and concise. 
                       Question: {question}
                       Context: {context}
                       Answer:"""

    prompt = system_prompt.format(context=clean_text(context), question=clean_text(question))
    result = query_mistral([{"role": "system", "content": prompt}])

    return result, []
stop_words = {"what","where","when","who","why","how","is","are","am","be","been","being","have","has","had","do","does","did","will","would",
    "shall","should","can","could","may","might","must","ought","shall","should","will","would"}

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by converting it to lowercase, removing punctuation, and removing stop words.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def calculate_relevance(query: str, chunks: List[str]) -> List[float]:
    """
    Calculate the relevance of each chunk to the query.
    """
    preprocessed_query = preprocess_text(query)
    preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
    
    vectorizer = TfidfVectorizer().fit([preprocessed_query] + preprocessed_chunks)
    query_vector = vectorizer.transform([preprocessed_query])
    chunk_vectors = vectorizer.transform(preprocessed_chunks)
    
    relevance_scores = cosine_similarity(query_vector, chunk_vectors)[0]
    return list(relevance_scores)

def analyze_query_complexity(query: str) -> Dict[str, any]:
    """
    Analyze the complexity of the query.
    """
    parts = re.split(r'\band\b|\bor\b|,', query.lower())
    return {
        "num_parts": len(parts),
        "has_multiple_questions": len(re.findall(r'\?', query)) > 1 or any(word in query.lower() for word in stop_words)
    }

def check_query_ambiguity(query: str, chunks: List[str]) -> Tuple[bool, Dict[str, any]]:
    """
    Check if the query is ambiguous.
    """
    relevance_scores = calculate_relevance(query, chunks)
    query_analysis = analyze_query_complexity(query)
    
    max_score = max(relevance_scores)
    mean_score = np.mean(relevance_scores)
    std_score = np.std(relevance_scores)
    
    is_ambiguous = False
    reasons = []

    if query_analysis["has_multiple_questions"] and query_analysis["num_parts"] > 2:
        is_ambiguous = True
        reasons.append("Query contains multiple distinct questions or comparisons")

    if max_score < 1.5 * mean_score:
        is_ambiguous = True
        reasons.append("No chunk is significantly more relevant than others")

    if std_score < 0.05 * max_score:
        is_ambiguous = True
        reasons.append("All chunks have similar relevance, query might be too broad")

    return is_ambiguous, {
        "query_analysis": query_analysis,
        "relevance_scores": relevance_scores,
        "max_score": max_score,
        "mean_score": mean_score,
        "std_score": std_score,
        "reasons_for_ambiguity": reasons if is_ambiguous else ["Query appears unambiguous"]
    }

def generate_follow_up_questions(question: str, answer: str) -> List[str]:
    system_prompt = """Based on the following context, question, and answer, generate three specific follow-up questions that a user might ask next. Do not include any introductory text or numbering.
    
    Original Question: {question}
    Answer: {answer}
    
    Clarifying Follow-up Questions:"""

    prompt = system_prompt.format(question=clean_text(question), answer=clean_text(answer))
    result = query_mistral([{"role": "system", "content": prompt}])
    
    questions = result.split('\n')
    questions = [q.split('.', 1)[-1].strip() for q in questions if q.strip()]
    
    return questions[:3]

def ask_follow_up_questions(query: str, chunks: List[str]) -> List[str]:
    """
    Ask follow-up questions based on the query and chunks.
    """
    is_ambiguous, analysis = check_query_ambiguity(query, chunks)
    if is_ambiguous:
        # If the query is ambiguous, generate follow-up questions to clarify the query
        follow_up_questions = generate_follow_up_questions(query)
        return follow_up_questions
    else:
        # If the query is not ambiguous, return an empty list
        return []

def calculate_similarity(text1, text2):
    # Use TF-IDF vectorization and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]

def find_similar_documents(answer, documents):
    # Calculate similarity between answer and each document
    similarities = []
    for document in documents:
        similarity = calculate_similarity(answer, document['page_content'])
        similarities.append((document, similarity))
    # Sort documents by similarity and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    threshold = 0.7  
    return [(doc, sim) for doc, sim in similarities if sim > threshold]

def generate_pdf_link(source):
    if isinstance(source, dict):
        file_name = source.get('document_name', 'Unknown')
        page_number = source.get('page_number', 'Unknown')
        link = f"{file_name}_{page_number}.pdf"
        return link
    else:
        print("Error: Source is not a dictionary")
        return None

def handle_query(query: str, source_lang: str) -> Tuple[str, str, str, List[str]]:
    detected_language = detect_language(query)
    if detected_language != 'en':
        query = GoogleTranslator(source=detected_language, target='en').translate(query)  # Use GoogleTranslator for string translation
    response = retrieve_chunks(query)
    if not response:
        return "No relevant information found.", "", "en", []
    combined_context = " ".join([chunk.page_content for chunk in response])
    answer, context = extract_answer_from_chain(combined_context, query)
    if source_lang != 'en':
        answer = GoogleTranslator(source='en', target=source_lang).translate(answer)  # Use GoogleTranslator for string translation
    complete_answers = handle_long_responses(answer)

    chunks = [chunk.page_content for chunk in response]
    follow_up_questions = ask_follow_up_questions(query, chunks)
    if source_lang != 'en':
        follow_up_questions = [GoogleTranslator(source='en', target=source_lang).translate(q) for q in follow_up_questions]

    sources_info = []
    for chunk in response:
        sources_info.append({
            'document_name': chunk.metadata.get('document_name', 'unknown'),
            'page_number': chunk.metadata.get('page_number', 'unknown'),
            'page_content': chunk.page_content
        })
    similar_documents = find_similar_documents(answer, sources_info)
    print(similar_documents)
    links = []
    for document, similarity in similar_documents:
        if isinstance(document, dict):
            pdf_name = document.get('document_name', 'Unknown')
            page_number = document.get('page_number', 'Unknown')
            link = generate_pdf_link(document)  
            links.append(f"[{pdf_name} (Page {page_number})]({link})")
        else:
            print("Error: Document is not a dictionary")
    sources_info_str_with_links = ", ".join(links)
    return " ".join(complete_answers) + f"\n\nSources: {sources_info_str_with_links}", detected_language, follow_up_questions


def handle_long_responses(answer):
    max_length = 2048
    if len(answer) > max_length:
        parts = [answer[i:i + max_length] for i in range(0, len(answer), max_length)]
        return parts
    else:
        return [answer]

def main():
    global documents, uploaded_file_names, translated_documents, vector_store, embeddings, latest_response, current_language

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []
    if "current_language" not in st.session_state:
        st.session_state.current_language = 'en'
    if 'translated_documents' not in st.session_state:
        st.session_state.translated_documents = {}
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None

    st.markdown("""
        <style>
            .stButton button {
                background-color: #007bff !important;
                color: white !important;
                border-radius: 5px !important;
                padding: 0.5em 1em !important;
            }
            .stButtonWhite button {
                background-color: white !important;
                color: black !important;
                border: 1px solid #007bff !important;
                border-radius: 5px !important;
                padding: 0.5em 1em !important;
            }
            .logo {
                position: absolute;
                top: 10px;
                right: 10px;
                width: 100px;
            }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.image("narwal_logo.png", use_column_width=True)
    st.title("OmniDoc AI Assistant: Advanced Multilingual Document Bot")
    st.write("""
        Welcome to **OmniDoc AI Assistant**, your comprehensive AI-powered assistant for all your document needs. 
        Our cutting-edge tool is designed to streamline your document handling and enhance your productivity with the following key features:

        1. **Handles Texts and Tables:** OmniDoc seamlessly extracts and processes both textual content and tabular data from your documents.
        2. **Performs Q&A and Summarizations:** Get precise answers to your queries and concise summaries of lengthy documents with ease.
        3. **Multilingual Capabilities:** Break language barriers effortlessly as OmniDoc supports document translation and interaction in multiple languages.
        4. **Provides Follow-Up Questions:** Engage in deeper, more meaningful interactions with our intelligent follow-up question generation.

        Experience the future of document management with OmniDoc AI Assistant!
    """)

    uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True, key="file_uploader")

    if uploaded_files:
        documents = []
        uploaded_file_names = []
        translated_documents = {}
        vector_store = None
        embeddings = None
        latest_response = ""
        current_language = 'en'

        st.write(f"Processing uploaded files.")
        file_names = []
        all_texts = []

        for file in uploaded_files:
            file_names.append(file.name)
            file_content = file.read()
            temp_file_path = os.path.join(tempfile.gettempdir(), file.name)

            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file_content)

            docs = extract_text_from_pdf(temp_file_path)

            if docs:
                all_texts.append((docs, file.name))
            else:
                st.write(f"No text extracted from {file.name}")

        if not all_texts:
            st.write("No valid text found in the uploaded documents.")
        else:
            all_documents = []
            for docs, file_name in all_texts:
                text_chunks = split_text_into_chunks(docs)
                for chunk in text_chunks:
                    chunk.metadata.update({'document_name': file_name})  # Update metadata with document name
                all_documents.extend(text_chunks)

            documents = all_documents
            uploaded_file_names = file_names
            st.success("Files processed successfully!")

            serialized_text_chunks = serialize_text_chunks(all_documents)
            vector_store, embeddings = create_faiss_index(serialized_text_chunks)

    
    if uploaded_file_names:
        st.write("Uploaded Files:")
        selected_file = st.selectbox("Select a file to translate", uploaded_file_names, key="select_file")

        if selected_file:
            target_lang_name = st.selectbox("Select target language:", list(LANGUAGE_CODES.keys()), key="select_language")
            target_lang = LANGUAGE_CODES.get(target_lang_name)

            if st.button("Translate Document", key="translate_button"):
                selected_doc = [doc for doc in documents if doc.metadata['document_name'] == selected_file]
                translated_pages = translate_text(selected_doc, source_lang='en', target_lang=target_lang)
                st.session_state.translated_document = '\n\n'.join(translated_pages)
                st.session_state.selected_document = selected_file
                st.session_state.current_language = target_lang
                st.success(f"Document has been translated to {target_lang_name}. You can now ask questions in {target_lang_name}.")

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    if st.session_state.followup_questions:
        st.write("Follow-up Questions:")
        cols = st.columns(len(st.session_state.followup_questions))
        for i, (col, question) in enumerate(zip(cols, st.session_state.followup_questions)):
            with col:
                if st.button(f"{i+1}. {question}", key=f"followup_{i}"):
                    st.session_state.query = question
                    break
        else:
            st.session_state.query = st.chat_input("You:", key="chat_input")
    else:
        st.session_state.query = st.chat_input("You:", key="chat_input")

    if st.session_state.query:
        with st.chat_message("user"):
            st.write(st.session_state.query)
        st.session_state.chat_history.append(("user", st.session_state.query))

         # Add a loading animation with an emoji
        with st.chat_message("assistant"):
            st.write("🤔 Thinking...")

        placeholder = st.empty()

        answer, _, followup_questions = handle_query(st.session_state.query, st.session_state.current_language)
        placeholder.markdown(answer, unsafe_allow_html=True)
        st.session_state.chat_history.append(("assistant", answer))
        st.session_state.latest_response = answer
        st.session_state.followup_questions = followup_questions
        st.experimental_rerun()

if __name__ == "__main__":
    main()
