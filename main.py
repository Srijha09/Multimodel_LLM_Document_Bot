import streamlit as st
from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langdetect import detect
import json
import tenacity
import os
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

class DocumentChatbot:
    def __init__(self):
        self.documents = []
        self.uploaded_file_names = []
        self.translated_documents = {}  
        self.vector_store = None
        self.embeddings = None
        self.latest_response = ""
        self.current_language = 'en'
        self.translator = GoogleTranslator()
        self.llama_parse = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")

    def clear_session_state(self):
        self.documents = []
        self.uploaded_file_names = []
        self.translated_documents = {}
        self.vector_store = None
        self.embeddings = None
        self.latest_response = ""
        self.current_language = 'en'

    @st.cache_data
    def extract_text_from_pdf(_self, file_path: str) -> List[Document]:
        try:
            document = _self.llama_parse.load_data(file_path)
            doc_list = []
            for page_number, page in enumerate(document, start=1):
                doc = Document(page_content=page.text, metadata={"page_number": page_number})
                doc_list.append(doc)
            return doc_list
        except Exception as e:
            print(f"Error extracting text from PDF using LlamaParse: {e}")
            return []

    def split_text_into_chunks(self, documents, chunk_size=1000, chunk_overlap=500):
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

    def create_faiss_index(self, text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="faiss_index"):
        if not text_chunks:
            raise ValueError("No text chunks to index.")
        documents = []
        for chunk in text_chunks:
            doc = Document(page_content=chunk.page_content, metadata={
                'document_name': chunk.metadata.get('document_name', 'unknown'),
                'page_number': chunk.metadata.get('page_number', 'unknown')
            })
            documents.append(doc)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        self.vector_store = vector_store
        self.embeddings = embeddings
        print("FAISS index created successfully.")

    def load_faiss_index(self, index_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = FAISS.load_local(index_path, embeddings)
        self.vector_store = vector_store
        self.embeddings = embeddings
        print("FAISS index loaded successfully.")

    @st.cache_data
    def detect_language(_self, text: str) -> str:
        try:
            detected_language = detect(text)
            return detected_language
        except Exception as e:
            print(f"Error detecting language: {e}")
        return 'en'

    @st.cache_data
    def translate_text(_self, text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        max_chunk_size = 5000
        text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        translated_chunks = []
        for chunk in text_chunks:
            translated_chunk = _self.translator.translate(chunk, src=source_lang, dest=target_lang)
            translated_chunks.append(translated_chunk)
        return ' '.join(translated_chunks)

    def retrieve_chunks(self, query, num_results=5):
        if self.vector_store is None:
            raise ValueError("FAISS index not loaded. Please ensure the FAISS index is created and loaded before querying.")
        
        response = self.vector_store.similarity_search(query, k=num_results)
        documents = [doc for doc in response if isinstance(doc, Document)]
        return documents

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(5), retry=tenacity.retry_if_exception_type(Exception))
    def query_mistral(self, messages):
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

    def extract_answer_from_chain(self, context, question):
        system_prompt = """You are a well-informed and proficient assistant. Your responses must be clear, detailed, and concise. 
                           Question: {question}
                           Context: {context}
                           Answer:"""

        prompt = system_prompt.format(context=context, question=question)
        result = self.query_mistral([{"role": "system", "content": prompt}])

        return result, []

    def is_question_ambiguous(self, question: str, answer: str) -> bool:
        system_prompt = """Analyze the following question. Determine if the question is ambiguous and could benefit from clarification.
        Respond with 'Yes' if the question is ambiguous, or 'No' if it's clear and specific.

        Question: {question}
        Answer: {answer}

        Is the question ambiguous?"""

        prompt = system_prompt.format(question=question, answer=answer)
        result = self.query_mistral([{"role": "system", "content": prompt}])
        
        return result.strip().lower() == 'yes'
    
    def generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        system_prompt = """Based on the following context, question, and answer, generate three specific follow-up questions that a user might ask next. Do not include any introductory text or numbering.
        
        Original Question: {question}
        Answer: {answer}
        
        Clarifying Follow-up Questions:"""

        prompt = system_prompt.format(question=question, answer=answer)
        result = self.query_mistral([{"role": "system", "content": prompt}])
        
        questions = result.split('\n')
        questions = [q.split('.', 1)[-1].strip() for q in questions if q.strip()]
        
        return questions[:3]
    
    def handle_query(self, query: str, source_lang: str) -> Tuple[str, str, str, List[str]]:
        print(f"Handling query: {query}")
        detected_language = self.detect_language(query)
        if detected_language != 'en':
            query = self.translate_text(query, source_lang=detected_language, target_lang='en')

        response = self.retrieve_chunks(query)
        if not response:
            return "No relevant information found.", "", "en", []

        combined_context = " ".join([chunk.page_content for chunk in response])
        answer, context = self.extract_answer_from_chain(combined_context, query)

        if source_lang != 'en':
            answer = self.translate_text(answer, source_lang='en', target_lang=source_lang)
        complete_answer = self.handle_long_responses(answer)

        follow_up_questions = []
        #if self.is_question_ambiguous(query, complete_answer):
        follow_up_questions = self.generate_follow_up_questions(query, complete_answer)

        sources_info = []
        for chunk in response:
            sources_info.append({
                'document_name': chunk.metadata.get('document_name', 'unknown'),
                'page_number': chunk.metadata.get('page_number', 'unknown')
            })

        doc_names = [source['document_name'] for source in sources_info]
        most_frequent_doc = max(set(doc_names), key=doc_names.count)

        filtered_sources_info = [source for source in sources_info if source['document_name'] == most_frequent_doc]

        page_numbers = [source['page_number'] for source in filtered_sources_info]
        if len(set(page_numbers)) == 1:
            sources_info_str = f"{most_frequent_doc} (Page {page_numbers[0]})"
        else:
            sources_info_str = ", ".join([f"{most_frequent_doc} (Page {page_number})" for page_number in set(page_numbers)])

        return " ".join(complete_answer) + f"\n\nSources: {sources_info_str}", " ".join(complete_answer), detected_language, follow_up_questions

    def handle_long_responses(self, answer):
        max_length = 2048
        if len(answer) > max_length:
            parts = [answer[i:i + max_length] for i in range(0, len(answer), max_length)]
            return parts
        else:
            return [answer]

chatbot = DocumentChatbot()

def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if "followup_questions" not in st.session_state:
        st.session_state.followup_questions = []
    if "current_language" not in st.session_state:
        st.session_state.current_language = 'en'

    st.markdown("""
        <style>
            .stButton button {
                background-color: #007bff !important;
                color: white !important;
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
    st.title("TranslateXpert: Advanced Multilingual Document Bot")
    st.write("Welcome to TranslateXpert, your AI-powered assistant for document translation, QnA and summarizations!")

    uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True, key="file_uploader")

    if uploaded_files:
        chatbot.clear_session_state()

        st.write(f"Processing uploaded files.")
        file_names = []
        all_texts = []

        for file in uploaded_files:
            file_names.append(file.name)
            file_content = file.read()
            temp_file_path = os.path.join(tempfile.gettempdir(), file.name)

            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file_content)

            documents = chatbot.extract_text_from_pdf(temp_file_path)

            if documents:
                all_texts.append((documents, file.name))
            else:
                st.write(f"No text extracted from {file.name}")

        if not all_texts:
            st.write("No valid text found in the uploaded documents.")
        else:
            all_documents = []
            for documents, file_name in all_texts:
                text_chunks = chatbot.split_text_into_chunks(documents)
                for chunk in text_chunks:
                    chunk.metadata.update({'document_name': file_name})  # Update metadata with document name
                all_documents.extend(text_chunks)

            chatbot.documents = all_documents
            chatbot.uploaded_file_names = file_names
            st.success("Files processed successfully!")

            chatbot.create_faiss_index(all_documents)

    if chatbot.uploaded_file_names:
        st.write("Uploaded Files:")
        selected_file = st.selectbox("Select a file to translate", chatbot.uploaded_file_names, key="select_file")

        if selected_file:
            target_lang_name = st.selectbox("Select target language:", list(LANGUAGE_CODES.keys()), key="select_language")
            target_lang = LANGUAGE_CODES.get(target_lang_name)

            if st.button("Translate Document", key="translate_button"):
                selected_doc_text = [doc.page_content for doc in chatbot.documents if doc.metadata['document_name'] == selected_file]
                combined_text = " ".join(selected_doc_text)
                translated_text = chatbot.translate_text(combined_text, target_lang=target_lang)
                st.write("Translated Text:")
                st.write(translated_text)
                chatbot.translated_documents[selected_file] = translated_text
                chatbot.latest_response = translated_text
                chatbot.current_language = target_lang
                st.session_state.current_language = target_lang

                translated_file_name = f"translated_{selected_file}.txt"
                st.download_button(
                    label="Download Translated Text",
                    data=translated_text,
                    file_name=translated_file_name,
                    mime="text/plain",
                    key="download_button"
                )

    st.subheader("Chat with your PDF")
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    if st.session_state.followup_questions:
        st.write("Follow-up Questions:")
        cols = st.columns(len(st.session_state.followup_questions))
        for i, (col, question) in enumerate(zip(cols, st.session_state.followup_questions)):
            with col:
                if st.button(f"{i+1}. {question}", key=f"followup_{i}"):
                    query = question
                    break
        else:
            query = st.chat_input("You:", key="chat_input")
    else:
        query = st.chat_input("You:", key="chat_input")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.chat_history.append(("user", query))

        context = chatbot.retrieve_chunks(query)
        answer, _, _, followup_questions = chatbot.handle_query(query, st.session_state.current_language)
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))
        st.session_state.latest_response = answer
        st.session_state.followup_questions = followup_questions

        st.experimental_rerun()

if __name__ == "__main__":
    main()






