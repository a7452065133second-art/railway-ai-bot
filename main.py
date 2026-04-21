import streamlit as st
from google import genai
import PyPDF2
import os
import tempfile
import time
import random

# --- 1. Configuration & Page Setup ---
st.set_page_config(page_title="Railway Ultimate AI", page_icon="🚂", layout="wide")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
client = genai.Client(api_key=GOOGLE_API_KEY)
KNOWLEDGE_FOLDER = "knowledge_base"

# --- 2. System Instructions (Two Different Brains) ---
CHAT_INSTRUCTION = """
ROLE: Senior Indian Railways Track Machine Engineer.

STRICT DATA POLICY:
1. TASK 1 (Manuals): Use the provided "MANUAL EXCERPTS" to answer technical questions.
2. TASK 2 (Uploaded Circuits): Analyze the "UPLOADED CIRCUIT PDF" only if the user asks about it.
3. MEMORY: Use the "RECENT CHAT HISTORY" to understand follow-up questions (like "Explain it step by step").
4. THE "I DON'T KNOW" RULE: If information is not in the manuals, circuit, or history, reply: "I'm sorry, but that information is not available in the current technical manuals."
5. BREVITY: Be extremely concise unless a "step-by-step" or "detailed" explanation is requested.
"""

MCQ_INSTRUCTION = """
You are an expert Indian Railways Technical Examiner. Your task is to generate high-quality Multiple Choice Questions (MCQs) based ONLY on the provided manual text.

RULES:
1. Provide exactly 4 options (A, B, C, D) for each question.
2. Mark the correct answer clearly at the end of each question.
3. Difficulty Levels:
   - 'Easy': Focus on names of parts, basic functions, and general definitions.
   - 'Medium': Focus on technical values, specific clearances, pressures, and sequences.
   - 'Hard': Focus on complex troubleshooting, maintenance logic, and multi-step engineering procedures.
4. Formatting:
   Question [Number]: [Question Text]
   A) [Option]
   B) [Option]
   C) [Option]
   D) [Option]
   Correct Answer: [Letter]
"""

# --- 3. Data Processing Functions ---
@st.cache_data
def load_and_index_pdfs_for_chat(folder):
    all_chunks = []
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
        for filename in files:
            path = os.path.join(folder, filename)
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            all_chunks.append({
                                "source": filename,
                                "page": i + 1,
                                "content": text
                            })
            except Exception: pass
    return all_chunks

@st.cache_data
def index_local_pdf_for_mcq(filepath):
    chunks = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 150:
                    chunks.append(text)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return chunks

# --- 4. Sidebar Navigation ---
st.sidebar.title("🚂 AI Navigation")
app_mode = st.sidebar.radio("Choose Application Mode:", ["💬 Chat & Analysis", "📝 MCQ Generator"])
st.sidebar.markdown("---")

# ==========================================
#        MODE 1: CHAT & ANALYSIS
# ==========================================
if app_mode == "💬 Chat & Analysis":
    st.title("🚂 Railway Track Machine Chatbot")
    
    # 1. State Management for Chat
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "current_upload_ref" not in st.session_state:
        st.session_state.current_upload_ref = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Load Chat Knowledge Base
    if not st.session_state.chunks:
        with st.spinner("Indexing Manuals..."):
            st.session_state.chunks = load_and_index_pdfs_for_chat(KNOWLEDGE_FOLDER)

    # 3. Sidebar Circuit Upload (Only shows in Chat mode)
    st.sidebar.header("Circuit Analysis (Optional)")
    uploaded_file = st.sidebar.file_uploader("Upload Circuit PDF", type=['pdf'])

    if uploaded_file and (not hasattr(st.session_state, 'last_f') or st.session_state.last_f != uploaded_file.name):
        with st.spinner("Uploading Diagram..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                st.session_state.current_upload_ref = client.files.upload(file=tmp_path)
                st.session_state.last_f = uploaded_file.name
                st.sidebar.success("✅ Diagram Ready")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

    # 4. Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about manuals or uploaded circuits..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            # Search
            query_keywords = prompt.lower().split()
            hits = []
            for item in st.session_state.chunks:
                score = sum(1 for word in query_keywords if word in item["content"].lower())
                if score > 0: hits.append(item)
            
            hits = sorted(hits, key=lambda x: sum(1 for w in query_keywords if w in x["content"].lower()), reverse=True)[:5]
            context_text = "\n\n---\n\n".join([f"FROM: {h['source']} (Pg {h['page']})\n{h['content']}" for h in hits])

            # Generation
            response_text = ""
            for attempt in range(3):
                try:
                    contents = []
                    # Memory
                    if len(st.session_state.messages) > 1:
                        history_str = "--- RECENT CHAT HISTORY ---\n"
                        for m in st.session_state.messages[-5:-1]: 
                            role = "USER" if m["role"] == "user" else "ENGINEER"
                            history_str += f"{role}: {m['content']}\n"
                        contents.append(history_str)

                    if context_text:
                        contents.append(f"RELEVANT MANUAL EXCERPTS:\n{context_text}")
                    if st.session_state.current_upload_ref:
                        contents.append(st.session_state.current_upload_ref)
                    contents.append(f"CURRENT USER QUESTION: {prompt}")

                    response = client.models.generate_content(
                        model="gemini-flash-lite-latest",
                        config={'system_instruction': CHAT_INSTRUCTION},
                        contents=contents
                    )
                    response_text = response.text
                    break
                except Exception as e:
                    if "503" in str(e) or "429" in str(e):
                        time.sleep(5) 
                    else:
                        st.error(f"Error: {e}")
                        break
            
            if response_text:
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

# ==========================================
#        MODE 2: MCQ GENERATOR
# ==========================================
elif app_mode == "📝 MCQ Generator":
    st.title("📝 Technical MCQ Generator")
    
    # Check for PDFs
    available_pdfs = []
    if os.path.exists(KNOWLEDGE_FOLDER):
        available_pdfs = sorted([f for f in os.listdir(KNOWLEDGE_FOLDER) if f.endswith('.pdf')])

    # Sidebar MCQ Settings (Only shows in MCQ mode)
    st.sidebar.header("Exam Settings")
    if available_pdfs:
        selected_file = st.sidebar.selectbox("Select Training Manual", available_pdfs)
    else:
        st.sidebar.error(f"No PDFs found in '{KNOWLEDGE_FOLDER}'!")
        selected_file = None
        
    num_questions = st.sidebar.number_input("Total Questions", min_value=1, max_value=200, value=10)
    difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
    generate_btn = st.sidebar.button("Generate MCQs")

    # MCQ Generation Logic
    if generate_btn and selected_file:
        filepath = os.path.join(KNOWLEDGE_FOLDER, selected_file)
        pages = index_local_pdf_for_mcq(filepath)
        
        if not pages:
            st.error("This PDF seems to be empty or contains only images.")
        else:
            st.info(f"📖 Reading '{selected_file}'. Preparing your {difficulty} level exam...")
            
            questions_generated = 0
            full_exam_paper = f"EXAM PAPER: {selected_file.replace('.pdf', '')}\nDIFFICULTY: {difficulty}\n" + ("="*30) + "\n\n"
            progress_bar = st.progress(0)
            status_text = st.empty()

            while questions_generated < num_questions:
                batch_size = min(5, num_questions - questions_generated)
                random_page_content = random.choice(pages)
                
                prompt = f"""
                Generate {batch_size} MCQs starting from Question Number {questions_generated + 1}.
                Difficulty: {difficulty}.
                
                Source Text:
                {random_page_content}
                """
                
                try:
                    response = client.models.generate_content(
                        model="gemini-flash-lite-latest",
                        config={'system_instruction': MCQ_INSTRUCTION},
                        contents=prompt
                    )
                    
                    batch_text = response.text
                    full_exam_paper += batch_text + "\n\n"
                    questions_generated += batch_size
                    
                    progress_bar.progress(questions_generated / num_questions)
                    status_text.text(f"Progress: {questions_generated}/{num_questions} questions created...")
                    
                except Exception as e:
                    if "503" in str(e) or "429" in str(e):
                        st.warning("Server busy, pausing for 5 seconds...")
                        time.sleep(5)
                    else:
                        st.error(f"Error during generation: {e}")
                        break

            st.success("✨ Your MCQ Exam Paper is ready!")
            st.text_area("Preview", value=full_exam_paper, height=400)
            st.download_button(
                label="📩 Download Exam as Text File",
                data=full_exam_paper,
                file_name=f"MCQ_{selected_file.replace('.pdf', '')}_{difficulty}.txt",
                mime="text/plain"
            )
