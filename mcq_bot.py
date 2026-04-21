import streamlit as st
from google import genai
import PyPDF2
import os
import random
import time

# --- 1. Configuration ---
GOOGLE_API_KEY = "AIzaSyBma2f1EOQHMHZGPBnjUYGWVBHpapGB3H0" 
client = genai.Client(api_key=GOOGLE_API_KEY)
KNOWLEDGE_FOLDER = "knowledge_base"

SYSTEM_INSTRUCTION = """
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

st.set_page_config(page_title="Railway MCQ Bot", page_icon="📝")
st.title("📝 Railway MCQ Generator (mcq_bot)")

# --- 2. Logic: Indexing the PDF ---
@st.cache_data
def index_local_pdf(filepath):
    chunks = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                # Only keep pages with enough content to make a question
                if text and len(text.strip()) > 150:
                    chunks.append(text)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return chunks

# --- 3. UI: Settings ---
available_pdfs = []
if os.path.exists(KNOWLEDGE_FOLDER):
    available_pdfs = sorted([f for f in os.listdir(KNOWLEDGE_FOLDER) if f.endswith('.pdf')])

with st.sidebar:
    st.header("Exam Settings")
    
    if available_pdfs:
        selected_file = st.selectbox("Select Training Manual", available_pdfs)
    else:
        st.error(f"No PDFs found in the '{KNOWLEDGE_FOLDER}' folder!")
        selected_file = None
        
    num_questions = st.number_input("Total Questions to Generate", min_value=1, max_value=200, value=10)
    difficulty = st.selectbox("Set Difficulty Level", ["Easy", "Medium", "Hard"])
    generate_btn = st.button("Generate MCQs")

# --- 4. Generation Logic ---
if generate_btn and selected_file:
    filepath = os.path.join(KNOWLEDGE_FOLDER, selected_file)
    pages = index_local_pdf(filepath)
    
    if not pages:
        st.error("This PDF seems to be empty or contains only images (no selectable text).")
    else:
        st.info(f"📖 Reading '{selected_file}'. Preparing your {difficulty} level exam...")
        
        questions_generated = 0
        full_exam_paper = f"EXAM PAPER: {selected_file.replace('.pdf', '')}\nDIFFICULTY: {difficulty}\n" + ("="*30) + "\n\n"
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Batching logic: AI works best generating 3-5 questions at a time
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
                    config={'system_instruction': SYSTEM_INSTRUCTION},
                    contents=prompt
                )
                
                batch_text = response.text
                full_exam_paper += batch_text + "\n\n"
                questions_generated += batch_size
                
                # Update UI
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
        
        # Displaying the result
        st.text_area("Preview", value=full_exam_paper, height=400)
        
        # Download Link
        st.download_button(
            label="📩 Download Exam as Text File",
            data=full_exam_paper,
            file_name=f"MCQ_{selected_file.replace('.pdf', '')}_{difficulty}.txt",
            mime="text/plain"
        )