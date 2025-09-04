import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import re
import string
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import evaluate

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from bert_score import score

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Resume Evaluator & Ranking System", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        padding: 20px;
    }
    .stApp {
        background-color: #000000;
        margin: 0 auto;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 800;
    }
    h2 {
        color: #2563EB;
        font-weight: 600;
    }
    h3 {
        color: #3B82F6;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .stTextArea>div>div>textarea {
        border-radius: 6px;
        border: 1px solid #E5E7EB;
    }
    div[data-testid="stRadio"] > div {
        gap: 1rem;
    }
    div[data-testid="stRadio"] label {
        background-color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #000000;
    }
    div[data-testid="stContainer"] {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stTable"] {
        background-color: black;
    }
    .e16nr0p31 {
        border-radius: 8px;
    }
    div.stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    div.stTabs [data-baseweb="tab"] {
        background-color: #000000;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
    }
    div.stTabs [aria-selected="true"] {
        background-color: #000000;
        color: white;
    }
    .stStatus {
        border-radius: 6px;
    }
    div[data-testid="stInfo"] {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)  # Caches the models so they don't reload on rerun
def load_models():
    """Loads the necessary models and resources."""
    print("", end="")
    model_id = "saideep-arikontham/bigbird-resume-fit-predictor_v1v1"
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
    base_model_name = "google/bigbird-roberta-base"
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_id)
    sentence_model = SentenceTransformer("all-mpnet-base-v2")

    # Load NLTK stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(["overqualified", "underqualified", "mismatch", "good"])

    return model, tokenizer, sentence_model, stop_words

with st.spinner("Loading Resume Evaluator..."):
    # Load models once and cache
    model, tokenizer, sentence_model, stop_words = load_models()


# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(["overqualified", "underqualified", "mismatch", "good"])

def preprocess_text(text):
    """Preprocess text by removing unwanted symbols, normalizing, and removing stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s%$/.-]", "", text)
    text = re.sub(r"-(?!\d)", "", text)  # Preserve hyphens only when followed by a number
    text = re.sub(r"(?<!\d)/|/(?!\d)", " ", text)  # Preserve GPA-like formats (e.g., 3.8/4.0)
    text = re.sub(r"\b(\w+)\.(?!\d)", r"\1", text)  # Remove periods unless in numbers
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("show less", "").replace("show more", "")
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def tokenize_data(job_description, resume, tokenizer):
    # Preprocess input texts
    job_description = preprocess_text(job_description)
    resume = preprocess_text(resume)

    # Define the tokenizer settings
    # Define chunk sizes
    job_max_length = 2046
    resume_max_length = 2046
    max_model_length = 4096

    # Tokenize job description
    job_inputs = tokenizer(
        job_description,
        truncation=True,
        max_length=job_max_length,
        padding="max_length",
        return_tensors="pt"
    )

    # Tokenize resume
    resume_inputs = tokenizer(
        resume,
        truncation=True,
        max_length=resume_max_length,
        padding="max_length",
        return_tensors="pt"
    )

    # Get separator token ID
    separator_id = tokenizer.sep_token_id
    if separator_id is None:
        separator_id = tokenizer.eos_token_id

    # Convert separator ID to correct dtype
    separator_tensor = torch.tensor([[separator_id]], dtype=job_inputs["input_ids"].dtype)

    # Combine tokens with separator
    combined_ids = torch.cat((job_inputs["input_ids"], separator_tensor, resume_inputs["input_ids"]), dim=1)
    combined_mask = torch.cat((job_inputs["attention_mask"], torch.tensor([[1]], dtype=job_inputs["attention_mask"].dtype), resume_inputs["attention_mask"]), dim=1)


    # Ensure we don't exceed the max length
    combined_ids = combined_ids[:, :max_model_length]
    combined_mask = combined_mask[:, :max_model_length]

    return {
        "input_ids": combined_ids,
        "attention_mask": combined_mask
    }


def predict_resume_fit(job_description, resume, model, tokenizer):
    # Tokenize input
    inputs = tokenize_data(job_description, resume, tokenizer)

    # Ensure model is in evaluation mode
    model.eval()

    # Move to GPU if available
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits
    logits = outputs.logits

    # Compute softmax probabilities
    probs = F.softmax(logits, dim=-1)

    # Get predicted class
    predicted_class = torch.argmax(probs, dim=-1).item()

    # Get probability of class 1
    class_1_prob = probs[:, 1].item() if probs.shape[1] > 1 else probs.item()

    return predicted_class, class_1_prob


def chunk_text(text, max_length=510, overlap=50):
    """Splits long text into overlapping chunks to fit the model's context limit."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i : i + max_length])
        chunks.append(chunk)
    return chunks

def get_text_embedding(text):
    """Generates an aggregated embedding for long text using chunking and mean pooling."""
    chunks = chunk_text(text)
    chunk_embeddings = sentence_model.encode(chunks)  # Get embeddings for each chunk
    
    if len(chunk_embeddings) == 0:
        return np.zeros(sentence_model.get_sentence_embedding_dimension())  # Return zero vector if no embeddings
    
    # Mean pooling to aggregate chunk embeddings into a single vector
    final_embedding = np.mean(chunk_embeddings, axis=0)  
    return final_embedding

# Similarity Metrics
def compute_bertscore(candidate, reference):
    """Compute BERTScore (Semantic Similarity)"""
    P, R, F1 = score([candidate], [reference], lang="en", model_type="roberta-base")
    return F1.item()  # Use F1 score for evaluation

def compute_cosine_similarity(text1, text2):
    """Compute Cosine Similarity (Lexical Similarity using TF-IDF)"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

def compute_jaccard_similarity(text1, text2):
    """Compute Jaccard Similarity (Word Overlap Measure)"""
    words1 = set(word_tokenize(text1.lower())) - stop_words
    words2 = set(word_tokenize(text2.lower())) - stop_words
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

def compare_job_resume(resume_text, job_text):
    """Computes similarity score between a job description and a resume using sentence embeddings."""
    job_embedding = get_text_embedding(job_text)
    resume_embedding = get_text_embedding(resume_text)

    # Compute cosine similarity
    similarity_score = cosine_similarity([job_embedding], [resume_embedding])[0][0]
    return similarity_score


# Streamlit App Layout
st.markdown("<h1 style='text-align: center; margin-bottom: 30px; color:#a2c1f5;'>Resume Evaluator & Ranking System</h1>", unsafe_allow_html=True)

# App description
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown(""" """, unsafe_allow_html=True)

# Job Description input in a container
st.markdown("<h2 style='margin-top: 20px;'>📝 Job Description</h2>", unsafe_allow_html=True)
job_container = st.container(border=True)
with job_container:
    job_description = st.text_area("Enter the job description:", "Looking for a data scientist with NLP experience.", height=200)

# User choice: Evaluate one resume or Rank multiple resumes
st.markdown("<h2 style='margin-top: 30px;'>🔍 Choose an Option</h2>", unsafe_allow_html=True)


with st.spinner("Switching options..."):  # Spinner while changing options
    option = st.radio("Select one:", ["Evaluate a Resume", "Rank Multiple Resumes"], horizontal=True)

# Initialize session state to store resume scores
if 'all_resume_scores' not in st.session_state:
    st.session_state.all_resume_scores = {}
if 'ranking_scores' not in st.session_state:
    st.session_state.ranking_scores = []
if 'resumes_processed' not in st.session_state:
    st.session_state.resumes_processed = False

# Evaluate Resume
if option == "Evaluate a Resume":
    st.markdown("<h2 style='margin-top: 30px;'>📄 Upload Your Resume</h2>", unsafe_allow_html=True)
    resume_container = st.container(border=True)
    with resume_container:
        resume_text = st.text_area("Paste your resume content here:", "I have experience in NLP and Machine Learning.", height=300)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        evaluate_button = st.button("Evaluate Resume", use_container_width=True)
    
    if evaluate_button:
        status_container = st.empty()  # Create status container
        status_container.info("🔄 Evaluating resume... Please wait.")

        with st.status("Processing the data...", expanded=True):
            
            job_cleaned = preprocess_text(job_description)
            resume_cleaned = preprocess_text(resume_text)

            # Compute similarity scores
            st.write("Getting predictions...")
            predicted_fit_class, predicted_fit_score = predict_resume_fit(job_cleaned, resume_cleaned, model, tokenizer)

            st.write("Computing metrics")
            cosine_sim = compute_cosine_similarity(job_cleaned, resume_cleaned)
            jaccard_sim = compute_jaccard_similarity(job_cleaned, resume_cleaned)
            # bertscore = compute_bertscore(resume_cleaned, job_cleaned)
            st_cosine_sim = compare_job_resume(resume_cleaned, job_cleaned)

            # Store results
            metric_scores = {
                "Fit Score" : predicted_fit_score,
                "TF-IDF Cosine Similarity": cosine_sim,
                "Jaccard Similarity": jaccard_sim,
                # "BERTScore (Semantic)": bertscore,
                "Embedding Cosine Similarity": st_cosine_sim
            }

        # Remove status message
        status_container.success("✅ Resume evaluated successfully!")

        # Display results in tabs
        st.markdown("<h2 style='margin-top: 30px;'>📊 Evaluation Results</h2>", unsafe_allow_html=True)
        
        # Create DataFrame for displaying results
        score_df = pd.DataFrame(list(metric_scores.items()), columns=["Metric", "Score"])
        
        # Create tabs for different result views
        tab1, tab2 = st.tabs(["📋 Table View", "📊 Chart View"])
        
        with tab1:
            st.markdown("<h3 style='margin-top: 20px;'>Similarity Scores Table</h3>", unsafe_allow_html=True)
            # Format scores to 4 decimal places for better readability
            formatted_df = score_df.copy()
            formatted_df["Score"] = formatted_df["Score"].apply(lambda x: f"{x:.4f}")
            st.table(formatted_df)

            # Additional explanation
            st.info("""
            **Higher scores indicate better alignment between the resume and job description:**
            - **TF-IDF Cosine Similarity**: Measures keyword overlap
            - **Jaccard Similarity**: Measures word-level overlap
            - **Embedding Cosine Similarity**: Measures context-aware similarity
            """)

        with tab2:
            st.markdown("<h3 style='margin-top: 20px;'>Similarity Scores Chart</h3>", unsafe_allow_html=True)

            # Use Streamlit's built-in bar chart
            chart_df = score_df.set_index("Metric")
            st.bar_chart(chart_df["Score"], use_container_width=True, horizontal=True)

            # If you want to add a note about the scores
            st.caption("Resume-Job Similarity Scores (higher is better)")
            
            # Additional explanation
            st.info("""
            **Higher scores indicate better alignment between the resume and job description:**
            - **TF-IDF Cosine Similarity**: Measures keyword overlap
            - **Jaccard Similarity**: Measures word-level overlap
            - **Embedding Cosine Similarity**: Measures context-aware similarity
            """)

# Rank Multiple Resumes
elif option == "Rank Multiple Resumes":
    st.markdown("<h2 style='margin-top: 30px;'>📂 Upload Multiple Resumes</h2>", unsafe_allow_html=True)
    num_resumes = st.number_input("Enter number of resumes to rank:", min_value=1, max_value=10, value=3)

    resumes = []
        # Initialize session state (if needed)
    if 'active_rank_tab' not in st.session_state:
        st.session_state['active_rank_tab'] = 0
    if 'selected_resume' not in st.session_state:
        st.session_state['selected_resume'] = None
    
    resume_container = st.container(border=True)
    with resume_container:
        for i in range(num_resumes):
            resumes.append(st.text_area(f"Paste Resume {i+1}:", "", height=200))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        rank_button = st.button("Rank Resumes", use_container_width=True)
    
    if rank_button:
        # Reset the session state for a new ranking
        st.session_state.all_resume_scores = {}
        st.session_state.ranking_scores = []
        
        status_container = st.empty()  # Create status container
        status_container.info("🔄 Ranking resumes... Please wait.")
        with st.status("Processing the data...", expanded=True):
            

            for idx, resume in enumerate(resumes):
                resume_name = f"Resume {idx+1}"
                st.write(f"Evaluating {resume_name}...")

                if not resume.strip():  # Skip empty resumes
                    continue
                    
                job_cleaned = preprocess_text(job_description)
                resume_cleaned = preprocess_text(resume)

                # Compute similarity scores
                fit_class, fit_score = predict_resume_fit(job_cleaned, resume_cleaned, model, tokenizer)
                cosine_sim = compute_cosine_similarity(job_cleaned, resume_cleaned)
                jaccard_sim = compute_jaccard_similarity(job_cleaned, resume_cleaned)
                # bertscore = compute_bertscore(resume_cleaned, job_cleaned)
                st_cosine_sim = compare_job_resume(resume_cleaned, job_cleaned)

                # Store individual scores for this resume
                
                st.session_state.all_resume_scores[resume_name] = {
                    "Fit Score" : fit_score,
                    "TF-IDF Cosine Similarity": cosine_sim,
                    "Jaccard Similarity": jaccard_sim,
                    # "BERTScore (Semantic)": bertscore,
                    "Embedding Cosine Similarity": st_cosine_sim
                }

                # Compute Mean Score
                weighted_score = 0.35*fit_score + 0.15*cosine_sim + 0.15*jaccard_sim + 0.35*st_cosine_sim
                st.session_state.ranking_scores.append((resume_name, weighted_score))
            
            st.write("Ranking Resumes...")
            # Sort resumes based on mean similarity score
            st.session_state.ranking_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Mark as processed
            st.session_state.resumes_processed = True

        # Remove status message
        status_container.success("✅ Resumes ranked successfully!")

    def update_active_rank_tab():
        st.session_state.active_rank_tab = st.session_state.rank_tab_selector

    def update_selected_resume():
        st.session_state.selected_resume = st.session_state.resume_selector


    # Display ranking results if available
    if st.session_state.resumes_processed and st.session_state.ranking_scores:
        # Convert to DataFrame for display
        rank_df = pd.DataFrame(st.session_state.ranking_scores,
                                columns=["Resume", "Mean Similarity Score"]).set_index("Resume")

        # Display ranking results in tabs
        st.markdown("<h2 style='margin-top: 30px;'>🏆 Resume Rankings</h2>", unsafe_allow_html=True)

        # Create a radio button for tab selection
        tab_options = ["📋 Table View", "📊 Chart View", "🔍 Individual Resume Scores"]

        # Use `index` to retain the last selected tab
        selected_tab_index = st.radio(
            "Select View:",
            options=range(len(tab_options)),
            format_func=lambda i: tab_options[i],
            index=st.session_state.get('active_rank_tab', 0),  # Default to 0 if not set
            horizontal=True,
            key="rank_tab_selector",
            on_change=update_active_rank_tab  # Call the callback
        )


        # Display content based on selected tab index
        if st.session_state.active_rank_tab == 0:  # Table View
            st.markdown("<h3>Ranking Table</h3>", unsafe_allow_html=True)
            display_df = rank_df.reset_index()
            
            # Add rankings
            display_df.insert(0, "Rank", range(1, len(display_df) + 1))
            
            # Format score
            display_df["Mean Similarity Score"] = display_df["Mean Similarity Score"].apply(lambda x: f"{x:.4f}")
            display_df = display_df.set_index("Rank")
            
            # Style the table
            st.markdown("""
            <style>
            .dataframe {
                font-size: 16px !important;
                text-align: center !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.table(display_df)

        elif st.session_state.active_rank_tab == 1:  # Chart View
            st.markdown("<h3>Ranking Chart</h3>", unsafe_allow_html=True)
            
            # Replace matplotlib with Streamlit bar chart
            st.bar_chart(rank_df, use_container_width=True, horizontal=True)
            
            st.info("""
            **The chart shows the overall match score for each resume:**
            - Higher scores indicate better matches with the job description
            - Scores are calculated as the average of multiple similarity metrics
            """)

        elif st.session_state.active_rank_tab == 2:  # Individual Resume Scores
            st.markdown("<h3>Individual Resume Scores</h3>", unsafe_allow_html=True)

            if st.session_state.all_resume_scores:
                resume_names = list(st.session_state.all_resume_scores.keys())

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    selected_resume = st.selectbox(
                        "Select a resume to view detailed scores:",
                        resume_names,
                        key="resume_selector",
                        on_change=update_selected_resume,
                        index=None  # Default to first resume
                    )

                # Use st.session_state.selected_resume to access the selected resume
                if st.session_state.selected_resume and st.session_state.selected_resume in st.session_state.all_resume_scores:
                    st.markdown(f"""
                    <div style='background-color: #424040; padding: 2px; border-radius: 8px; margin: 20px 0;'>
                        <h3 style='text-align: center; margin: 0; color: #ffffff;'>
                            Detailed Scores for {st.session_state.selected_resume}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                    selected_scores = st.session_state.all_resume_scores[st.session_state.selected_resume]
                    score_df = pd.DataFrame(list(selected_scores.items()), columns=["Metric", "Score"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("<h4>Score Table</h4>", unsafe_allow_html=True)
                        formatted_df = score_df.copy()
                        formatted_df["Score"] = formatted_df["Score"].apply(lambda x: f"{x:.4f}")
                        st.table(formatted_df)

                    with col2:
                        st.markdown("<h4>Score Chart</h4>", unsafe_allow_html=True)
                        
                        # Replace matplotlib with Streamlit bar chart
                        chart_df = score_df.set_index("Metric")
                        st.bar_chart(chart_df, use_container_width=True, horizontal=True)

                else:
                    st.warning("Please select a resume to view its detailed scores.")
            else:
                st.warning("No resume data available. Please process resumes first.")

# Footer
st.markdown("""
<div style='margin-top: 50px; text-align: center;'>
    <hr>
    <p style='color: #6B7280; font-size: 14px;'>
        Resume Evaluator & Ranking System • Powered by NLP and Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)