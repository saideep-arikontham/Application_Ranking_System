import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import nltk
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


st.set_page_config(page_title="Resume Evaluator & Ranking System", layout="wide")


@st.cache_resource(show_spinner=False)  # Caches the models so they don't reload on rerun
def load_models():
    """Loads the necessary models and resources."""
    print("", end="")
    model_id = "saideep-arikontham/roberta-resume-fit-predictor"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    base_model_name = "roberta-base"
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_id)
    sentence_model = SentenceTransformer("all-mpnet-base-v2")

    # Load NLTK stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(["overqualified", "underqualified", "mismatch", "good"])

    return model, tokenizer, sentence_model, stop_words

with st.spinner("Loading Resume Evaluator..."):
    # Load models once and cache
    model, tokenizer, sentence_model, stop_words = load_models()


# Global variables for chunking
job_chunk_size = 256
resume_chunk_size = 256
max_length = job_chunk_size + resume_chunk_size + 2



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

def predict_with_voting(job_text, resume_text, model, tokenizer, device="cpu", return_probabilities=True):
    """Predicts the class using chunking and a voting mechanism."""
    model.to(device)
    model.eval()

    with torch.no_grad():
        job_text = preprocess_text(job_text)
        resume_text = preprocess_text(resume_text)

        # Tokenize job data
        job_tokens = tokenizer.tokenize(job_text)
        # Tokenize resume data
        resume_tokens = tokenizer.tokenize(resume_text)

        predictions = []
        probabilities = []

        # Chunk job data
        for i in range(0, len(job_tokens), job_chunk_size):
            job_chunk = job_tokens[i:i + job_chunk_size]

            # Chunk resume data
            for j in range(0, len(resume_tokens), resume_chunk_size):
                resume_chunk = resume_tokens[j:j + resume_chunk_size]

                # Combine the chunks and truncate
                combined_tokens = ['[CLS]'] + job_chunk + ['[SEP]'] + resume_chunk
                combined_tokens = combined_tokens[:max_length]

                # Convert to input IDs and attention mask
                input_ids = tokenizer.convert_tokens_to_ids(combined_tokens)
                attention_mask = [1] * len(input_ids)  # 1 for real tokens, 0 for padding

                # Pad to max_length
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

                # Convert to tensors and send to device
                input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                probs = torch.softmax(logits, dim=-1)

                predicted_class = torch.argmax(logits, dim=-1).item()
                predictions.append(predicted_class)
                probabilities.append(probs[0, 1].item())  # Probability of class "1"

        # Hard Voting
        if predictions:
            counts = Counter(predictions)
            most_voted_class = counts.most_common(1)[0][0]
        else:
            most_voted_class = 0  # Default if no predictions

        # Average Probability for Class "1"
        average_probability = np.mean(probabilities) if probabilities else 0.0

    return most_voted_class, average_probability

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
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📄 Resume Evaluator & Ranking System</h1>", unsafe_allow_html=True)

# Job Description input in a container
st.subheader("📝 Job Description")
job_container = st.container(border=True)
with job_container:
    job_description = st.text_area("Enter the job description:", "Looking for a data scientist with NLP experience.", height=200)

# User choice: Evaluate one resume or Rank multiple resumes
st.subheader("🔍 Choose an Option")


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
    st.subheader("📄 Upload Your Resume")
    resume_container = st.container(border=True)
    with resume_container:
        resume_text = st.text_area("Paste your resume content here:", "I have experience in NLP and Machine Learning.", height=300)

    if st.button("Evaluate Resume", use_container_width=True):
        status_container = st.empty()  # Create status container
        status_container.info("🔄 Evaluating resume... Please wait.")

        with st.status("Processing the data...", expanded=True):
            st.write("Getting predictions...")
            time.sleep(2)

            st.write("Computing metrics")

            job_cleaned = preprocess_text(job_description)
            resume_cleaned = preprocess_text(resume_text)

            # Compute similarity scores
            cosine_sim = compute_cosine_similarity(job_cleaned, resume_cleaned)
            jaccard_sim = compute_jaccard_similarity(job_cleaned, resume_cleaned)
            bertscore = compute_bertscore(resume_cleaned, job_cleaned)
            st_cosine_sim = compare_job_resume(resume_cleaned, job_cleaned)

            # Store results
            similarity_scores = {
                "TF-IDF Cosine Similarity": cosine_sim,
                "Jaccard Similarity": jaccard_sim,
                "BERTScore (Semantic)": bertscore,
                "Embedding Cosine Similarity": st_cosine_sim
            }

        # Remove status message
        status_container.success("✅ Resume evaluated successfully!")

        # Display results in tabs
        st.subheader("📊 Evaluation Results")
        
        # Create DataFrame for displaying results
        score_df = pd.DataFrame(list(similarity_scores.items()), columns=["Metric", "Score"])
        
        # Create tabs for different result views
        tab1, tab2 = st.tabs(["📋 Table View", "📊 Chart View"])
        
        with tab1:
            st.subheader("Similarity Scores Table")
            # Format scores to 4 decimal places for better readability
            formatted_df = score_df.copy()
            formatted_df["Score"] = formatted_df["Score"].apply(lambda x: f"{x:.4f}")
            st.table(formatted_df)

            # Additional explanation
            st.info("""
            **Higher scores indicate better alignment between the resume and job description:**
            - **TF-IDF Cosine Similarity**: Measures keyword overlap
            - **Jaccard Similarity**: Measures word-level overlap
            - **BERTScore**: Measures semantic similarity
            - **Embedding Cosine Similarity**: Measures context-aware similarity
            """)

        with tab2:
            st.subheader("Similarity Scores Chart")
            chart_df = score_df.set_index("Metric")
            st.bar_chart(chart_df, horizontal=True, height=300)
            
            # Additional explanation
            st.info("""
            **Higher scores indicate better alignment between the resume and job description:**
            - **TF-IDF Cosine Similarity**: Measures keyword overlap
            - **Jaccard Similarity**: Measures word-level overlap
            - **BERTScore**: Measures semantic similarity
            - **Embedding Cosine Similarity**: Measures context-aware similarity
            """)

# Rank Multiple Resumes
elif option == "Rank Multiple Resumes":
    st.subheader("📂 Upload Multiple Resumes")
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

    if st.button("Rank Resumes", use_container_width=True):
        # Reset the session state for a new ranking
        st.session_state.all_resume_scores = {}
        st.session_state.ranking_scores = []
        
        status_container = st.empty()  # Create status container
        status_container.info("🔄 Ranking resumes... Please wait.")
        with st.status("Processing the data...", expanded=True):
            st.write("Getting predictions...")
            time.sleep(2)

            st.write("Computing metrics...")
            for idx, resume in enumerate(resumes):
                if not resume.strip():  # Skip empty resumes
                    continue
                    
                job_cleaned = preprocess_text(job_description)
                resume_cleaned = preprocess_text(resume)

                # Compute similarity scores
                cosine_sim = compute_cosine_similarity(job_cleaned, resume_cleaned)
                jaccard_sim = compute_jaccard_similarity(job_cleaned, resume_cleaned)
                bertscore = compute_bertscore(resume_cleaned, job_cleaned)
                st_cosine_sim = compare_job_resume(resume_cleaned, job_cleaned)

                # Store individual scores for this resume
                resume_name = f"Resume {idx+1}"
                st.session_state.all_resume_scores[resume_name] = {
                    "TF-IDF Cosine Similarity": cosine_sim,
                    "Jaccard Similarity": jaccard_sim,
                    "BERTScore (Semantic)": bertscore,
                    "Embedding Cosine Similarity": st_cosine_sim
                }

                # Compute Mean Score
                mean_score = round(np.mean([cosine_sim, jaccard_sim, bertscore, st_cosine_sim]), 4)
                st.session_state.ranking_scores.append((resume_name, mean_score))
            
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
        st.subheader("🏆 Resume Rankings")

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
            st.subheader("Ranking Table")
            display_df = rank_df.reset_index()
            display_df["Mean Similarity Score"] = display_df["Mean Similarity Score"].apply(lambda x: f"{x:.4f}")
            st.table(display_df)

        elif st.session_state.active_rank_tab == 1:  # Chart View
            st.subheader("Ranking Chart")
            st.bar_chart(rank_df, horizontal=True)

            st.info("""
            **The chart shows the overall match score for each resume:**
            - Higher scores indicate better matches with the job description
            - Scores are calculated as the average of multiple similarity metrics
            """)

        elif st.session_state.active_rank_tab == 2:  # Individual Resume Scores
            st.subheader("Individual Resume Scores")

            if st.session_state.all_resume_scores:
                resume_names = list(st.session_state.all_resume_scores.keys())

                selected_resume = st.selectbox(
                    "Select a resume to view detailed scores:",
                    resume_names,
                    key="resume_selector",
                    on_change=update_selected_resume,
                    index=None  # Default to first resume
                )

                # Use st.session_state.selected_resume to access the selected resume
                if st.session_state.selected_resume and st.session_state.selected_resume in st.session_state.all_resume_scores:
                    st.write(f"### Detailed Scores for {st.session_state.selected_resume}")

                    selected_scores = st.session_state.all_resume_scores[st.session_state.selected_resume]
                    score_df = pd.DataFrame(list(selected_scores.items()), columns=["Metric", "Score"])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Score Table")
                        formatted_df = score_df.copy()
                        formatted_df["Score"] = formatted_df["Score"].apply(lambda x: f"{x:.4f}")
                        st.table(formatted_df)

                    with col2:
                        st.subheader("Score Chart")
                        chart_df = score_df.set_index("Metric")
                        st.bar_chart(chart_df, height=300, horizontal=True)

                else:
                    st.warning("Please select a resume to view its detailed scores.")
            else:
                st.warning("No resume data available. Please process resumes first.")