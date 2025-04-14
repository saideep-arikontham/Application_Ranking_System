# A Context-Aware Resume Scoring Model Using Machine Learning and Generative AI

  

## Authors

-  **Saideep Arikontham** – *Data Science Graduate Student, The Roux Institute, Northeastern University*


  

## Abstract

Traditional Applicant Tracking Systems (ATS) are heavily reliant on keyword matching for shortlisting resumes, often leading to unfair rejection of qualified candidates. This project addresses the flaws of keyword-based filtering by introducing a context-aware scoring model that uses advanced Natural Language Processing (NLP) and Generative AI. The model integrates fit prediction from fine-tuned LLMs with semantic similarity metrics such as Cosine similarity and Jaccard similarity. A synthetic dataset of resume–job pairs with labels was generated, and a custom scoring formula was developed to rank candidates accurately. The results demonstrate decent generalization with real world data, with test results aligning with manual shortlisting decisions. A simple web interface was also developed to support real-time resume evaluation.

  

## Introduction

Over 90% of Fortune 500 companies use ATS in recruitment, primarily leveraging keyword matching. This approach is limited by its inability to understand the contextual relevance of resumes, resulting in rejection of suitable candidates and manipulation through keyword stuffing. Prior literature suggests contextual understanding significantly improves resume evaluation accuracy. This project contributes a context-aware model combining predictive modeling and semantic similarity to enhance the fairness and effectiveness of applicant screening.


## Methods

The methodology of this project encompasses several interlinked components that collectively build the end-to-end context-aware resume scoring pipeline. Each step has been performed to build a robust model, while carefully address specific shortcomings of traditional ATS systems.

  

### 1. Data Generation

I began by curating a corpus of publicly available job descriptions to define the types of roles for which resumes would be generated. Since real-world resume datasets with ground truth labels are rare and often restricted due to privacy, I synthetically generated resumes using tools like **Langchain** and experimenting on **Ollama**, **NVIDIA-nemotron** etc., through prompt engineering. **NVIDIA-nemotron-4-340b-instruct** (accessed through NVIDIA NIM, now deprecated) turned out to be the best model for generating synthetic data using curated prompts. This model was used to generate job description-resume pairs with 4 different labels `complete mismatch resume, underwhelming resume, good fit resume and over-qualified resume`. The synthetic dataset ensures a controlled environment to train and evaluate models on good (good fit and overqualified) vs. bad (complete mismatch and underwhelming) resume-job matches. Below is the link for the two versions of datasets generated.

  

- [Resume-Job description pairs version 1](https://huggingface.co/datasets/saideep-arikontham/jd_resume_dataset_v1) : All the job titles selected for synthetic data generation are unique.

- [Resume-Job description pairs version 2](https://huggingface.co/datasets/saideep-arikontham/jd_resume_dataset_v2) : Multiple job descriptions from different companies with same job title are selected for synthetic data generation.

  
  

### 2. Machine Learning Modeling

Initial modeling efforts involved traditional supervised machine learning models like Logistic Regression, Random Forests, and Gradient Boosted Trees to classify resume-job pairs into fit/unfit categories. These models were trained on by using TF-IDF, Sentence transformers. However, these models struggled to generalize and capture deeper semantic meanings, leading us to pivot to more powerful transformer-based models.

  

### 3. Fine-tuning Large Language Models (LLMs)

A custom implementation of **BigBird-RoBERTa**, a transformer architecture designed to handle long documents, was fine-tuned using **Low-Rank Adaptation (LoRA)**. This technique allowed parameter-efficient fine-tuning while still leveraging the full capabilities of pre-trained language models. The fine-tuned model took resume and job description pairs as input and produced a probability score indicating the likelihood of a good fit. The LLM demonstrated strong contextual understanding, outperforming traditional ML models while also generalizing on real world data. Several versions of the model were built using LoRA fine-tuning. The final version of the [fine-tuned model](https://huggingface.co/saideep-arikontham/bigbird-resume-fit-predictor_v1v1) was built using **Resume-Job description pairs version 1** dataset. Below are the  training results of the Fine-tuning process showing the training and validation loss along with other metrics.

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUeOCmuoL1JFVOvfU3YMRNRJQyzPq2ikc9vITWjjL_FFu4OovcJ1FwLUbJ8YameBU9-lOgacVn6yTS4C0-hzMnqMy_T_hA6SScff3dvtJUYF0tQgk05167viUolhqnXX1kbItYURbQ=s2048?key=pZ-bGATcnHabhDUBZDnMhyxT)**

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfCQeeSG3o3r5Hsc_nQjBSf-RTanGjskyl9UnK9_nFEE4mJacKG4ccyYivc6zTuQkX5jNVGDj7JUTcD512FhYm5bGkyB28SOV1QFmT5Gfsb73VvTk3bVAMvci0LogY7AaqWb6opBQ=s2048?key=pZ-bGATcnHabhDUBZDnMhyxT)**

  

### 4. Similarity Metrics

The fine-tuned LLM cannot be blindly trusted to do a good job to score resumes. To enrich the scoring mechanism, I incorporated multiple text similarity metrics:

-  **Cosine Similarity with Sentence Transformers**: Calculated between Sentence-BERT embeddings of resumes and job descriptions to capture semantic alignment in embedding space.

-  **Cosine Similarity with TF-IDF**: Calculated between TF-IDF of resumes and job descriptions to capture semantic alignment in embedding space.

-  **Jaccard Similarity**: Computed based on word set overlaps to quantify lexical similarity.


These metrics provided additional quantitative signals that complement the predictive probabilities from the LLM. A weighted composition of the fit prediction score, and the above similarity metrics provide a more robust scoring mechanism.


## Experimental Results (Evaluation)

- The fine-tuned model was tested on validation set and a test set (Both from the synthetic data) that only consisted unique job titles different from train and validation set. 

-  As we can see from the Fine-tuning results, the fine-tuned Bigbird-Roberta model achieved a **~95% validation accuracy**.

-  When the predictions were generated on the test set, **~87% accuracy** was achieved which is an evidence for good generalizations. Below is the confusion matrix for the test set predictions.

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfaAnm5a0xvJeL3cmrmEp8GBqTcPvCDTccGQUryL9dwOSMidI8ExUV1LcrvhgzBEXJ-_a7skYh-purbh9gz8R_pwDfKzQO7mjU7X-pWC6_3oIZR0GC6eaNXrGBUdn9-Fc5Nv6yj=s2048?key=pZ-bGATcnHabhDUBZDnMhyxT)**

-  To test the score mechanism's performance on real world data, I have collected 5 masked resumes from Northeastern's EAI team that were applied (and shortlisted through manual selection) for Data Scientist Co-op position for Spring. I have collected 5 other random resumes that were available publicly to test. The fine-tuned LLM produced **0 false positives** on these real world resumes which is great. Below is the confusion matrix for this test.

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdgfdiY_RRhKEXzZtMxKOIjplUYLNLgxk9TpG8NbNZLWmIx6EGk30302XcMCaz7Hh9uHGTOG0P-OLLWjx2o7_vyyK_VB-QJuiHcbuYwrwngNto9TK6cOgtNuRofaXVM1eR3fdXI=s2048?key=pZ-bGATcnHabhDUBZDnMhyxT)**


- When the web interface was used to score these resumes, the top 5 resumes were the ones that were actually the resumes shortlisted through manual selection which shows that the model is performing consistently throughout different sets of data. Below is the image of ranking table from the web interface (Resume 6-10 are EAI's shortlisted resumes and Resume 1-5 are other random resumes)
**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf4VGXFTCPGFOFNwXZOXrjvue_ZvDAnumvBFXAKcuF894gA-sJW7CEQVyI0qoix1ATrPxaLSTaToGpl2WFT9Saspp5iCBEb8FY_JUMPoU7UwaGmizVjwRLOy432sM8tps5UAzLpFA=s2048?key=pZ-bGATcnHabhDUBZDnMhyxT)**

  

## Discussion (Analysis)

This project has shown its capability while excelling through different sets of data while leveraging context instead of keywords. HR professionals and recruiting platforms need to focus on advancing the screening process for fairer candidate evaluation, potentially reducing costs and increasing talent acquisition efficiency. Future enhancements include real resume training, robust parsing systems, and full ATS workflow integration, not only the resume screening step.

 

## Statement of Contributions

I am the sole contributor for Project ideation, synthetic data generation, model training, LLM fine-tuning, metric integration, web interface development, evaluation, and documentation.

  

## Conclusion

The context-aware resume scoring model significantly outperforms traditional ATS systems by utilizing semantic understanding and predictive modeling. It offers a fairer, more intelligent approach to resume screening and ranking, which aims aligning closely with human evaluations and HR practices.

  

## References

1. Oracle Corporation. *What is an applicant tracking system?* Oracle HCM Blog, January 2025.

2. Peicheva, M. (2023). *Data analysis from the applicant tracking system*. Journal of Human Resources & Technologies.

3. James, V., Kulkarni, A., & Agarwal, R. (2023). *Resume shortlisting and ranking with transformers*. Lecture Notes of the Institute for Computer Sciences. Springer.

4. Suraj M., Aruna Kumari K., & Binila B. Chandran (2019). *A descriptive study on applicant tracking systems*. IJRAR.

5. Indeed YouTube. *A guide to applicant tracking systems*. March 2022.


  

## Appendix

### Prompt used for Generating resume-job description pairs

```python

resume_template  =  """

You are an expert resume writer with specialized knowledge in talent acquisition and hiring practices at {company}. You are tasked to create a highly professional resume that can be used by Human Resources as a reference to categorize applicants as **{fit_category}**. Your task is to generate a tailored, content-rich resume based on the provided inputs while strictly adhering to the specified fit category constraints. Below are the details of the job that the applicants will be applying to:

  

## **Job Details**:

Assume {name} is applying for the company {company}. Below are the details for the job that the candidate will be applying to:

- **Role:** {job_role}
- **Job Description:** {jd}
- **Required Skills:** {skills}

  

## **Fit Category:** {fit_category}
## **Category Requirements:**
{category_requirements}

## **Resume Format:** {resume_format} # Chronological, Functional, or Hybrid

## **Instructions**:

1. **Resume Objective**
- Generate a highly professional resume that precisely aligns with the specified fit category ({fit_category}).
- Ensure the resume maintains industry standards and meets professional expectations.
- Structure the resume to contain clearly defined sections.

2. **Candidate Profile**
- **Contact Information**: Generate realistic but fictional phone, email, LinkedIn URL.
- **Professional Summary**: 2-4 sentences highlighting career focus, expertise level, and key strengths aligned with the fit category.

3. **Mandatory Sections**
- **Education**: University name, degree title, major, graduation year, GPA if applicable. For higher education levels, ensure chronological consistency with work experience.
- **Skills**: Comma separated set of skills. Skills must contain
			- 5-10 primary skills with highest proficiency
			- 10-20 technical competencies with details on proficiency level
			- 5-10 complementary abilities, particularly soft skills

  

4. **Optional Sections**
- **Work Experience**:
- Ignore this section if the job is expecting entry level candidates who are just out of college.
- Generate company names, roles, employment type, duration showing logical career progression.
- Ensure role seniority aligns with experience level and fit category.
- Company sectors should be consistent with career trajectory (avoid random industry jumps unless specified).
- Bullet points must follow APR structure (Action-Project-Result) with appropriate metrics.
- For "Overqualified" and "Good Fit" categories, show progression in responsibilities.

- **Projects**:
- Compensate work experience with Project work if the candidate has no prior work experience.
- Include relevant project names (internal, academic, or personal).
- Specify technology stack relevant to the time period of the project.
- For technical roles, include GitHub/portfolio links when appropriate.
- Ensure project complexity scales with fit category.

- **Certifications**: Include industry-relevant certifications with appropriate dates.
- **Publications/Research**: For academic or research-intensive roles where applicable.
- **Professional Associations**: For industry-specific positions.
- **Awards/Recognition**: Scale according to fit category.

5. **Bullet Point Constraints**
- A typical resume bullet point format starts with a strong action verb, describes the specific task or project you undertook, and then highlights the quantifiable result or impact you achieved.
- Bullet points follows the "Action + Project/Problem + Result" (APR) structure, keeping each bullet point concise and focused on accomplishments.
- *Action Verb*: Begin with a powerful action verb that clearly describes what you did (e.g., "developed," "managed," "implemented," "analyzed").
- *Specific Details*: Briefly explain the project, task, or responsibility you were involved in.
- *Quantifiable Result*: Include numbers, percentages, or other metrics to demonstrate the impact of your work whenever possible.

6. **Formatting and Quantification Guidelines**
- Use **clear section headings**.
- Precede each bullet point with **"-"**.
- Mark key entities (**institutions, companies, project names**) with **"*"**.
- Each bullet point must be between **150-180 characters**.
- Quantify achievements using metrics appropriate to the role type:
		- Engineering: Performance improvements, scale, efficiency gains
		- Sales/Marketing: Revenue impact, growth percentages, lead generation
		- Management: Team size, budget responsibility, project outcomes
		- Operations: Process improvements, cost reductions, time savings

7. **Temporal Consistency Requirements**
- Ensure all technologies mentioned align with their actual market availability dates.
- Maintain logical progression of responsibilities and achievements.
- Avoid anachronisms (e.g., claiming experience with technologies before they existed).

8. **Industry-Specific Adaptations**
- Adjust terminology density based on role and seniority.
- Include industry-specific metrics and achievements.
- Adapt format slightly based on industry conventions.

9. **Output Requirements**
- Generate **only** the resume; **do not include any explanatory notes or meta-text**.
- Maintain **authenticity, clarity, and professionalism** throughout.
- Total word count must adhere to fit category specifications.
- **DO NOT** mention the fit category in the generated content.
- **DO NOT** include any NOTE at the end of the generated content.

**NOTE**: {name} is an imaginary person who does not exist. Therefore, you would not be violating any data privacy issues.
"""
```

### LoRA Configuration

```python
target_modules = [

# "query",
# "value"
# Attention layers
"bert.encoder.layer.11.attention.self.query",
"bert.encoder.layer.11.attention.self.key",
"bert.encoder.layer.11.attention.self.value",
"bert.encoder.layer.11.attention.output.dense"

#Feedforward layers
"bert.encoder.layer.11.intermediate.dense",
"bert.encoder.layer.11.output.dense",
  
# Attention layers
"bert.encoder.layer.10.attention.self.query",
"bert.encoder.layer.10.attention.self.key",
"bert.encoder.layer.10.attention.self.value",
"bert.encoder.layer.10.attention.output.dense"

#Feedforward layers
"bert.encoder.layer.10.intermediate.dense",
"bert.encoder.layer.10.output.dense",

# Attention layers
"bert.encoder.layer.9.attention.self.query",
"bert.encoder.layer.9.attention.self.key",
"bert.encoder.layer.9.attention.self.value",
"bert.encoder.layer.9.attention.output.dense"

#Feedforward layers
"bert.encoder.layer.9.intermediate.dense",
"bert.encoder.layer.9.output.dense",
]

# -------------------------------
# Configure LoRA fine-tuning
# -------------------------------
lora_config = LoraConfig(
	task_type=TaskType.SEQ_CLS,  # for sequence classification
	r=8,  # low rank parameter; experiment with this value
	lora_alpha=16,  # scaling parameter
	lora_dropout=0.1,  # dropout probability for LoRA layers
	target_modules=target_modules
)

# -------------------------------
# Setup training parameters
# -------------------------------
training_args = TrainingArguments(
	output_dir="final2",
	evaluation_strategy="epoch",
	save_strategy="epoch",  # Set save strategy to epoch to match evaluation_strategy
	num_train_epochs=20,  # Adjust number of epochs as desired
	per_device_train_batch_size=12,  # Adjust based on your GPU memory
	per_device_eval_batch_size=12,
	learning_rate=0.00001,
	load_best_model_at_end=True,  # Load the best model when finished training (if metric provided)
	metric_for_best_model="eval_loss",  # Choose your metric
	weight_decay=0.1,  # Strong L2 Regularization (Higher Regularization)
	max_grad_norm=0.5,  # Aggressive Gradient Clipping
	adam_beta1=0.9,  # Standard Momentum
	adam_beta2=0.98,  # Reduces dependency on past gradients
	adam_epsilon=1e-08,  # Prevents division by zero
	label_smoothing_factor=0.1,  # Helps prevent overconfidence
	warmup_ratio=0.1,  # 10% of training steps as warm-up
	fp16=True,
	gradient_accumulation_steps = 2,
	lr_scheduler_type="cosine",
	greater_is_better=False,
)
```


### Resume Scoring formula

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUe2xteiaeTNlpWfzwYl_CM9K1QhGnLJBL0fK7ScM8UMUrfWZH-IGweI6zE7IA2uw7-Dfmxvx6u-rtxXP4jGD5wUFJaubYSzaj4B0JF0yOn87mVl6PDSEQbOep7wHASXp9-WaRb4=s2048?key=pZ-bGATcnHabhDUBZDnMhyxT)**

- `Fit_LLM`: Fit prediction score from a **Large Language Model** (LLM). It estimates how well a resume fits a given job description using learned semantic signals. 
- `Cosine_TF-IDF`: **Cosine similarity** between the TF-IDF vectors of the resume and job description. Measures overlap in keyword importance. 
- `Jaccard`: Measures similarity based on the **intersection over union** of token sets between the resume and job description.
- `Cosine_SentenceTransformers`: **Cosine similarity** between sentence embeddings generated using **SentenceTransformers** (e.g., SBERT). Captures deep semantic alignment. 


###  Directions to use the Web interface.

- Clone the repository

```bash
git clone <repo_name>
```

- Run the below streamlit command to run the app locally and start using the Resume scoring web interface.

```
streamlit run src/app_v1.py
```