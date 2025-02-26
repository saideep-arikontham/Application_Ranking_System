from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import asyncio
import time


import os
import pandas as pd

# Set Pandas to display all columns
pd.set_option('display.max_columns', None)

# Initialize Model
model = ChatNVIDIA(
  model="nvidia/nemotron-4-340b-instruct",
  api_key="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",
  temperature=1.0
)
# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Generate a challenging machine learning question related to {topic}."
)

# Create Chain
chain = prompt | model

# Topics for generating ML questions
topics = [
    "Supervised Learning",
    "Unsupervised Learning",
    "Neural Networks",
    "Gradient Descent",
    "Regularization in ML",
    "Support Vector Machines",
    "Ensemble Learning",
    "Dimensionality Reduction",
    "Hyperparameter Tuning",
    "Bias-Variance Tradeoff"
]


# ------------------ 🚀 Iterative (Sequential) Calls ------------------ #
def generate_questions_iteratively():
    start_time = time.time()

    questions = []
    for topic in topics:
        response = chain.invoke({"topic": topic})
        questions.append(response.content)

    end_time = time.time()
    print(f"\n⏳ Iterative Execution Time: {end_time - start_time:.2f} seconds")
    return questions


# ------------------ 🚀 Asynchronous Calls (Parallel) ------------------ #
async def generate_question_async(topic):
    return await chain.ainvoke({"topic": topic})


async def batch_generate_questions_async():
    start_time = time.time()

    tasks = [generate_question_async(topic) for topic in topics]
    results = await asyncio.gather(*tasks)  # Execute all tasks concurrently
    questions = [res.content for res in results]

    end_time = time.time()
    print(f"\n⚡ Async Execution Time: {end_time - start_time:.2f} seconds")
    return questions


# ------------------ 🚀 Run and Compare ------------------ #
if __name__ == "__main__":
    print("Running Iterative Execution...")
    questions_iterative = generate_questions_iteratively()

    print("\nRunning Asynchronous Execution...")
    questions_async = asyncio.run(batch_generate_questions_async())



# ========================================================================
# OUTPUT:

# Running Iterative Execution...
# ⏳ Iterative Execution Time: 279.28 seconds
#
# Running Asynchronous Execution...
# ⚡ Async Execution Time: 68.90 seconds
# ========================================================================