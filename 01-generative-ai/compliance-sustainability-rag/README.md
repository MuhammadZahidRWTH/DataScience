# Compliance & Sustainability Analyzer

This project leverages a Retrieval-Augmented Generation (RAG) model to analyze product compliance and sustainability data. The model uses a combination of Hugging Face’s transformers, FAISS vector store, and LangChain components to provide actionable insights on compliance status, hazard classifications, sustainability metrics (CO2 emissions, energy usage), and more.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Components](#components)
- [Usage](#usage)
- [Report Generation](#report-generation)
- [Visualization](#visualization)
- [Example Questions](#example-questions)
- [License](#license)

## Project Overview

The Compliance & Sustainability Analyzer project performs several tasks:
- Loads and processes compliance and sustainability data.
- Embeds text data using pre-trained transformer models.
- Stores the documents and embeddings in a FAISS vector store.
- Provides detailed reports on compliance and sustainability metrics.
- Allows users to ask questions based on the available data.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/MuhammadZahidRWTH/Compliance-Sustainability-Analyzer.git
cd Compliance-Sustainability-Analyzer
Step 2: Create a virtual environment
bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Step 3: Install dependencies
bash
pip install -r requirements.txt
Step 4: Download pre-trained models
Make sure to download the required models from Hugging Face or other model repositories.

📦 Dependencies
LangChain

FAISS

Transformers

Pandas

Matplotlib

Seaborn

🧱 Components
1. Data Loading and Preprocessing
Loads data from CSV files.

Handles numeric and non-numeric columns.

Cleans and converts data appropriately.

2. Document Store Creation
Uses DataFrameLoader to convert CSV data into documents.

3. Embedding and Vector Store
Applies Hugging Face embeddings.

Stores vectors in FAISS for fast retrieval.

4. LLM Pipeline and RAG Chain
Uses Google FLAN-T5 for text generation.

Sets up a retrieval-based QA chain.

5. Report Generation
Generates detailed compliance and sustainability reports.

6. Visualizations
Uses matplotlib and seaborn to create insightful plots.

🚀 Usage
Initialize Components
python
qa_chain, df = initialize_components()
Generate Reports
python
compliance_report = generate_report(qa_chain, "compliance")
print(compliance_report)

sustainability_report = generate_report(qa_chain, "sustainability")
print(sustainability_report)
Answer Questions
python
question = "Which supplier has the worst compliance record?"
answer = answer_question(qa_chain, question)
print(f"Answer: {answer['answer']}")
📄 Report Generation
1. Compliance Report
Includes:

Overall compliance status (percentage compliant)

Common hazard classifications

Regional compliance patterns

Product categories with highest non-compliance

Correlations between compliance and sustainability metrics

2. Sustainability Report
Includes:

CO₂ emissions distribution

Energy usage patterns

Material efficiency

Products with best/worst sustainability scores

Regional differences

Areas for improvement

📊 Visualization
Compliance Visualization
Count plot of product compliance status

Hazard Classification Visualization
Count plot of common hazard classifications

Sustainability Visualization
Scatter plot of energy usage vs CO₂ emissions

python
compliance_img = generate_visualization(df, 'compliance')
sustainability_img = generate_visualization(df, 'sustainability')
❓ Example Questions
You can ask the system:

"Which supplier has the worst compliance record?"

"What is the average CO₂ emissions for products from Europe?"

"Are there any products that are both highly sustainable and fully compliant?"

"What is the relationship between lead time and product availability?"