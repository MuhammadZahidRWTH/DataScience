from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_components():
    """Initialize all required components with proper configuration"""
    # 1. Load and prepare data
    df = pd.read_csv('preprocessed_data.csv')
    
    # Convert numeric columns properly
    numeric_cols = ['Energy Efficiency', 'Carbon Footprint (tons)', 'CO2 Emissions (tons)', 
                   'Energy Usage (kWh)', 'Material Efficiency (%)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Handle non-numeric columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna('Unknown')
    
    # Prepare text for embedding
    text_columns = ['Product Name_x', 'Category', 'Compliance Status', 
                   'Hazard Classification', 'Region', 'Product Name_y']
    df['text'] = df.apply(lambda row: ' | '.join(str(row[col]) for col in text_columns), axis=1)
    
    # 2. Create document store
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Create embeddings (using direct sentence-transformers)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # 4. Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("compliance_faiss_index")
    
    # 5. Set up LLM pipeline
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        device="cpu"
    )
    
    # 6. Create LLM wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 7. Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain, df



def generate_report(qa_chain, report_type):
    """Generate compliance or sustainability report"""
    if report_type == "compliance":
        prompt = """
        Analyze the compliance data and generate a comprehensive report covering:
        1. Overall compliance status (percentage compliant)
        2. Most common hazard classifications
        3. Regional compliance patterns
        4. Product categories with highest non-compliance
        5. Any correlations between compliance and sustainability metrics
        
        Provide specific insights and actionable recommendations.
        """
    else:
        prompt = """
        Analyze the sustainability metrics including:
        - CO2 Emissions distribution
        - Energy Usage patterns
        - Material Efficiency
        
        Identify:
        1. Products with best/worst sustainability scores
        2. Correlations between different sustainability metrics
        3. Regional differences in sustainability
        4. Potential improvement areas
        
        Provide specific recommendations for improving sustainability.
        """
    return qa_chain.invoke({"query": prompt})['result']

def answer_question(qa_chain, question):
    """Answer a question with source references"""
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result['result'],
        "sources": [doc.metadata for doc in result['source_documents']]
    }

def generate_visualization(df, metric):
    """Generate and analyze visualization"""
    plt.figure(figsize=(10, 6))
    
    if metric == 'compliance':
        sns.countplot(x='Compliance Status', data=df)
        plt.title('Product Compliance Status')
    elif metric == 'hazards':
        sns.countplot(y='Hazard Classification', data=df, 
                     order=df['Hazard Classification'].value_counts().index)
        plt.title('Product Hazard Classifications')
    elif metric == 'sustainability':
        sns.scatterplot(x='Energy Usage (kWh)', y='CO2 Emissions (tons)', 
                       hue='Material Efficiency (%)', data=df)
        plt.title('Energy Usage vs CO2 Emissions')
    
    img_path = f"{metric}_plot.png"
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    
    return img_path

def main():
    print("=== Compliance & Sustainability Analyzer ===")
    print("Using RAG with local LLM and vector embeddings\n")
    
    # Initialize components
    qa_chain, df = initialize_components()
    
    # Generate reports
    print("\nGenerating Compliance Report...")
    print(generate_report(qa_chain, "compliance"))
    
    print("\nAnalyzing Sustainability Metrics...")
    print(generate_report(qa_chain, "sustainability"))
    
    # Example questions
    questions = [
        "Which supplier has the worst compliance record?",
        "What is the average CO2 emissions for products from Europe?",
        "Are there any products that are both highly sustainable and fully compliant?",
        "What is the relationship between lead time and product availability?"
    ]
    
    print("\nAnswering Sample Questions:")
    for q in questions:
        print(f"\nQ: {q}")
        result = answer_question(qa_chain, q)
        print(f"A: {result['answer']}")
        print("Sources:", result['sources'][0])
    
    # Generate visualizations
    print("\nGenerating Visualizations:")
    for metric in ['compliance', 'hazards', 'sustainability']:
        img_path = generate_visualization(df, metric)
        print(f"\nGenerated {metric} visualization: {img_path}")

if __name__ == "__main__":
    main()
