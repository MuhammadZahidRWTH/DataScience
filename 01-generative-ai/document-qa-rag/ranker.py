import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load Model and Tokenizer
model_name="BAAI/bge-reranker-base"

reranker_model=AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
reranker_tokenizer=AutoTokenizer.from_pretrained(model_name)


def rerank_documents(query,docs,top_n=3):
    pairs = [(query, doc.page_content if hasattr(doc, "page_content") else str(doc)) for doc in docs]

    inputs=reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    with torch.no_grad():
        scores=reranker_model(**inputs).logits.squeeze(-1)



    # Sort by score
    scored_docs=list(zip(scores.tolist(),docs))

    for score,doc in scored_docs:
        if not hasattr(doc,"metadata"):
            doc.metadata={}
        doc.metadata['rerank_score']=score

    scored_docs.sort(key=lambda x:x[0],reverse=True)

    # Return top n docs

    return [doc for score, doc in scored_docs[:top_n]]