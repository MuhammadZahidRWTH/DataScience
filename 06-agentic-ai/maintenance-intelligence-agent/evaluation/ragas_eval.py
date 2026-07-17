# evaluation/ragas_eval.py
# RAGAS Evaluation for MaintenanceGPT RAG Knowledge Agent
#
# Evaluates RAG responses using RAGAS metrics:
#   - Faithfulness: are answers grounded in retrieved context?
#   - Answer Relevancy: does the answer address the question?
#
# Implements reliable agent evaluation for production agentic AI systems.
# research direction for production agentic AI systems.

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_PROCESSED

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("  [RAGAS] ragas package not installed. Run: pip install ragas")
    print("  [RAGAS] Running in mock evaluation mode for demonstration.\n")


def load_rag_responses() -> list[dict]:
    """Load RAG responses produced by the RAG Knowledge Agent."""
    try:
        with open(DATA_PROCESSED / "rag_responses.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("  [RAGAS] No RAG responses found. Run pipeline.py first.")
        return []


def load_manual_context() -> str:
    """Load machine manuals for context verification."""
    manual_path = ROOT / "data" / "manuals" / "machine_manuals.txt"
    try:
        with open(manual_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def build_ragas_dataset(rag_responses: list[dict], manual_context: str) -> dict:
    """
    Build RAGAS evaluation dataset from RAG responses.

    RAGAS expects:
    - question: the query asked
    - answer: the LLM response
    - contexts: list of retrieved document chunks
    - ground_truth: reference answer (optional for some metrics)
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for r in rag_responses:
        machine = r.get("machine_id", "unknown")
        fault = r.get("fault", "unknown")
        sensor = r.get("sensor", "unknown")

        question = (
            f"What is the maintenance procedure for {machine} "
            f"with a {fault} fault on {sensor} sensor?"
        )

        answer = r.get("manual_procedure", "")
        sources = r.get("sources", [manual_context[:500]])

        # ground truth — expected action based on fault type
        gt_map = {
            "bearing_wear": "Inspect bearings, apply lubrication, schedule replacement if vibration exceeds threshold.",
            "overheating": "Stop machine, inspect coolant system, check for blockages, verify flow rate.",
            "pressure_issue": "Inspect hydraulic seals, check fluid level and quality.",
        }
        ground_truth = gt_map.get(fault, "Refer to machine maintenance manual for specific procedures.")

        questions.append(question)
        answers.append(answer)
        contexts.append(sources if sources else [manual_context[:500]])
        ground_truths.append(ground_truth)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_mock_evaluation(rag_responses: list[dict]) -> dict:
    """
    Mock evaluation when RAGAS is not installed.
    Computes simple heuristic scores to demonstrate the evaluation framework.
    """
    print("  [RAGAS] Running heuristic evaluation (mock mode)...\n")

    scores = []
    for r in rag_responses:
        answer = r.get("manual_procedure", "")
        fault = r.get("fault", "")
        machine = r.get("machine_id", "")

        # faithfulness proxy: does answer mention machine/fault keywords?
        keywords = [machine.lower(), fault.replace("_", " ")]
        keyword_hits = sum(1 for kw in keywords if kw in answer.lower())
        faithfulness_score = min(keyword_hits / len(keywords), 1.0)

        # relevancy proxy: answer length and specificity
        word_count = len(answer.split())
        relevancy_score = min(word_count / 50, 1.0)

        scores.append({
            "machine": machine,
            "fault": fault,
            "faithfulness": round(faithfulness_score, 3),
            "answer_relevancy": round(relevancy_score, 3),
        })

        print(f"  {machine} | {fault}")
        print(f"    faithfulness:     {faithfulness_score:.3f}")
        print(f"    answer_relevancy: {relevancy_score:.3f}")
        print()

    avg_faithfulness = sum(s["faithfulness"] for s in scores) / len(scores) if scores else 0
    avg_relevancy = sum(s["answer_relevancy"] for s in scores) / len(scores) if scores else 0

    return {
        "mode": "mock_heuristic",
        "n_responses": len(scores),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_answer_relevancy": round(avg_relevancy, 3),
        "per_response": scores,
    }


def run_ragas_evaluation(dataset: dict) -> dict:
    """Run full RAGAS evaluation with faithfulness and answer relevancy."""
    print("  [RAGAS] Running full RAGAS evaluation...\n")

    ragas_dataset = Dataset.from_dict(dataset)

    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    scores = result.to_pandas()
    print(scores[["question", "faithfulness", "answer_relevancy"]].to_string(index=False))

    return {
        "mode": "ragas",
        "avg_faithfulness": round(float(scores["faithfulness"].mean()), 3),
        "avg_answer_relevancy": round(float(scores["answer_relevancy"].mean()), 3),
        "per_response": scores.to_dict(orient="records"),
    }


def save_results(results: dict) -> None:
    """Save evaluation results to data/processed/."""
    output_path = DATA_PROCESSED / "ragas_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  [RAGAS] Results saved to {output_path}")


def main():
    print("Evaluating RAG Knowledge Agent outputs:")
    print("  Metrics: Faithfulness · Answer Relevancy")
    print("  Evaluation: RAGAS faithfulness + answer relevancy\n")
    print("-" * 50)

    rag_responses = load_rag_responses()
    if not rag_responses:
        return

    manual_context = load_manual_context()
    print(f"  [RAGAS] Loaded {len(rag_responses)} RAG responses")
    print(f"  [RAGAS] Manual context: {len(manual_context)} characters\n")

    if RAGAS_AVAILABLE:
        dataset = build_ragas_dataset(rag_responses, manual_context)
        results = run_ragas_evaluation(dataset)
    else:
        results = run_mock_evaluation(rag_responses)

    save_results(results)

    print("\n" + "=" * 50)
    print("  Evaluation Summary")
    print("=" * 50)
    print(f"  Mode:               {results['mode']}")
    print(f"  Responses evaluated: {results['n_responses']}")
    print(f"  Avg Faithfulness:    {results['avg_faithfulness']}")
    print(f"  Avg Answer Relevancy:{results['avg_answer_relevancy']}")
    print("=" * 50)
    print("\nInterpretation:")
    print("  Faithfulness > 0.7     → answers grounded in manual context")
    print("  Answer Relevancy > 0.7 → answers address the maintenance question")
    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
