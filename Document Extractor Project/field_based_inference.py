import re
from extractor_utils import classify_document_type

def infer_document_type(fields, raw_text=""):
    """
    Infer document type using a scoring system based on extracted fields.
    """
    scoring_config = {
        "credit": [
            "card_number", "credit_limit", "interest_rate",
            "payment_due_date", "minimum_payment", "previous_balance", "new_balance"
        ],
        "personal_account": [
            "account_number", "account_type", "statement_period",
            "opening_balance", "closing_balance", "available_balance", "transaction_number"
        ],
        "garnishment": [
            "garnishment_amount", "creditor_name", "debtor_name",
            "effective_date", "duration", "legal_authority"
        ],
        "investment": [
            "portfolio_id", "portfolio_value", "asset_number", "risk_profile"
        ]
    }

    # Calculate field match score
    scores = {}
    for doc_type, keys in scoring_config.items():
        scores[doc_type] = sum(1 for k in keys if fields.get(k))

    # Select the document type with the highest score
    max_score_type = max(scores, key=lambda k: scores[k])
    if scores[max_score_type] > 0:
        return max_score_type

    # Fallback logic if all scores are zero
    language = fields.get("language", "").lower()
    text = raw_text.lower()

    if language == "es" and any(keyword in text for keyword in ["regularizar", "saldo pendiente", "pagar antes de"]):
        return "credit"
    if language == "fr" and any(keyword in text for keyword in ["solde impayé", "paiement requis"]):
        return "credit"
    if language == "fr" and any(keyword in text for keyword in ["solde disponible", "période de relevé"]):
        return "personal_account"
    if language == "de" and any(keyword in text for keyword in ["zahlung erforderlich", "offener betrag", "kontoüberweisung"]):
        return "credit"
    if language == "en" and any(keyword in text for keyword in ["investment portfolio summary", "portfolio valuation"]):
        return "investment"
    if language == "en" and "account statement" in text:
        return "personal_account"

    # Final fallback: keyword-based guess
    keyword_guess = classify_document_type(raw_text)
    if keyword_guess != "unknown":
        return keyword_guess

    return "unknown"
