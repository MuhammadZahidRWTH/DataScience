import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langdetect import detect, DetectorFactory
import tempfile

DetectorFactory.seed = 0  # Consistent language detection

SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "it"}

DOC_TYPE_KEYWORDS = {
    "garnishment": ["garnishment", "pfändung", "saisie", "embargo", "pignoramento", "zustellung", "vollstreckung"],
    "investment": ["investment", "portfolio", "anlage", "portefeuille", "inversione", "investment portfolio summary",
        "certificato di deposito", "deposito", "deposit certificate"],
    "credit": [
        "credit", "karte", "crédit", "credito", "tarjeta", "loan", "mortgage",
        "hypothek", "darlehen"
    ],
    "personal_account": ["account","Account Number", "konto", "compte", "cuenta", "conto", "solde disponible", "période de relevé"]
}


def extract_text_from_pdf(file_path):
    """Extract text from PDF. Use OCR fallback if PyPDF2 fails."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        pass

    if text.strip():
        return text

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            images = convert_from_path(file_path, dpi=300, output_folder=tmpdir)
            for img in images:
                text += pytesseract.image_to_string(img, lang='deu+eng+fra+spa+ita')
    except Exception:
        pass

    return text


def detect_language(text):
    """Detects document language. Returns ISO 639-1 code."""
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "unsupported"
    except:
        return "unknown"


def classify_document_type(text):
    """Classifies the document type using multilingual keyword matching."""
    lowered = text.lower()
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return doc_type
    return "unknown"


def structure_extracted_fields(flat_fields: dict) -> dict:
    schema = {
        "general": [
            "document_id", "document_type", "document_date", "customer_name",
            "customer_id", "institution_name", "institution_address", "language"
        ],
        "investment": [
            "portfolio_id", "portfolio_value", "asset_number", "risk_profile"
        ],
        "personal_account": [
            "account_number", "account_type", "statement_period", "opening_balance",
            "closing_balance", "available_balance", "transaction_number"
        ],
        "garnishment": [
            "debtor_name", "creditor_name", "garnishment_amount", "effective_date",
            "duration", "legal_authority"
        ],
        "credit": [
            "card_number", "credit_limit", "interest_rate", "payment_due_date",
            "statement_period", "minimum_payment", "previous_balance", "new_balance"
        ]
    }

    grouped = {}
    for section, keys in schema.items():
        section_data = {
            key: flat_fields[key] for key in keys
            if key in flat_fields and flat_fields[key] is not None
        }
        if section_data:
            grouped[section] = section_data

    # Debugging aid: if nothing matches, attach raw fields for investigation
    if not grouped:
        grouped["debug"] = {k: v for k, v in flat_fields.items() if v is not None}

    return grouped