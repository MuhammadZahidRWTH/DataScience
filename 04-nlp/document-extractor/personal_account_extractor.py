import re
from normalize import normalize_amount, normalize_date

# Field patterns for different languages
FIELD_LABELS = {
    "account_type": {
        "en": [r"Account Type"],
        "de": [r"Kontotyp", r"Kontoart"],
        "fr": [r"Type\s+de\s+compte", r"Type\s+du\s+compte"],
        "es": [r"Tipo de cuenta"],
        "it": [r"Tipo di conto"]
    },
    "account_number": {
        "en": [r"Account\s*Number", r"IBAN"],
        "de": [r"Konto\s*nummer", r"IBAN"],
        "fr": [r"Numéro\s*de\s*compte", r"IBAN"],
        "es": [r"Número\s*de\s*cuenta", r"IBAN"],
        "it": [r"Numero\s*di\s*conto", r"IBAN"]
    },
    "statement_period": {
        "en": [r"Statement Period"],
        "de": [r"Abrechnungszeitraum"],
        "fr": [r"P[eé]riode\s+de\s+relev[eé]", r"relev[eé]\s+du\s+.*au"],
        "es": [r"Periodo de estado", r"Per[ií]odo del estado"],
        "it": [r"Periodo estratto", r"Periodo di rendiconto"]
    },
    "opening_balance": {
        "en": [r"Opening Balance", r"Previous Balance"],
        "de": [r"Anfangssaldo", r"Vorheriger Kontostand"],
        "fr": [r"Solde\s+de\s+d[eé]part", r"Solde\s+initial"],
        "es": [r"Saldo inicial"],
        "it": [r"Saldo di apertura"]
    },
    "closing_balance": {
        "en": [r"Closing Balance", r"Ending Balance"],
        "de": [r"Endsaldo", r"Kontostand"],
        "fr": [r"Solde\s+final", r"Solde\s+de\s+cl[oô]ture"],
        "es": [r"Saldo final", r"Saldo actual"],
        "it": [r"Saldo di chiusura"]
    },
    "available_balance": {
        "en": [r"Available Balance"],
        "de": [r"Verf[uü]gbarer Betrag"],
        "fr": [r"Solde\s+disponible", r"S[o0]lde\s+disponible"],
        "es": [r"Saldo disponible", r"Disponible"],
        "it": [r"Saldo disponibile"]
    },
    "transaction_number": {
        "en": [
            r"Transaction Number",      # Explicit count label
            r"Number of Transactions",  # Alternative label
            r"Transaction\s+Details\s+\(\d+\)"  # Matches "Transaction Details (5)"
        ],
        "de": [
            r"Transaktionsnummer",
            r"Anzahl\s+der\s+Transaktionen",
            r"Transaktionsdetails\s+\(\d+\)"
        ],
        "fr": [
            r"Numéro\s+de\s+transaction",
            r"Nombre\s+de\s+transactions",
            r"Détails\s+de\s+transaction\s+\(\d+\)"
        ],
        "es": [
            r"Número\s+de\s+transacción",
            r"Cantidad\s+de\s+transacciones",
            r"Detalles\s+de\s+transacción\s+\(\d+\)"
        ],
        "it": [
            r"Numero\s+di\s+transazione",
            r"Quantità\s+di\s+transazioni",
            r"Dettagli\s+di\s+transazione\s+\(\d+\)"
        ]
    }
}

# Function to preprocess OCR text by correcting known errors
def preprocess_ocr_text(text):
    corrections = {
        "S0lde": "Solde",
        "s0lde": "solde",
        "cl0ture": "clôture",
        "Relevé": "Relevé",
        "disponib1e": "disponible",
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# Function to extract personal account fields from text
def extract_personal_account_fields(text, language):
    fields = {}
    text = preprocess_ocr_text(text)  # Step 1: Preprocess OCR text to correct common issues
    lines = text.splitlines()  # Split text into lines for easier matching

    # Function to match field patterns based on language and label
    def match_patterns(label, lang_patterns):
        for pattern in lang_patterns:
            full_pattern = rf"{pattern}[\s:\.\-]*([^\n]+)"  # Match field pattern followed by values
            match = re.search(full_pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                return normalize_amount(raw) if label.endswith("_balance") else raw
        return None

    # Step 2: Iterate through field labels and match patterns
    for field, lang_map in FIELD_LABELS.items():
        patterns = lang_map.get(language, [])  # Get patterns based on language
        result = match_patterns(field, patterns)
        if result:
            fields[field] = result

    # --- Account number (IBAN-style) ---
    if not fields.get("account_number"):
        iban_match = re.search(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){3,6}\s?\d{2}", text)  # Try matching IBAN format
        if iban_match:
            account_number = iban_match.group(0).replace(" ", "")  # Remove spaces from IBAN
            formatted_iban = " ".join([account_number[i:i+4] for i in range(0, len(account_number), 4)])  # Format IBAN in groups
            fields["account_number"] = formatted_iban

    # --- Fallback: infer statement period from two dates ---
    if not fields.get("statement_period"):
        dates = re.findall(r"(\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{2,4})", text)  # Extract potential date ranges
        if len(dates) >= 2:
            start = normalize_date(dates[0], language)  # Normalize start date
            end = normalize_date(dates[1], language)  # Normalize end date
            if start and end:
                fields["statement_period"] = f"{start} – {end}"  # Format statement period

    # --- Transaction Counting ---
    fields["transaction_number"] = 0  # Initialize count

    # Method 1: Count rows in transaction table
    if "Transaction Details" in text:
        trans_section = re.search(r"Transaction Details(.*?)(?:\n\s*\n|\Z)", text, re.DOTALL)  # Find transaction section
        if trans_section:
            trans_rows = re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b", trans_section.group(1))  # Find transaction rows
            fields["transaction_number"] = max(0, len(trans_rows) - 1)  # Subtract 1 to exclude header

    # Method 2: Fallback to line counting if table not found
    if fields["transaction_number"] == 0:
        trans_lines = re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}.+?[+-]\d+,\d{2}\b", text)  # Match transaction lines
        fields["transaction_number"] = max(0, len(trans_lines) - 1)  # Subtract 1 safely

    # --- Special Handling for Spanish Overdraft Notices ---
    if "Operaciones que causaron el descubierto" in text:
        table_match = re.search(r"Operaciones que causaron el descubierto\s*[\|]?\s*(.*?)(?:\n\s*\n|\Z)", text, re.DOTALL)  # Detect overdraft-related table
        if table_match:
            transactions = re.findall(r"(\d{2}/\d{2}/\d{4})\s*\|?\s*([^\|]+?)\s*\|?\s*(-?\d[\d\.]*,\d{2})", table_match.group(1))  # Extract transactions
            fields["transaction_number"] = len(transactions)  # Set transaction count

    # --- Fallback if no transactions found ---
    if "transaction_number" not in fields or fields["transaction_number"] == 0:
        transactions = re.findall(r"(\d{2}/\d{2}/\d{4}).*?(-?\d[\d\.]*,\d{2})", text)  # Match date + amount pattern
        fields["transaction_number"] = len(transactions)  # Count transactions


    return fields
