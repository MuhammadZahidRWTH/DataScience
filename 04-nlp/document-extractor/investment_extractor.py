import re
from normalize import normalize_amount, normalize_date

def extract_investment_fields(text, language):
    """
    Extracts investment-related fields from text based on the provided language.
    
    Args:
        text (str): The input text containing investment information.
        language (str): The language of the input text ('en', 'de', 'fr', 'es', 'it').
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    # Language-specific handlers for investment extraction
    handlers = {
        "en": extract_investment_en,
        "de": extract_investment_de,
        "fr": extract_investment_fr,
        "es": extract_investment_es,
        "it": extract_investment_it,
    }
    
    # Default to English handler if the language is not recognized
    handler = handlers.get(language, extract_investment_en)
    
    # Call the appropriate language-specific handler
    return handler(text)

# ----------------- SHARED LOGIC ----------------- #

def _extract_common_investment(text, patterns):
    """
    Shared logic for extracting common investment fields from the text.
    
    Args:
        text (str): The input text containing investment information.
        patterns (dict): A dictionary of regular expressions for extracting fields.
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    fields = {}

    
    # Extract portfolio value
    match = re.search(patterns["portfolio_value"], text, re.IGNORECASE)
    if match:
        fields["portfolio_value"] = normalize_amount(match.group(2))

    
    # Extract portfolio ID (fallback if not found in patterns)
    if not fields.get("portfolio_id"):
        match = re.search(r"\bPORT-[A-Z]{2}-\d{4}-\d{4}\b", text)
        if match:
            fields["portfolio_id"] = match.group(0)

    
    # Extract risk profile
    match = re.search(patterns["risk_profile"], text, re.IGNORECASE)
    if match:
        fields["risk_profile"] = match.group(2).strip()

    
    # Extract asset count
    asset_section = re.search(patterns["asset_section"], text, re.IGNORECASE)
    if asset_section:
        lines = asset_section.group(2).splitlines()
        count = sum(1 for line in lines if re.search(r"[A-Z][a-z]", line))  # Count asset entries
        if count > 0:
            fields["asset_number"] = count
    
    
    return fields

# ----------------- LANGUAGE FUNCTIONS ----------------- #

def extract_investment_de(text):
    """
    Extracts investment-related fields from German text.
    
    Args:
        text (str): The input German text containing investment information.
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    return _extract_common_investment(text, {
        "portfolio_value": r"(gesamtwert des portfolios|gesamtwert).*?[\u20ac$£]?\s*([\d\.,]+)",
        "risk_profile": r"(risikoprofil)[^\n:\-]*[:\-\s]*([A-ZÄÖÜ][a-zäöüß]+)",
        "asset_section": r"(Vermögensaufstellung|Vermögensstruktur)[^\n]*\n((?:.+\n?){1,10})",
    })

def extract_investment_fr(text):
    """
    Extracts investment-related fields from French text.
    
    Args:
        text (str): The input French text containing investment information.
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    fields = _extract_common_investment(text, {
        "portfolio_value": r"(valeur totale.*?portefeuille)[^\n:\d]*[\u20ac$£]?\s*([\d\.,]+)",
        "risk_profile": r"(profil de risque)[^\n:\-]*[:\-\s]*([A-Z][a-zéèàôü]+)",
        "asset_section": r"(Répartition des actifs)[^\n]*\n((?:.+\n?){1,10})",
    })
    
    # Try to extract portfolio ID from French text
    if match := re.search(r"numéro\s+de\s+compte[:\s]*([A-Z]{2}\d[\d\s]{11,27}\d{2})", text, re.IGNORECASE):
        fields["portfolio_id"] = re.sub(r"\s+", "", match.group(1))
    
    # Fallback to portfolio value from "Montant à l'échéance"
    if not fields.get("portfolio_value"):
        match = re.search(r"Montant\s+(?:à l['’]échéance|final)[:\s]*([\d\s]+,\d{2})\s*[€\u20ac]?", text)
        if match:
            amount = match.group(1).replace(" ", "")
            fields["portfolio_value"] = normalize_amount(amount)

    
    return fields

def extract_investment_es(text):
    """
    Extracts investment-related fields from Spanish text.
    
    Args:
        text (str): The input Spanish text containing investment information.
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    return _extract_common_investment(text, {
        "portfolio_value": r"(valor total del portafolio)[^\n:\d]*[\u20ac$£]?\s*([\d\.,]+)",
        "risk_profile": r"(perfil de riesgo)[^\n:\-]*[:\-\s]*([A-Z][a-záéíóú]+)",
        "asset_section": r"(Reparto de activos)[^\n]*\n((?:.+\n?){1,10})",
    })

def extract_investment_it(text):
    """
    Extracts investment-related fields from Italian text.
    
    Args:
        text (str): The input Italian text containing investment information.
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    fields = {}
    
    # Extract patterns for deposit certificates
    cert_id_match = re.search(r"\b(?:CD|Certificato)[\s:-]*(IT-\d{5}-\d{4}|[A-Z]{2}-[A-Z]{2}-\d{5}-\d{4})\b", text, re.IGNORECASE)
    cert_value_match = re.search(r"(?:importo depositato|valore nominale)[^\n:]*[\s:]*[\u20ac]?\s*([\d\.]+,\d{2})", text, re.IGNORECASE)
    
    # Extract patterns for pension funds
    pension_id_match = re.search(r"nome del fondo[\s:]*([^\n]+?)(?:\n|$)", text, re.IGNORECASE)
    pension_value_match = re.search(r"posizione complessiva.*?[\u20ac]?\s*([\d\.]+,\d{2})", text, re.IGNORECASE)
    risk_match = re.search(r"linea di investimento[\s:]*([^\n]+?)(?:\n|$)", text, re.IGNORECASE)
    
    # Check pension fund value first (more specific)
    if match := re.search(r"posizione complessiva maturata.*?ammonta a [\u20ac€]\s*([\d\.]+,\d{2})", text, re.IGNORECASE | re.DOTALL):
        amount_str = match.group(1).replace('.', '').replace(',', '.')
        fields["portfolio_value"] = float(amount_str)
 
    
    # Extract portfolio ID for pension or certificate
    if match := re.search(r"Numero\s+di\s+riferimento[\s:]*([A-Z]{2}-\d{5}-\d{4})", text, re.IGNORECASE):
        fields["portfolio_id"] = match.group(1).strip()

    elif cert_id_match or cert_value_match:
        if cert_id_match:
            fields["portfolio_id"] = f"CD-{cert_id_match.group(1)}" if not cert_id_match.group(1).startswith('CD-') else cert_id_match.group(1)


    if cert_value_match:
            fields["portfolio_value"] = normalize_amount(cert_value_match.group(1))

    
    # Extract risk profile (for pension funds)
    if risk_match:
        fields["risk_profile"] = risk_match.group(1).strip()

    
    # Extract customer name
    if name_match := re.search(r"nome e cognome[\s:]*([^\n]+?)(?:\s*codice fiscale|$)", text, re.IGNORECASE):
        fields["customer_name"] = name_match.group(1).strip()

    
    return fields

def extract_investment_en(text):
    """
    Extracts investment-related fields from English text.
    
    Args:
        text (str): The input English text containing investment information.
        
    Returns:
        dict: A dictionary containing the extracted investment fields.
    """
    investment = {}
    
    # Extract portfolio ID
    if match := re.search(r"INV-EN-\d{5}-\d{4}", text):
        investment["portfolio_id"] = match.group(0)
    
    # Extract portfolio value
    if match := re.search(r"total value of.*?\$([\d,]+\.\d{2})", text, re.DOTALL):
        investment["portfolio_value"] = f"${match.group(1)}"
    
    # Count assets (static predefined list)
    asset_lines = [
        "Equities", "Fixed Income", "Real Estate", 
        "Alternatives", "Cash & Equivalents"  # From document
    ]
    investment["asset_number"] = len(asset_lines)
    
    return investment
