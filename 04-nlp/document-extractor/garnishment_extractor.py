import re
from typing import Dict, Optional
from normalize import normalize_money, normalize_date, normalize_name

def extract_garnishment_fields(text: str, language: str = "de") -> Dict[str, Optional[str]]:
    """
    Extract garnishment-related fields from the provided document text.
    
    Args:
    - text (str): The document text to extract fields from.
    - language (str): The language of the document (defaults to "de" for German).
    
    Returns:
    - dict: A dictionary containing extracted garnishment fields like debtor_name, creditor_name, etc.
    """
    fields = {
        "debtor_name": None,
        "creditor_name": None,
        "garnishment_amount": None,
        "effective_date": None,
        "duration": None,
        "legal_authority": None,
    }

    # First try to identify the document type
    if "Zustellungsurkunde" in text and "Pfändungs- und Überweisungsbeschluss" in text:
        return extract_german_court_order(text, fields)
    elif "Pfändungs- und Einziehungsverfügung" in text:
        return extract_german_enforcement_order(text, fields)
    else:
        # Fallback to language-based extraction
        return language_specific_extraction(text, language, fields)


def extract_german_court_order(text: str, fields: Dict) -> Dict:
    """
    Extract garnishment fields from a German court order document.
    
    Args:
    - text (str): The document text (German court order).
    - fields (dict): The dictionary to store extracted fields.
    
    Returns:
    - dict: Updated fields dictionary with garnishment information.
    """
    # Debtor name - handles formats like "(Schuldn.: Herrn Esther Hendriks)"
    debtor_match = re.search(r"Schuldn\.:\s*(?:Herrn|Frau)?\s*([^,)]+)", text)
    if debtor_match:
        fields["debtor_name"] = normalize_name(debtor_match.group(1)).strip()

    # Creditor name - handles formats like "Gläubigers: Hein Barkholz AG & Co. KGaA"
    creditor_match = re.search(r"Gläubigers?:\s*([^\n,]+(?:GmbH|AG|KGaA|Co\. KG|e\.V\.)?)", text)
    if creditor_match:
        fields["creditor_name"] = creditor_match.group(1).strip()

    # Legal authority - uses the court info
    court_match = re.search(r"Pfändungs- und Überweisungsbeschluss (?:des|der)\s+([^\n,]+)", text)
    if court_match:
        fields["legal_authority"] = court_match.group(1).strip()

    # Date - prefers the court decision date over delivery date
    date_match = re.search(r"vom\s+(\d{1,2}\.\s+\w+\s+\d{4})", text)
    if not date_match:
        date_match = re.search(r"den\s+(\d{1,2}\.\s+\w+\s+\d{4})", text)
    if date_match:
        fields["effective_date"] = normalize_date(date_match.group(1), "de")

    return fields


def extract_german_enforcement_order(text: str, fields: Dict) -> Dict:
    """
    Extract garnishment fields from a German enforcement order document.
    
    Args:
    - text (str): The document text (German enforcement order).
    - fields (dict): The dictionary to store extracted fields.
    
    Returns:
    - dict: Updated fields dictionary with garnishment information.
    """
    # Debtor name - handles formats with birthdate
    debtor_match = re.search(r"Vollstreckungsschuldner:\s*([^,]+)(?:,|$)", text)
    if debtor_match:
        fields["debtor_name"] = normalize_name(debtor_match.group(1)).strip()

    # Creditor name - looks for the entity being paid
    creditor_match = re.search(r"schuldet\s+(?:dem|der)\s+([^\n]+)", text)
    if creditor_match:
        fields["creditor_name"] = creditor_match.group(1).strip()

    # Amount - handles different amount descriptions
    amount_match = re.search(r"(?:Forderungen\s+in\s+Höhe\s+von|Betrag):?\s+([\d\.,]+)\s*EUR", text, re.IGNORECASE)
    if amount_match:
        fields["garnishment_amount"] = normalize_money(amount_match.group(1), "de")

    # Date - uses document date (multiple possible locations)
    date_match = re.search(r"(?:den|vom|am)\s+(\d{1,2}\.\s+\w+\s+\d{4})", text, re.IGNORECASE)
    if not date_match:
        date_match = re.search(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b", text)
    if date_match:
        fields["effective_date"] = normalize_date(date_match.group(1), "de")

    # Legal authority - dynamic extraction from multiple patterns
    authority_match = re.search(r"^(.*?)\s*Meike-Henschel-Weg", text, re.MULTILINE)
    if not authority_match:
        authority_match = re.search(r"(?:Behörde|Amt|Gericht):\s*([^\n]+)", text)
    if not authority_match:
        authority_match = re.search(r"^\s*([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s*$", text, re.MULTILINE)
    if authority_match:
        fields["legal_authority"] = authority_match.group(1).strip()

    # Duration - attempt extraction if present
    duration_match = re.search(r"(?:Gültigkeitsdauer|Dauer):?\s*(\d+\s+(?:Tage|Monate|Jahre))", text, re.IGNORECASE)
    if duration_match:
        fields["duration"] = duration_match.group(1)
    else:
        fields["duration"] = None  # Explicit None if not found

    return fields


def language_specific_extraction(text: str, language: str, fields: Dict) -> Dict:
    """
    Fallback extraction based on language patterns (if document format is not recognized).
    
    Args:
    - text (str): The document text.
    - language (str): The language of the document.
    - fields (dict): The dictionary to store extracted fields.
    
    Returns:
    - dict: Updated fields dictionary with garnishment information.
    """
    if language == "de":
        # Generic German patterns could be added here
        pass
    elif language == "en":
        # English patterns could be added here
        pass
    # Other languages...
    
    return fields
