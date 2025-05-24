import re
import uuid
from normalize import normalize_date, normalize_name, normalize_address
from extractor_utils import detect_language

def extract_general_fields(text, language, document_type=None):
    
    fields = {
        "document_id": None,  # Start with None
        "language": language,
        "document_type": document_type or "unknown",
        "document_date": None,
        "customer_name": None,
        "customer_id": None,
        "institution_name": None,
        "institution_address": None,
    }

    # Try to extract the document ID using the regex pattern
    document_id_match = re.search(r"\b[A-Z]{2,5}(?:-[A-Z]{2})?-\d{4,6}-\d{4}\b", text)
    if document_id_match:
        fields["document_id"] = document_id_match.group(0)
    else:
        # Only generate a random UUID if no document ID was found
        print("[DEBUG] No document ID found.")
        if fields["document_id"] is None:
            print("[DEBUG] Generating a random UUID for document ID.")
            fields["document_id"] = str(uuid.uuid4())


    # --- Language-specific routing ---
    if language == "de":
        extract_general_de(text, fields)
    elif language == "fr":
        extract_general_fr(text, fields)
    elif language == "es":
        extract_general_es(text, fields)
    elif language == "it":
        extract_general_it(text, fields)
    elif language == "en":
        extract_general_en(text, fields)

    return fields

# ---------------------- GERMAN ----------------------
def extract_general_de(text, fields):
    """German document field extraction."""

    if not fields.get("document_date"):
    # Pattern 1: German date format (e.g., "12. März 2025" or "den 12. März 2025")
        date_match = re.search(
        r"(?:den\s+)?(\d{1,2}\.\s*[A-Za-zÄÖÜäöüß]+\s+\d{4})", 
        text, 
        re.IGNORECASE
    )
    if date_match:
        fields["document_date"] = normalize_date(date_match.group(1), "de")
    else:
        # Pattern 2: Numeric date formats (DD.MM.YYYY, DD-MM-YYYY, DD/MM/YYYY)
        match = re.search(
            r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\b", 
            text
        )
        if match:
            # Ensure consistent DD.MM.YYYY format output
            fields["document_date"] = f"{match.group(1).zfill(2)}.{match.group(2).zfill(2)}.{match.group(3)}"

    # --- CUSTOMER NAME ---
    if not fields.get("customer_name"):
        match = re.search(r"Schuldn\.:\s*(?:Herrn|Frau)?\s*([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)", text)
        if match:
            fields["customer_name"] = normalize_name(match.group(1))
        else:
            match = re.search(r"Name[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)", text)
            if match:
                fields["customer_name"] = normalize_name(match.group(1).strip())

    # --- CUSTOMER ID (includes Kundennummer and Az.) ---
    if not fields.get("customer_id"):
        match = re.search(r"Kundennummer[:\s]*([A-Z0-9\-]+)", text)
        if match:
            fields["customer_id"] = match.group(1).strip()
        else:
            match = re.search(r"Az\.\s*([A-Z0-9]+\s*[/-]\s*[A-Z0-9]+)", text)
            if match:
                fields["customer_id"] = match.group(1).replace(" ", "")
            else:
                match = re.search(r"vertreten durch[^\n]+Az\.\s*([A-Z0-9/]+)", text)
                if match:
                    fields["customer_id"] = match.group(1)

    # --- INSTITUTION NAME ---
    if not fields.get("institution_name"):
        match = re.search(r"Empfänger:\s*\*\*([^\n]+)\*\*\s*\*\*([^\n]+)\*\*\s*\*\*([^\n]+)\*\*", text)
        if match:
            fields["institution_name"] = match.group(1).strip()
            fields["institution_address"] = f"{match.group(2).strip()}, {match.group(3).strip()}"
        else:
            match = re.search(r"Empfänger:\s*([^\n]+)\n\s*([^\n]+)\n\s*([^\n]+)", text)
            if match:
                fields["institution_name"] = match.group(1).strip()
                fields["institution_address"] = f"{match.group(2).strip()}, {match.group(3).strip()}"
            else:
                match = re.search(r"(Deutsche Kreditbank|N26 Bank|Commerzbank|Sparkasse)", text)
                if match:
                    fields["institution_name"] = match.group(1).strip()

    # --- INSTITUTION ADDRESS (fallback: Anschrift or general German address) ---
    if not fields.get("institution_address"):
        address_match = re.search(r"(?:Adresse|Anschrift)[:\s]*([^\n]+)", text)
        if address_match:
            raw_address = address_match.group(1).strip()
        else:
            # General German address pattern: street name + number + ZIP + city
            m = re.search(r"([A-ZÄÖÜa-zäöüß\s\-]+ \d+[a-zA-Z]?,?\s*\d{5}\s+[A-ZÄÖÜa-zäöüß\s\-]+)", text)
            raw_address = m.group(1).strip() if m else None

        if raw_address:
            fields["institution_address"] = normalize_address(raw_address)

# ---------------------- Placeholders ----------------------

def extract_general_it(text, fields):
    # 📆 Document date — handles "Data di emissione"
    match = re.search(r"(?:Data di emissione|Data)[:\s]+(\d{1,2}[.\s]+[A-Za-zàèìòù]+[.\s]+\d{4})", text, re.IGNORECASE)
    if match:
        date_str = match.group(1)
        fields["document_date"] = normalize_date(date_str, lang="it")

    # 👤 Customer name — handles "Nome e Cognome" or "Intestatario"
    match = re.search(r"(?:Nome e Cognome|Intestatario|Titolare)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
    if match:
        fields["customer_name"] = normalize_name(match.group(1))

    # 🆔 Customer ID — handles "Codice Fiscale"
      # 🧩 Fallback: if Account Number is present, use it as customer_id if not set
    if not fields.get("customer_id"):
        match = re.search(r"(?:Account Number)[:\s]*([A-Z0-9\*]{4,})", text, re.IGNORECASE)
        if match:
            account_number = match.group(1).strip()
            fields["customer_id"] = account_number
    # 🆔 Codice Fiscale → customer_id
    if not fields.get("customer_id"):
        match = re.search(r"Codice\s+Fiscale[:\s]*([A-Z0-9]{16})", text)
        if match:
            fields["customer_id"] = match.group(1).strip()
            print(f"[DEBUG] Matched Codice Fiscale: {fields['customer_id']}")

    # 🏦 Institution name — match known banks
    match = re.search(r"(UniCredit|Intesa Sanpaolo|Banca d’Italia|Banca Italiana di Credito)", text, re.IGNORECASE)
    if match:
        fields["institution_name"] = match.group(1)

    # 🏠 Institution address — from "Indirizzo" or "Sede Legale"
    match = re.search(r"(?:Indirizzo|Sede Legale)[:\s]+(.+?,\s*\d{5}\s+[^\n,]+)", text)
    if match:
        fields["institution_address"] = normalize_address(match.group(1))



# ---------------------- Placeholders ----------------------
def extract_general_en(text, fields):
    import re
    from normalize import normalize_date, normalize_name, normalize_address

        # 🆔 Customer ID — support common US terms
    match = re.search(r"(Customer ID|Client No|Account ID|Account Number|User ID|Customer Number)[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if match:
        fields["customer_id"] = match.group(2).strip()


    # 📆 Document Date — from "Statement Date: March 15, 2025"
    match = re.search(r"Statement Date[:\s]+(.+?\s+\d{4})", text, re.IGNORECASE)
    if match:
        fields["document_date"] = normalize_date(match.group(1), "en")

    # 👤 Customer Name
    match = re.search(r"Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
    if match:
        name = match.group(1).strip()
        if "address" in name.lower():
            name = name.split("Address")[0].strip()
        fields["customer_name"] = normalize_name(name)

    # 🏦 Institution Name
    match = re.search(r"(Global Financial Bank|Bank of America|Citibank|Wells Fargo)", text, re.IGNORECASE)
    if match:
        fields["institution_name"] = match.group(1)

    # 🏠 Institution Address
    address_patterns = [
        r"(?:Address|Head Office|Branch Location)[:\s]+(.+?,\s*[A-Z]{2}\s*\d{5},\s*USA)",
        r"(?:Address)[:\s]+(\d{1,5}\s+[^\n,]+,\s*\w+,\s*[A-Z]{2}\s*\d{5})"
    ]
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_address = match.group(1).strip()
            fields["institution_address"] = normalize_address(raw_address)
            break

# ---------------------- Placeholders ----------------------

def extract_general_es(text, fields):
    # 📅 Document Date
    match = re.search(r"(?:Fecha|Emitido el|Fecha de emisión)[:\s]+(\d{1,2}\s+de\s+[a-zA-Zñ]+(?:\s+de)?\s+\d{4})", text, re.IGNORECASE)
    if match:
        fields["document_date"] = normalize_date(match.group(1), "es")

    # 👤 Customer Name
    if not fields.get("customer_name"):
        match = re.search(
            r"(?:Nombre(?:\s+completo)?|Titular|Cliente)[\s:\-]+((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+|$)){2,5})(?=\s*(Dirección|Domicilio|Direcc[ií]?[oó]n)?|$)",
            text)
        if match:
            fields["customer_name"] = normalize_name(match.group(1).strip())

    # 🆔 Customer ID
    if not fields.get("customer_id"):
        match = re.search(r"(?:NIF|DNI|ID Cliente|Identificaci[oó]n)[:\s\-]*([A-Z0-9\-]{6,})", text, re.IGNORECASE)
        if match:
            fields["customer_id"] = match.group(1).strip()

    # 🏦 Institution Name — robust word-boundary match
    if not fields.get("institution_name"):
        known_institutions = [
            "Banco Hispano Internacional", "Banco Santander", "BBVA", "CaixaBank", "Bankinter",
            "Banco de España", "Agencia Tributaria", "Banco Popular", "ING"
        ]
        for name in known_institutions:
            if re.search(rf"\b{name}\b", text, re.IGNORECASE):
                fields["institution_name"] = name
                break

    # 🏠 Institution Address — clean first address from corrupted string
    if not fields.get("institution_address"):
        # Primary address match
        match = re.search(
            r"(?:Direcci[oó]n|Direcc[ií]?[oó]n|Domicilio|Sede)[:\s\-]*([\w\s\,\-]+?\d{4,5}\s+[A-Z][^\n,]+)",
            text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
        else:
            # Fallback: collect all lines with postal pattern
            raw = ""
            for line in text.splitlines():
                if re.search(r"\d{4,5}\s+[A-Z][a-z]", line):
                    raw += " " + line.strip()

        if raw:
            # 🧠 OCR Fixes
            raw = raw.replace("Ca11e", "Calle").replace("Caste1lana", "Castellana")
            raw = raw.replace("Espana", "España").replace("Macrid", "Madrid")

            # 📍 Extract only the first valid address
            address_matches = re.findall(r"[A-Z][a-z]+[\w\s]+?\d{1,3},\s*\d{4,5}\s+[A-Z][a-z]+", raw)
            if address_matches:
                fields["institution_address"] = normalize_address(address_matches[0])


# ---------------------- Placeholders ----------------------
def extract_general_fr(text, fields):

    # Fix common OCR issues (e.g., "tévrier" to "février")
    text = text.replace("tévrier", "février").replace('\xa0', ' ').replace('\u202f', ' ')

    # Month names for French
    months = r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"

    # Enhanced flexible date regex with multiple potential formats
    date_match = re.search(
        rf"Date\s*[:\-]?\s*(\d{{1,2}}\s+{months}\s+\d{{4}})",
        text,
        re.IGNORECASE | re.UNICODE,
    )

    if date_match:
        fields["document_date"] = normalize_date(date_match.group(1), "fr")
    else:
        print("[DEBUG] Still no date match found.")

    # 👤 Customer name — "Nom: Sophie Mercier"
    match = re.search(r"Nom[:\s]+([^\n]+)", text)
    if match:
        name = match.group(1).strip()
        if "adresse" in name.lower():
            name = name.split("Adresse")[0].strip()
        fields["customer_name"] = normalize_name(name)


    # 🏦 Institution name — header/footer match
    match = re.search(r"(Banque Européenne d'Investissement)", text, re.IGNORECASE)
    if match:
        fields["institution_name"] = match.group(1)

    # 🏠 Institution address — "Adresse: 45 Rue de la Paix, 75002 Paris, France"
    match = re.search(r"Adresse[:\s]+(.+?,\s*\d{5}\s+\w+,\s+\w+)", text)
    if match:
        fields["institution_address"] = normalize_address(match.group(1).strip())



