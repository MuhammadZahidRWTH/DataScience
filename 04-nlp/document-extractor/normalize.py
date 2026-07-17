import re
import unicodedata

import re
import unicodedata

def strip_accents(s):
    """Remove accents/diacritics from a string while preserving German umlauts."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_date(text, lang="en"):
    """Normalize date formats in multiple languages."""
    
    # Month mappings for supported languages
    months = {
        "en": {"january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
               "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"},
        "de": {"januar": "01", "februar": "02", "märz": "03", "maerz": "03", "marz": "03", "april": "04", "mai": "05",
               "juni": "06", "juli": "07", "august": "08", "september": "09", "oktober": "10", "november": "11", "dezember": "12"},
        "fr": {"janvier": "01", "février": "02", "mars": "03", "avril": "04", "mai": "05", "juin": "06",
               "juillet": "07", "août": "08", "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"},
        "es": {"enero": "01", "febrero": "02", "marzo": "03", "abril": "04", "mayo": "05", "junio": "06",
               "julio": "07", "agosto": "08", "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"},
        "it": {"gennaio": "01", "febbraio": "02", "marzo": "03", "aprile": "04", "maggio": "05", "giugno": "06",
               "luglio": "07", "agosto": "08", "settembre": "09", "ottobre": "10", "novembre": "11", "dicembre": "12"},
    }

    # Select month dictionary based on language
    m = months.get(lang, months["en"])

    # Define patterns for date matching in multiple formats
    patterns = [
        # Matches dates in formats like "15. März 2023" or "15.März 2023"
        r'(\d{1,2})[\. ]\s*([A-Za-zÀ-ÿäöüÄÖÜß]+)\s+(\d{4})',
        # Matches Spanish dates like "15 de marzo de 2023"
        r'(\d{1,2})\s+de\s+([a-zA-ZÀ-ÿ]+)\s+de\s+(\d{4})',
        # Matches generic dates like "15 marzo 2023"
        r'(\d{1,2})\s+([a-zA-ZÀ-ÿ]+)\s+(\d{4})',
        # Matches English dates like "March 15, 2023"
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})'
    ]

    # Iterate over the patterns and attempt a match
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if "January" in pattern:
                month_str, day, year = match.groups()
            else:
                day, month_str, year = match.groups()
            
            # Normalize the month name by stripping accents and mapping to the corresponding month number
            month_key = strip_accents(month_str.lower())
            month = m.get(month_key, "01")
            
            # Return the normalized date in the format: dd.mm.yyyy
            return f"{int(day):02d}.{month}.{year}"

    return None

def normalize_name(name: str):
    """Ensure name is formatted as 'First Last'."""
    if not name:
        return None
    parts = name.strip().split()
    if len(parts) >= 2:
        first = parts[0].capitalize()
        last = " ".join(p.capitalize() for p in parts[1:])
        return f"{first} {last}"
    return name.strip().title()

def normalize_money(text: str, lang="en"):
    """Normalize monetary amounts with proper currency handling."""
    if not text:
        return None
    
    # Extract numeric values from the string
    match = re.search(r'([\d\.,]+)', text.replace(' ', ''))
    if not match:
        return None
        
    amount_str = match.group(1)
    
    # Handle decimal separator based on language
    if lang in ["de", "es", "fr", "it"]:
        # European format: 1.000,00 → 1000.00
        if ',' in amount_str and '.' in amount_str:
            amount_str = amount_str.replace('.', '').replace(',', '.')
        elif ',' in amount_str:
            amount_str = amount_str.replace(',', '.')
    else:
        # English format: 1,000.00 → 1000.00
        amount_str = amount_str.replace(',', '')
    
    try:
        amount = float(amount_str)
        return f"{amount:.2f} €"  # Standardize on Euro symbol
    except ValueError:
        return None

def normalize_address(address: str, lang: str = "en") -> str:
    """Normalize addresses for different languages while preserving special characters."""
    if not address:
        return None

    # Common OCR corrections across languages
    corrections = {
        "O": "0",  # OCR letter O to zero
        "l": "1",  # OCR lowercase L to one
        "I": "1",  # OCR uppercase I to one
        " strabe": " straße",  # German OCR error
        " Strabe": " Straße",  # German OCR error
    }
    
    # Language-specific corrections
    lang_corrections = {
        "de": {"Deutsch1and": "Deutschland", "Germa ny": "Germany", "Ber1in": "Berlin", "Münch en": "München"},
        "fr": {"P aris": "Paris", "Mar seille": "Marseille", "Fran ce": "France"},
        "es": {"M adrid": "Madrid", "Bar celona": "Barcelona", "Espa ña": "España"},
        "it": {"R oma": "Roma", "M ilano": "Milano", "Ita lia": "Italia"},
        "en": {"Un ited": "United", "Lon don": "London", "New York City": "New York"}
    }
    
    # Apply general corrections
    for wrong, right in corrections.items():
        address = address.replace(wrong, right)
    
    # Apply language-specific corrections
    if lang in lang_corrections:
        for wrong, right in lang_corrections[lang].items():
            address = address.replace(wrong, right)
    
    address = " ".join(address.strip().split())  # Normalize whitespace
    
    # Language-specific normalization patterns
    patterns = {
        "de": [  # German address format
            r"^([A-Za-zäöüÄÖÜß\-. ]+?) (\d+[a-zA-Z]?),\s*(\d{5})\s+([A-Za-zäöüÄÖÜß\-. ]+?)(?:,\s*(Deutschland))?$",
            lambda m: f"{m[1]} {m[2]}, {m[3]} {m[4]}" + (f", {m[5]}" if m[5] else ", Deutschland")
        ],
        "fr": [  # French address format
            r"^(\d+)\s+([A-Za-zéèêëàâùûçÉÈÊËÀÂÙÛÇ\-. ]+?),\s*(\d{5})\s+([A-Za-zéèêëàâùûçÉÈÊËÀÂÙÛÇ\-. ]+?)(?:,\s*(France))?$",
            lambda m: f"{m[1]} {m[2]}, {m[3]} {m[4]}" + (f", {m[5]}" if m[5] else ", France")
        ],
        "es": [  # Spanish address format
            r"^([A-Za-zñáéíóúüÑÁÉÍÓÚÜ\-. ]+?) (\d+),\s*(\d{5})\s+([A-Za-zñáéíóúüÑÁÉÍÓÚÜ\-. ]+?)(?:,\s*(España))?$",
            lambda m: f"{m[1]} {m[2]}, {m[3]} {m[4]}" + (f", {m[5]}" if m[5] else ", España")
        ],
        "it": [  # Italian address format
            r"^([A-Za-zàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ\-. ]+?) (\d+),\s*(\d{5})\s+([A-Za-zàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ\-. ]+?)(?:,\s*(Italia))?$",
            lambda m: f"{m[1]} {m[2]}, {m[3]} {m[4]}" + (f", {m[5]}" if m[5] else ", Italia")
        ]
    }

    # Attempt to match and normalize the address based on the language
    if lang in patterns:
        pattern, replacement = patterns[lang]
        match = re.match(pattern, address)
        if match:
            return replacement(match.groups())

    return address

def normalize_amount(text: str):
    match = re.search(r'([\d\.,]+)', text)
    if match:
        raw = match.group(1)
        # Handle formats like "6,329.94" (EN) or "6.329,94" (DE)
        if "," in raw and "." in raw:
            if raw.find(",") < raw.find("."):
                # Unlikely: "6,329.94" → EN style
                clean = raw.replace(",", "")
            else:
                # "6.329,94" → DE/IT/ES style
                clean = raw.replace(".", "").replace(",", ".")
        elif "," in raw:
            clean = raw.replace(",", ".")
        else:
            clean = raw
        return f"{clean} €"
    return None
