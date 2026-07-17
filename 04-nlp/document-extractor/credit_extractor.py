import re
from normalize import normalize_money, normalize_date

def extract_credit_fields(text, language):
    handlers = {
        "en": extract_credit_en,
        "de": extract_credit_de,
        "fr": extract_credit_fr,
        "es": extract_credit_es,
        "it": extract_credit_it,
    }
    handler = handlers.get(language, extract_credit_en)
    return handler(text, language)

# --------------------- LANGUAGE HANDLERS ---------------------

def extract_credit_en(text, lang): 
    return _extract_common_credit_fields(text, lang, {
        "card_number": r"(?:Card\s*Number|Reference\s*Number\s*Account Number)",
        "credit_limit": r"Credit\s*Limit",
        "interest_rate": r"(?:Interest\s*Rate|APR|Annual\s*Percentage\s*Yield|APY|Interest\s*Rate)",
        "payment_due_date": r"Due\s*Date",
        "statement_period": r"Statement\s*Period",
        "minimum_payment": r"Minimum\s*Payment",
        "previous_balance": r"Previous\s*Balance",
        "new_balance": r"(?:New|Current)\s*Balance"
    })

def extract_credit_de(text, lang): 
    return _extract_common_credit_fields(text, lang, {
        "card_number": r"(?:Karten\s*nummer|Referenz\s*nummer|Konto\s*Nr)",
        "credit_limit": r"(?:Kredit\s*limit|Kredit\s*rahmen|Kredit\s*grenze)",
        "interest_rate": r"(?:Zins\s*satz|Effektiver\s*Jahres\s*zins)",
        "payment_due_date": r"(?:Fälligkeits\s*datum|Zahlungs\s*ziel)",
        "statement_period": r"Abrechnungs\s*zeitraum",
        "minimum_payment": r"(?:Mindest\s*zahlung|Minimal\s*betrag)",
        "previous_balance": r"(?:Vorheriger\s*Konto\s*stand|Letzter\s*Saldo)",
        "new_balance": r"(?:Neuer\s*Konto\s*stand|Aktueller\s*Saldo|Betrag)"
    })


def extract_credit_fr(text, lang): 
    # First try credit card fields
    card_fields = _extract_common_credit_fields(text, lang, {
        "card_number": r"Num[eé]ro de carte",
        "credit_limit": r"(?:Limite de cr[eé]dit|Cr[eé]dit maximum)",
        "interest_rate": r"Taux d[’']int[eé]r[eê]t",
        "payment_due_date": r"Date d[’']?[eé]ch[eé]ance",
        "statement_period": r"P[eé]riode de relev[eé]",
        "minimum_payment": r"Paiement minimum",
        "previous_balance": r"Solde pr[eé]c[eé]dent",
        "new_balance": r"Solde nouveau"
    })
    
    if any(card_fields.values()):  # If credit card fields found
        return card_fields
    
    # Fallback to loan-specific fields
    return {
        "credit_limit": _search_money(r"Montant du prêt[\s:\-]*([\d\s.,]+)", text, lang),
        "interest_rate": _search_percent(r"Taux d['’]intérêt[\s:\-]*([\d.,]+)", text),
        "payment_due_date": _search_date(r"Date du premier remboursement[\s:\-]*(.+)", text, lang),
        "statement_period": _search(r"Durée du prêt[\s:\-]*(\d+\s*mois)", text),
        "minimum_payment": _search_money(r"Mensualité[\s:\-]*([\d\s.,]+)", text, lang),
        "new_balance": _search_money(r"Montant total à rembourser[\s:\-]*([\d\s.,]+)", text, lang)
    }

def extract_credit_es(text, lang): return _extract_common_credit_fields(text, lang, {
    "card_number": r"N[uú]mero de (?:tarjeta|referencia)",
    "credit_limit": r"(?:L[ií]mite de Cr[eé]dito|Cr[eé]dito m[aá]ximo)",
    "interest_rate": r"Tasa de Inter[eé]s(?: Anual)?",
    "payment_due_date": r"(?:Fecha l[ií]mite de pago|Fecha de vencimiento)",
    "statement_period": r"Periodo de estado",
    "minimum_payment": r"Pago m[ií]nimo",
    "previous_balance": r"Saldo anterior",
    "new_balance": r"Saldo actual"
})

def extract_credit_it(text, lang): return _extract_common_credit_fields(text, lang, {
    "card_number": r"Numero di (?:carta|conto)",
    "credit_limit": r"(?:Limite di credito|Credito massimo)",
    "interest_rate": r"Tasso di interesse",
    "payment_due_date": r"Scadenza pagamento",
    "statement_period": r"Periodo di rendiconto",
    "minimum_payment": r"Pagamento minimo",
    "previous_balance": r"Saldo precedente",
    "new_balance": r"Saldo nuovo"
})

# --------------------- COMMON FIELD EXTRACTION ---------------------

def _extract_common_credit_fields(text, lang, labels):
    fields = {}

    if "card_number" in labels:
        # Match masked cards (****-****-****-1234) or full numbers
        matches = re.findall(
            rf"{labels['card_number']}[\s:\-]*([*]{{4}}[- ]?[*]{{4}}[- ]?[*]{{4}}[- ]?\d{{4}}|\d{{4}}[- ]?\d{{4}}[- ]?\d{{4}}[- ]?\d{{4}})",
            text,
            re.IGNORECASE
        )
        
        if matches:
            # Store first match (remove separators for consistency)
            fields["card_number"] = matches[0].replace("-", "").replace(" ", "")
            print(f"[SECURE] Found card number ending with: {fields['card_number'][-4:]}")

    if "credit_limit" in labels:
        fields["credit_limit"] = _search_money(rf"{labels['credit_limit']}[\s:\-\.]*[\u20ac$£]?\s*([\d.,]+)", text, lang)

    if "interest_rate" in labels:
        fields["interest_rate"] = _search_percent(rf"{labels['interest_rate']}[\s:\-\.]*([\d.,]+)", text)
    if rate_match := re.search(r"Interest\s+rate\s+on\s+savings:\s*([\d\.]+)%", text):
        fields["interest_rate"] = f"{rate_match.group(1)}%"    

    if "payment_due_date" in labels:
        fields["payment_due_date"] = _search_date(rf"{labels['payment_due_date']}[\s:\-\.]+(.{{5,30}}?)", text, lang)

    if "statement_period" in labels:
        fields["statement_period"] = _search(rf"{labels['statement_period']}[\s:\-\.]+(.+?)(?:\n|$)", text)

    if "minimum_payment" in labels:
        fields["minimum_payment"] = _search_money(rf"{labels['minimum_payment']}[\s:\-\.]*[\u20ac$£]?\s*([\d.,]+)", text, lang)

    if "previous_balance" in labels:
        fields["previous_balance"] = _search_money(rf"{labels['previous_balance']}[\s:\-\.]*[\u20ac$£]?\s*([\d.,]+)", text, lang)

    if "new_balance" in labels:
        fields["new_balance"] = _search_money(rf"{labels['new_balance']}[\s:\-\.]*[\u20ac$£]?\s*([\d.,]+)", text, lang)
        if not fields["new_balance"]:
            # Fallback using raw currency presence
            fallback = re.search(r"[\u20ac$£]\s*([\d.,]+).*?(balance|saldo|Kontostand)", text, re.IGNORECASE)
            if fallback:
                fields["new_balance"] = normalize_money(fallback.group(1), lang)

    return {k: v for k, v in fields.items() if v}

# --------------------- UTILITY FUNCTIONS ---------------------

def _search(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def _search_money(pattern, text, lang):
    match = re.search(pattern, text, re.IGNORECASE)
    return normalize_money(match.group(1), lang) if match else None

def _search_percent(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return f"{match.group(1).replace(',', '.').strip()} %" if match else None

def _search_date(pattern, text, lang):
    match = re.search(pattern, text, re.IGNORECASE)
    return normalize_date(match.group(1), lang) if match else None
