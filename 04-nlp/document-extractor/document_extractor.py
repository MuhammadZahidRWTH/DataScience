import os
import json
import sys

# Import necessary functions from modules
from extractor_utils import extract_text_from_pdf, detect_language
from general_extractor import extract_general_fields
from credit_extractor import extract_credit_fields
from personal_account_extractor import extract_personal_account_fields
from investment_extractor import extract_investment_fields
from garnishment_extractor import extract_garnishment_fields
from field_based_inference import infer_document_type


def filter_fields_by_type(fields, doc_type):
    """
    Filters the fields to include only general fields and those specific to the document type,
    while excluding fields with null or empty values.

    Args:
        fields (dict): The extracted fields from the document.
        doc_type (str): The type of document (e.g., "investment", "personal_account").

    Returns:
        dict: The filtered fields.
    """
    
    def is_valid(value):
        """Check if the field value is not null, empty, or just whitespace."""
        return value not in [None, "", " "]

    # General fields that are always included
    general_fields = ["document_id", "document_type", "document_date", "customer_name", "customer_id", 
                      "institution_name", "institution_address", "language"]
    
    # Document type specific fields (based on the document type detected)
    document_type_fields = {
        "investment": ["portfolio_id", "portfolio_value", "asset_number", "risk_profile"],
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
        # Add other types if needed
    }
    
    # Start with general fields
    filtered_fields = {key: value for key, value in fields.items() if key in general_fields and is_valid(value)}
    
    # Add document type specific fields if they exist
    if doc_type in document_type_fields:
        for field in document_type_fields[doc_type]:
            if field in fields and is_valid(fields[field]):
                filtered_fields[field] = fields[field]
    
    return filtered_fields


def process_document(file_path):
    """
    Processes the document by extracting text, detecting language, extracting fields,
    inferring document type, and saving the structured output.

    Args:
        file_path (str): The path to the document (PDF).
    """
    print(f"[•] Processing: {file_path}")

    # Step 1: Extract text from the PDF document
    text = extract_text_from_pdf(file_path)
    if not text:
        print("[!] No text extracted — check if OCR fallback is working.")

    # Step 2: Detect the language of the extracted text
    language = detect_language(text) or "unknown"

    # Step 3: Extract general fields from the text
    fields = extract_general_fields(text, language)

    # Step 4: Extract document-specific fields
    fields.update(extract_credit_fields(text, language))
    fields.update(extract_investment_fields(text, language))
    fields.update(extract_personal_account_fields(text, language))
    fields.update(extract_garnishment_fields(text, language))

    # Step 5: Infer the document type from the extracted fields
    inferred_type = infer_document_type(fields, raw_text=text)
    fields["document_type"] = inferred_type

    # Step 6: Add language to the fields
    fields["language"] = language

    # Step 7: Filter fields based on document type
    filtered_fields = filter_fields_by_type(fields, inferred_type)

    # Step 8: Restructure fields into grouped output
    output = {
        "file_name": os.path.basename(file_path),
        "general": {key: filtered_fields[key] for key in filtered_fields if key in [
            "document_id", "document_type", "document_date", "customer_name", 
            "customer_id", "institution_name", "institution_address", "language"]},
        inferred_type: {key: filtered_fields[key] for key in filtered_fields if key not in [
            "document_id", "document_type", "document_date", "customer_name", 
            "customer_id", "institution_name", "institution_address", "language"]}
    }

    # Step 9: Save the structured output to a new folder
    output_dir = "outputs"  # Define the folder name for saving output
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Saved result to {output_path}")


if __name__ == "__main__":
    # Ensure that a file path is provided via command line arguments
    if len(sys.argv) < 2:
        print("Usage: python document_extractor.py <path_to_pdf>")
        sys.exit(1)

    # Process the provided document
    process_document(sys.argv[1])
