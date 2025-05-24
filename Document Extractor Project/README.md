# Document Extractor

## 🔍 Overview

This project provides a multilingual document extraction pipeline capable of parsing financial documents in **five languages** (EN, DE, FR, ES, IT) and **four document types**:

* **Credit**
* **Garnishment**
* **Investment**
* **Personal Account**

It extracts relevant structured information and returns a clean, standardized JSON output suitable for downstream processes.

---

## ✅ Requirements

* Python 3.8 or newer
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDFs)
* MacOS or Linux-like environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Tesseract OCR:

```bash
# Mac (Homebrew)
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr
```

---

## 🚀 Usage

To extract fields from a PDF:

```bash
python document_extractor.py ./path/to/document.pdf
```

This will:

* Run OCR (if needed)
* Detect language
* Infer document type
* Extract general and specific fields
* Output a cleaned JSON file to the `new_output/` folder

---

## 📊 Output Format

Each output JSON file includes:

* `general`: document-level fields
* One additional section based on inferred document type:

  * `credit`, `garnishment`, `investment`, or `personal_account`

### ✏️ Field Formatting Rules

| Type              | Format Example                       |
| ----------------- | ------------------------------------ |
| Dates             | `dd.mm.yyyy`                         |
| Names             | `First Last`                         |
| Addresses         | `StreetName 12, City 12345, Country` |
| Money             | `1234.56 €`                          |
| Other Text Fields | As-is from document                  |

---

## 📊 Evaluation Strategy

A Jupyter notebook (`notebooks/testing_and_evaluation.ipynb`) contains:

* OCR vs. text extraction testing
* Multilingual robustness testing
* Field accuracy checks (precision, recall)
* Inference validation using `field_based_inference.py`

---

## 📄 Packaging Notes

All logic is modular:

* `document_extractor.py`: entry script
* Individual extractors per doc type and language
* Reusable normalization logic in `normalize.py`

### ⌛ Runtime

Supports batch or single file execution. Suitable for CLI or REST API integration.

---

## 📈 Integration Plan (Production Readiness)

To integrate into a live pipeline:

1. Wrap `document_extractor.py` into a REST API (e.g. FastAPI)
2. Log OCR raw output for debugging
3. Add batch-processing capability
4. Add PDF validation pre-checks (corruption, password lock)
5. Persist JSON output into database or message queue
6. Add monitoring hooks for extraction confidence levels
