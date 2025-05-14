# Data Enrichment Script

## Overview

The `data_enrichment_improved.py` script is designed to process and enrich data from the Manual Extractor or directly from PDF files. It adds valuable metadata, extracts entities, identifies keywords, and generates summaries using GPU acceleration when available.

## Features

- **Multiple Input Sources**: Process JSON files from Manual Extractor or folders containing PDF files
- **GPU Acceleration**: Utilizes CUDA for faster processing when available
- **Text Cleaning**: Removes stopwords, normalizes text, and improves readability
- **Entity Extraction**: Identifies named entities (people, organizations, locations, etc.)
- **Keyword Extraction**: Extracts important keywords from the text
- **Text Summarization**: Generates concise summaries of the content
- **Progress Reporting**: Provides real-time progress updates

## Usage

### Command Line Arguments

```
python data_enrichment_improved.py [options]
```

Options:
- `--input-file PATH`: Path to the input JSON file from Manual Extractor
- `--output-file PATH`: Path to save the enriched output JSON file
- `--input-folder PATH`: Path to the folder containing PDF files

### Examples

1. Process a JSON file from Manual Extractor:
   ```
   python data_enrichment_improved.py --input-file extracted_data.json --output-file enriched_data.json
   ```

2. Process a folder containing PDF files:
   ```
   python data_enrichment_improved.py --input-folder PDFs --output-file enriched_data.json
   ```

## Integration with UI

This script is integrated with the Model Distillation UI and can be executed from the Scripts page. The UI provides a user-friendly interface for configuring and running the script with different input sources.

### UI Parameters

- **Input File**: Path to the extracted data JSON file from Manual Extractor
- **Output File**: Path to save the enriched data
- **Source Folder**: Path to the folder containing PDF files (alternative to Input File)

## Requirements

- Python 3.6+
- PyTorch (for GPU acceleration)
- Transformers (for summarization)
- Spacy (for entity extraction)
- PyMuPDF (for PDF processing)
- NLTK (for text processing)

## Output Format

The script generates a JSON file containing enriched records with the following fields:

- **path**: Path to the original file
- **filename**: Name of the original file
- **text**: Original text content
- **cleaned_text**: Preprocessed and cleaned text
- **entities**: List of extracted named entities
- **keywords**: List of important keywords
- **summary**: Concise summary of the content
- **metadata**: Additional metadata from the source

## Notes

- GPU acceleration requires a CUDA-compatible GPU and PyTorch with CUDA support
- For large PDF files, the script may use significant memory
- The quality of entity extraction and summarization depends on the installed models
