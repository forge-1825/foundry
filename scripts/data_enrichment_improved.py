#!/usr/bin/env python
"""
Data Enrichment Script with GPU Acceleration

This script processes input data from the Manual Extractor and enriches it with:
1. Text cleaning and preprocessing
2. Entity extraction
3. Keyword extraction
4. Text summarization

It supports both JSON input files and direct folder processing.
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
import re
import glob
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_enrichment.log", mode='w')
    ]
)

# Try to import GPU-related libraries
try:
    import torch
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        logging.info(f"CUDA is available with {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("CUDA is not available. Using CPU only.")
except ImportError:
    logging.warning("PyTorch not installed. GPU acceleration will be disabled.")
    has_cuda = False

# Try to import NLP libraries with fallbacks
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    stopwords_list = set(stopwords.words('english'))
    logging.info("NLTK loaded successfully")
except ImportError:
    logging.warning("NLTK not installed. Using simplified text processing.")
    nltk = None
    
    # Simple sentence tokenizer fallback
    def sent_tokenize(text):
        """Simple sentence tokenizer fallback."""
        text = text.replace('!', '.').replace('?', '.')
        sentences = []
        for s in text.split('.'):
            if s.strip():
                sentences.append(s.strip() + '.')
        return sentences
    
    # Simple stopwords fallback
    stopwords_list = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'to', 'of', 'for', 'with', 'in', 'on', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may',
        'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
        'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
        'theirs', 'am', 'who', 'whom', 'whose', 'which', 'there', 'here'
    ])

# Try to import summarization libraries
try:
    from transformers import pipeline
    summarizer = None
    if has_cuda:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
            logging.info("Loaded summarization pipeline on GPU")
        except Exception as e:
            logging.error(f"Error loading summarization model on GPU: {e}")
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
            logging.info("Loaded summarization pipeline on CPU (GPU failed)")
    else:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        logging.info("Loaded summarization pipeline on CPU")
except ImportError:
    logging.warning("Transformers not installed. Summarization will be disabled.")
    summarizer = None

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF
    logging.info("PyMuPDF loaded successfully for PDF processing")
except ImportError:
    logging.warning("PyMuPDF not installed. PDF processing will be limited.")
    fitz = None

# Try to import entity extraction libraries
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logging.info("Spacy loaded successfully for entity extraction")
except ImportError:
    logging.warning("Spacy not installed. Entity extraction will be limited.")
    nlp = None

def report_progress(current, total, message="Processing"):
    """Report progress to both console and log file."""
    progress_percent = (current / total) * 100 if total > 0 else 0
    logging.info("[PROGRESS] %s: %.2f%% (%d/%d)", message, progress_percent, current, total)
    # In a real implementation, this would send progress to the UI via an API call
    print(f"\r[PROGRESS] {message}: {progress_percent:.2f}% ({current}/{total})", end="")

def clean_text(text: str) -> str:
    """
    Clean and preprocess text by removing stopwords and normalizing.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Use spaCy for advanced cleaning if available
    if nlp:
        try:
            doc = nlp(text)
            # Remove stopwords and punctuation
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            return " ".join(tokens)
        except Exception as e:
            logging.error(f"Error cleaning text with spaCy: {e}")
    
    # Fallback to basic cleaning
    words = text.split()
    words = [word.lower() for word in words if word.lower() not in stopwords_list]
    return " ".join(words)

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract named entities from text.
    Returns a list of dictionaries with entity text, label, and position.
    """
    if not text or not isinstance(text, str):
        return []
    
    if nlp:
        try:
            doc = nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            return entities
        except Exception as e:
            logging.error(f"Error extracting entities: {e}")
    
    # Simple fallback for entity extraction
    return []

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract important keywords from text.
    """
    if not text or not isinstance(text, str):
        return []
    
    try:
        # Simple frequency-based keyword extraction
        words = text.lower().split()
        words = [word for word in words if word not in stopwords_list]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []

def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Generate a summary of the text.
    """
    if not text or not isinstance(text, str) or len(text) < 100:
        return text
    
    if summarizer:
        try:
            # Truncate text if it's too long
            if len(text) > 1024:
                text = text[:1024]
            
            summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Error summarizing text: {e}")
    
    # Fallback to extractive summarization
    sentences = sent_tokenize(text)
    if len(sentences) <= 3:
        return text
    
    # Simple extractive summarization
    return " ".join(sentences[:3])

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from a PDF file.
    """
    if not fitz:
        logging.error("PyMuPDF not installed. Cannot extract PDF content.")
        return {"text": "", "error": "PyMuPDF not installed"}
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        return {
            "text": text,
            "page_count": len(doc),
            "metadata": doc.metadata
        }
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return {"text": "", "error": str(e)}

def process_pdf_file(pdf_path: str) -> Dict[str, Any]:
    """
    Process a single PDF file and enrich its content.
    """
    logging.info(f"Processing PDF file: {pdf_path}")
    
    # Extract text from PDF
    pdf_data = extract_text_from_pdf(pdf_path)
    
    if "error" in pdf_data:
        return {
            "path": pdf_path,
            "error": pdf_data["error"]
        }
    
    # Clean and enrich the text
    text = pdf_data.get("text", "")
    cleaned_text = clean_text(text)
    entities = extract_entities(text)
    keywords = extract_keywords(text)
    summary = summarize_text(text)
    
    return {
        "path": pdf_path,
        "filename": os.path.basename(pdf_path),
        "page_count": pdf_data.get("page_count", 0),
        "metadata": pdf_data.get("metadata", {}),
        "text": text,
        "cleaned_text": cleaned_text,
        "entities": entities,
        "keywords": keywords,
        "summary": summary
    }

def process_json_file(json_path: str) -> List[Dict[str, Any]]:
    """
    Process a JSON file from the Manual Extractor and enrich its content.
    """
    logging.info(f"Processing JSON file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logging.error(f"Invalid JSON format in {json_path}. Expected a list.")
            return []
        
        enriched_records = []
        total_records = len(data)
        
        for i, record in enumerate(data):
            report_progress(i+1, total_records, "Enriching records")
            
            # Get the text content from the record
            text = ""
            if "raw_content" in record:
                text = record["raw_content"]
            elif "text" in record:
                text = record["text"]
            elif "structured_content" in record and isinstance(record["structured_content"], dict):
                # Combine all structured content
                for section, content in record["structured_content"].items():
                    text += content + "\n\n"
            
            # Skip if no text content
            if not text:
                logging.warning(f"No text content found in record {i}")
                enriched_records.append(record)  # Keep the original record
                continue
            
            # Enrich the record
            cleaned_text = clean_text(text)
            entities = extract_entities(text)
            keywords = extract_keywords(text)
            summary = summarize_text(text)
            
            # Create enriched record
            enriched_record = record.copy()  # Keep all original fields
            enriched_record.update({
                "cleaned_text": cleaned_text,
                "entities": entities,
                "keywords": keywords,
                "summary": summary
            })
            
            enriched_records.append(enriched_record)
        
        return enriched_records
    except Exception as e:
        logging.error(f"Error processing JSON file {json_path}: {e}")
        return []

def process_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a folder and enrich their content.
    """
    logging.info(f"Processing folder: {folder_path}")
    
    # Find all PDF files in the folder
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {folder_path}")
        return []
    
    logging.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
    
    # Process each PDF file
    enriched_records = []
    for i, pdf_file in enumerate(pdf_files):
        report_progress(i+1, len(pdf_files), "Processing PDF files")
        record = process_pdf_file(pdf_file)
        enriched_records.append(record)
    
    return enriched_records

def main():
    """
    Main function to process input data and generate enriched output.
    """
    parser = argparse.ArgumentParser(description="Data Enrichment with GPU Acceleration")
    parser.add_argument("--input-file", help="Path to the input JSON file from Manual Extractor")
    parser.add_argument("--output-file", help="Path to save the enriched output JSON file")
    parser.add_argument("--input-folder", help="Path to the folder containing PDF files")
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_file and not args.input_folder:
        logging.error("Either --input-file or --input-folder must be provided")
        parser.print_help()
        sys.exit(1)
    
    if not args.output_file:
        logging.error("--output-file must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    
    # Process input data
    enriched_records = []
    
    if args.input_file and os.path.exists(args.input_file):
        logging.info(f"Processing input file: {args.input_file}")
        enriched_records = process_json_file(args.input_file)
    elif args.input_folder and os.path.exists(args.input_folder):
        logging.info(f"Processing input folder: {args.input_folder}")
        enriched_records = process_folder(args.input_folder)
    else:
        logging.error("Input file or folder not found")
        sys.exit(1)
    
    # Save enriched data to output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_records, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(enriched_records)} enriched records to {args.output_file}")
    except Exception as e:
        logging.error(f"Error saving enriched data to {args.output_file}: {e}")
        sys.exit(1)
    
    # Print summary
    print("\nData Enrichment Summary:")
    print(f"- Total records processed: {len(enriched_records)}")
    print(f"- Entities extracted: {sum(len(record.get('entities', [])) for record in enriched_records)}")
    print(f"- Keywords extracted: {sum(len(record.get('keywords', [])) for record in enriched_records)}")
    print(f"- Output file: {args.output_file}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info(f"Total processing time: {elapsed_time:.2f} seconds")
