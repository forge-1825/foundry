import os
import sys
import json
import time
import logging
import concurrent.futures
from tqdm import tqdm
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Try to import required packages with fallbacks
try:
    import fitz  # PyMuPDF for PDF extraction
except ImportError:
    logging.error("PyMuPDF (fitz) not installed. Please run install_dependencies.py first.")
    fitz = None

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    logging.error("spaCy not installed. Please run install_dependencies.py first.")
    spacy = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    logging.error("pytesseract or PIL not installed. OCR functionality will be disabled.")
    pytesseract = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    logging.error("transformers not installed. Summarization will be disabled.")
    pipeline = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    logging.error("scikit-learn not installed. Keyword extraction will be limited.")
    TfidfVectorizer = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.error("sentence-transformers not installed. Semantic similarity will be disabled.")
    SentenceTransformer = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
except ImportError:
    logging.error("NLTK not installed. Text processing will be limited.")
    nltk = None
    sent_tokenize = lambda text: text.split('. ')  # Simple fallback
    stopwords = None

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
# Add file handler only if we can write to the directory
try:
    log_file = "/tmp/enrichment.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logging to file: {log_file}")
except Exception as e:
    logging.warning(f"Could not set up file logging: {e}")

# ------------------------------
# Download NLTK resources if available
# ------------------------------
if nltk:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logging.info("Downloaded NLTK resources")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        logging.warning("Some text processing functions may not work properly.")
else:
    logging.warning("NLTK not available. Using simplified text processing.")

# ------------------------------
# Load spaCy model if available - using larger model for better accuracy
# ------------------------------
nlp = None
if spacy:
    try:
        # Use the larger model for better accuracy if available
        try:
            nlp = spacy.load("en_core_web_trf")  # Transformer-based model (most accurate)
            logging.info("Loaded spaCy model: en_core_web_trf (transformer-based)")
        except:
            try:
                nlp = spacy.load("en_core_web_lg")  # Large model (more accurate than sm)
                logging.info("Loaded spaCy model: en_core_web_lg")
            except:
                try:
                    nlp = spacy.load("en_core_web_sm")  # Small model (fallback)
                    logging.info("Loaded spaCy model: en_core_web_sm (fallback)")
                except:
                    logging.error("No spaCy model available. Please run: python -m spacy download en_core_web_sm")
                    nlp = None
        
        # Add pipeline components if model loaded successfully
        if nlp and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        
        # Optionally add coreference resolution if neuralcoref is installed
        try:
            import neuralcoref
            if nlp and "neuralcoref" not in nlp.pipe_names:
                neuralcoref.add_to_pipe(nlp)
                logging.info("Added coreference resolution to spaCy pipeline")
        except ImportError:
            logging.warning("neuralcoref not installed, skipping coreference resolution")
        
    except Exception as e:
        logging.error(f"Error loading spaCy model: {e}")
        nlp = None

if not nlp:
    logging.warning("NLP processing will be limited without a spaCy model.")

# ------------------------------
# Load summarization model if available
# ------------------------------
summarizer = None
if pipeline:
    try:
        # Use BART-large-CNN for summarization
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        logging.info("Loaded summarization pipeline with facebook/bart-large-cnn")
    except Exception as e:
        logging.error(f"Error initializing summarization pipeline: {e}")
        logging.warning("Summarization will be disabled.")
else:
    logging.warning("Transformers not available. Summarization will be disabled.")

# ------------------------------
# Load sentence transformer for semantic similarity if available
# ------------------------------
sentence_model = None
if SentenceTransformer:
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
    except Exception as e:
        logging.error(f"Error loading sentence transformer model: {e}")
        logging.warning("Semantic similarity features will be disabled.")
else:
    logging.warning("SentenceTransformer not available. Semantic similarity features will be disabled.")

# ------------------------------
# Text preprocessing functions
# ------------------------------
def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing headers, footers, page numbers, and other noise.
    """
    # Remove page numbers (various formats)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', text)
    
    # Remove common headers and footers
    text = re.sub(r'\n\s*CONFIDENTIAL\s*\n', '\n', text)
    text = re.sub(r'\n\s*DRAFT\s*\n', '\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove URLs (simplified pattern)
    text = re.sub(r'https?://\S+', '', text)
    
    return text.strip()

def clean_text(text: str) -> str:
    """
    Lemmatize text and remove stopwords/punctuation using spaCy if available.
    Returns a cleaned version of the text.
    """
    if not text:
        return ""
        
    if nlp:
        try:
            # Process with spaCy
            doc = nlp(text)
            
            # Get lemmatized tokens, excluding stopwords and punctuation
            tokens = [token.lemma_ for token in doc 
                    if not token.is_stop and not token.is_punct 
                    and not token.is_space and len(token.text.strip()) > 1]
            
            return " ".join(tokens)
        except Exception as e:
            logging.error(f"Error cleaning text with spaCy: {e}")
            # Fall back to basic cleaning
    
    # Basic cleaning fallback if spaCy is not available
    try:
        # Simple tokenization by whitespace
        words = text.split()
        
        # Remove common punctuation
        words = [word.strip('.,;:!?()[]{}"\'-') for word in words]
        
        # Remove empty strings and short words
        words = [word for word in words if word and len(word) > 1]
        
        # Remove stopwords if available
        if stopwords:
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word.lower() not in stop_words]
            
        return " ".join(words)
    except Exception as e:
        logging.error(f"Error in fallback text cleaning: {e}")
        return text

def smart_text_chunking(text: str, max_tokens: int = 1024) -> List[str]:
    """
    Split text into chunks that respect sentence boundaries.
    This is more intelligent than simple word-based chunking.
    """
    if not text:
        return []
        
    try:
        # Split text into sentences using NLTK if available
        if nltk and sent_tokenize:
            sentences = sent_tokenize(text)
        else:
            # Simple sentence splitting fallback
            sentences = [s.strip() + '.' for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Approximate token count (words + punctuation)
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed the limit, start a new chunk
            if current_length + sentence_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        logging.error(f"Error in smart text chunking: {e}")
        # Fallback to simple chunking
        words = text.split()
        return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

# ------------------------------
# Entity extraction and filtering
# ------------------------------
def extract_entities(text: str, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using spaCy with confidence filtering.
    Returns a list of dictionaries with entity text, label, and confidence score.
    """
    if not text or not nlp:
        return []
        
    try:
        doc = nlp(text)
        
        # Extract entities with confidence scores
        entities = []
        for ent in doc.ents:
            # Estimate confidence based on entity length and context
            # This is a simplified approach since spaCy doesn't provide confidence scores directly
            confidence = min(1.0, 0.5 + (len(ent.text.split()) / 10))
            
            if confidence >= confidence_threshold:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": round(confidence, 2)
                })
        
        return entities
    except Exception as e:
        logging.error(f"Error extracting entities: {e}")
        return []

def categorize_entities(entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize entities by type and merge similar entities.
    """
    try:
        categories = {}
        
        for entity in entities:
            label = entity["label"]
            if label not in categories:
                categories[label] = []
            
            # Check if similar entity already exists in this category
            is_duplicate = False
            for existing in categories[label]:
                if existing["text"].lower() == entity["text"].lower():
                    # Update with higher confidence if applicable
                    if entity["confidence"] > existing["confidence"]:
                        existing.update(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                categories[label].append(entity)
        
        return categories
    except Exception as e:
        logging.error(f"Error categorizing entities: {e}")
        return {"ERROR": entities}

# ------------------------------
# Advanced summarization
# ------------------------------
def summarize_text(text: str, max_chunk_tokens: int = 1024) -> str:
    """
    Generate a summary using the Hugging Face summarization pipeline.
    Uses smart chunking to preserve sentence boundaries.
    """
    if not text or not summarizer:
        # If summarizer is not available, return a simple truncated version
        if text:
            words = text.split()
            if len(words) > 100:
                return " ".join(words[:100]) + "..."
            return text
        return ""
    
    try:
        if not text.strip():
            return ""
        
        # Use smart chunking to split text
        chunks = smart_text_chunking(text, max_chunk_tokens)
        
        if len(chunks) == 1:
            # For short texts, summarize directly
            return summarizer(chunks[0], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        else:
            # For longer texts, summarize each chunk and then combine
            chunk_summaries = []
            for idx, chunk in enumerate(chunks, 1):
                try:
                    summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                    chunk_summaries.append(summary)
                    logging.info(f"Chunk {idx}/{len(chunks)} summarized successfully.")
                except Exception as e:
                    logging.error(f"Error summarizing chunk {idx}: {e}")
            
            # If we have multiple summaries, combine them and summarize again
            if len(chunk_summaries) > 1:
                combined = " ".join(chunk_summaries)
                # If combined summary is still too long, summarize it again
                if len(combined.split()) > max_chunk_tokens:
                    return summarizer(combined, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                return combined
            elif len(chunk_summaries) == 1:
                return chunk_summaries[0]
            else:
                return ""
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        # Return a simple truncated version as fallback
        words = text.split()
        if len(words) > 100:
            return " ".join(words[:100]) + "..."
        return text

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract important keywords from text using TF-IDF if available,
    or frequency-based extraction as fallback.
    """
    if not text:
        return []
        
    try:
        # Get stopwords if available
        if stopwords:
            stop_words = set(stopwords.words('english'))
        else:
            # Basic stopwords list as fallback
            stop_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 
                             'on', 'with', 'as', 'was', 'be', 'this', 'are', 'by', 'an', 'at'])
        
        # Tokenize and clean
        words = [word.lower() for word in re.findall(r'\b\w+\b', text) 
                if word.lower() not in stop_words and len(word) > 2]
        
        # If text is too short, return the words directly
        if len(words) < 20:
            return list(set(words))[:top_n]
        
        # Use TF-IDF to find important words if available
        if TfidfVectorizer:
            try:
                vectorizer = TfidfVectorizer(max_features=100)
                
                # Split text into sentences for better TF-IDF results
                if nltk and sent_tokenize:
                    sentences = sent_tokenize(text)
                else:
                    # Simple sentence splitting fallback
                    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
                
                if len(sentences) < 3:  # Need at least a few sentences
                    sentences = [text]  # Use whole text as one sentence
                    
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Sum TF-IDF scores across all sentences
                scores = np.sum(tfidf_matrix.toarray(), axis=0)
                
                # Get top keywords
                top_indices = scores.argsort()[-top_n:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                
                return keywords
            except Exception as e:
                logging.error(f"TF-IDF keyword extraction failed: {e}")
                # Fall back to frequency-based extraction
        
        # Fallback: frequency-based keyword extraction
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []

# ------------------------------
# Advanced PDF extraction
# ------------------------------
def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text, tables, and images from a PDF file using PyMuPDF.
    Returns a dictionary with structured content.
    """
    if not fitz:
        logging.error("PyMuPDF (fitz) is not installed. Cannot extract PDF content.")
        return {"text": "", "pages": [], "metadata": {}, "error": "PyMuPDF not installed"}
    
    try:
        doc = fitz.open(pdf_path)
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "has_images": False,
            "has_tables": False
        }
        
        # Extract document metadata
        metadata = doc.metadata
        if metadata:
            result["metadata"] = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
            }
        
        # Process each page
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            page_dict = {"page_number": page_num + 1, "text": page_text}
            
            # Check for tables (simplified detection)
            if re.search(r'\|[-+]+\|', page_text) or re.search(r'[+][-+]+[+]', page_text):
                result["has_tables"] = True
            
            # Extract images if present and OCR is available
            images = []
            image_list = page.get_images(full=True)
            if image_list:
                result["has_images"] = True
                
                # Only attempt OCR if pytesseract is available
                if pytesseract:
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Try to perform OCR on the image
                            try:
                                import io
                                image = Image.open(io.BytesIO(image_bytes))
                                ocr_text = pytesseract.image_to_string(image)
                                if ocr_text.strip():
                                    images.append({
                                        "index": img_index,
                                        "ocr_text": ocr_text.strip()
                                    })
                            except Exception as e:
                                logging.warning(f"OCR failed for image on page {page_num+1}: {e}")
                        except Exception as e:
                            logging.warning(f"Failed to process image on page {page_num+1}: {e}")

            
            if images:
                page_dict["images"] = images
            
            result["pages"].append(page_dict)
            result["text"] += page_text + "\n\n"
        
        # Preprocess the extracted text
        result["text"] = preprocess_text(result["text"])
        logging.info(f"Extracted {len(result['text'])} characters from {pdf_path}")
        
        return result
    except Exception as e:
        logging.error(f"Error extracting content from {pdf_path}: {e}")
        return {"text": "", "pages": [], "metadata": {}, "error": str(e)}

# ------------------------------
# Main enrichment function
# ------------------------------
def enrich_record(pdf_data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    """
    Given extracted PDF data and its source URL,
    add cleaned text, extracted entities, keywords, and a summary.
    """
    raw_text = pdf_data.get("text", "")
    if not raw_text.strip():
        logging.warning(f"No raw text for URL: {source_url}")
        return {
            "url": source_url,
            "cleaned_text": "",
            "entities": [],
            "keywords": [],
            "summary": "",
            "metadata": pdf_data.get("metadata", {})
        }
    
    try:
        # Clean and process text
        cleaned = clean_text(raw_text)
        
        # Extract and categorize entities
        entities = extract_entities(raw_text)
        categorized_entities = categorize_entities(entities)
        
        # Extract keywords
        keywords = extract_keywords(raw_text)
        
        # Generate summary
        summary = summarize_text(raw_text)
        
        # Create enriched record
        enriched = {
            "url": source_url,
            "cleaned_text": cleaned,
            "entities": entities,
            "categorized_entities": categorized_entities,
            "keywords": keywords,
            "summary": summary,
            "metadata": pdf_data.get("metadata", {}),
            "has_images": pdf_data.get("has_images", False),
            "has_tables": pdf_data.get("has_tables", False),
            "page_count": len(pdf_data.get("pages", [])),
        }
        
        # Add quality metrics
        enriched["quality_metrics"] = {
            "entity_count": len(entities),
            "keyword_count": len(keywords),
            "summary_length": len(summary.split()),
            "text_length": len(raw_text.split()),
        }
        
        return enriched
    except Exception as e:
        logging.error(f"Error enriching record for URL {source_url}: {e}")
        return {
            "url": source_url,
            "cleaned_text": "",
            "entities": [],
            "keywords": [],
            "summary": "",
            "metadata": pdf_data.get("metadata", {}),
            "error": str(e)
        }

# ------------------------------
# Parallel processing functions
# ------------------------------
def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """Process a single PDF file and return enriched data."""
    try:
        logging.info(f"Processing PDF file: {pdf_path}")
        pdf_data = extract_text_from_pdf(pdf_path)
        enriched = enrich_record(pdf_data, pdf_path)
        return enriched
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return {"url": pdf_path, "error": str(e)}

# ------------------------------
# Main script logic
# ------------------------------
def main(input_pdf_folder=None, output_json_path=None, input_file=None):
    # Use Docker container paths by default
    if not input_pdf_folder and not input_file:
        input_pdf_folder = "/data/extracted"
    # Set default output JSON file path for enriched data if not provided
    if not output_json_path:
        output_json_path = "/data/enriched/enriched_data.json"
    
    # Number of worker processes for parallel processing
    max_workers = max(1, os.cpu_count() - 1)  # Use all but one CPU core
    
    enriched_records = []
    
    # Check if we have an input JSON file from manual extractor
    if input_file and os.path.exists(input_file):
        logging.info(f"Processing input file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            
            # Process each record in the extracted data
            for record in tqdm(extracted_data, desc="Enriching records"):
                try:
                    # If the record has text content, enrich it directly
                    if "text" in record:
                        enriched = enrich_record(record, record.get("url", "unknown"))
                        enriched_records.append(enriched)
                        logging.info(f"Successfully processed record from {record.get('url', 'unknown')}")
                    else:
                        logging.warning(f"Record missing text content: {record.get('url', 'unknown')}")
                        enriched_records.append({
                            "url": record.get("url", "unknown"),
                            "error": "Missing text content"
                        })
                except Exception as e:
                    logging.error(f"Error processing record: {e}")
                    enriched_records.append({
                        "url": record.get("url", "unknown"),
                        "error": str(e)
                    })
        except Exception as e:
            logging.error(f"Error reading input file {input_file}: {e}")
            sys.exit(1)
    # Otherwise process PDF files from the input folder
    elif input_pdf_folder and os.path.exists(input_pdf_folder):
        logging.info(f"Processing PDF files from folder: {input_pdf_folder}")
        
        # List all PDF files in the folder (recursively if needed)
        pdf_files = []
        for root, dirs, files in os.walk(input_pdf_folder):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {input_pdf_folder}")
            sys.exit(1)
            
        logging.info(f"Found {len(pdf_files)} PDF files in {input_pdf_folder}.")
        logging.info(f"Using {max_workers} worker processes for parallel processing.")
        
        # Process PDFs in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_pdf), total=len(pdf_files), desc="Enriching PDFs"):
                pdf_path = future_to_pdf[future]
                try:
                    enriched = future.result()
                    enriched_records.append(enriched)
                    logging.info(f"Successfully processed: {pdf_path}")
                except Exception as e:
                    logging.error(f"Processing failed for {pdf_path}: {e}")
                    enriched_records.append({"url": pdf_path, "error": str(e)})
    else:
        logging.error(f"Neither input file nor input folder found")
        sys.exit(1)
    
    # Save enriched data to JSON
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_json_path)
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(enriched_records, f, ensure_ascii=False, indent=4)
        logging.info(f"Enriched data successfully saved to {output_json_path}")
        
        # Generate summary statistics
        entity_counts = sum(len(record.get("entities", [])) for record in enriched_records)
        keyword_counts = sum(len(record.get("keywords", [])) for record in enriched_records)
        error_count = sum(1 for record in enriched_records if "error" in record)
        
        logging.info(f"Processing summary:")
        logging.info(f"  - Total PDFs processed: {len(enriched_records)}")
        logging.info(f"  - Successful: {len(enriched_records) - error_count}")
        logging.info(f"  - Failed: {error_count}")
        logging.info(f"  - Total entities extracted: {entity_counts}")
        logging.info(f"  - Total keywords extracted: {keyword_counts}")
        
    except Exception as e:
        logging.error(f"Error saving enriched JSON: {e}")
    
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Enrich PDF documents with NLP processing')
    parser.add_argument('--input-file', type=str, help='Input JSON file path (from manual extractor)')
    parser.add_argument('--output-file', type=str, help='Output JSON file path for enriched data')
    parser.add_argument('--input-folder', type=str, help='Input folder containing PDF files')
    args = parser.parse_args()
    
    # Call main function with provided arguments
    main(
        input_pdf_folder=args.input_folder,
        output_json_path=args.output_file,
        input_file=args.input_file
    )
