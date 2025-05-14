# pyright: reportMissingImports=false
import os
import sys
import json
import time
import logging
import concurrent.futures
from tqdm import tqdm
import re
import numpy as np
import gc
from typing import List, Dict, Any, Optional, Tuple

# Check for environment variables that force CPU mode
force_cpu_mode = os.environ.get('USE_CUDA', '1') == '0' or os.environ.get('CUDA_VISIBLE_DEVICES', '') == '-1'
if force_cpu_mode:
    logging.info("CPU mode forced by environment variables")
    print("\nGPU ACCELERATION DISABLED: CPU mode forced by environment variables\n")
    has_cuda = False
else:
    # Try to import torch for CUDA detection
    try:
        import torch
        # Check if CUDA is available and properly configured
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            # Set PyTorch to use less GPU memory
            try:
                torch.cuda.set_per_process_memory_fraction(0.7)  # Reduced to 70% to avoid memory issues
                logging.info("Set GPU memory fraction to 70%")
            except Exception as e:
                logging.warning(f"Failed to set GPU memory fraction: {e}")
                
            # Verify CUDA is actually working by creating a small tensor
            try:
                # Test with a simple operation
                test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                test_result = test_tensor.sum().item()
                
                # Additional test with a slightly more complex operation
                try:
                    # Try a matrix multiplication as an additional test
                    test_matrix1 = torch.rand(10, 10, device='cuda')
                    test_matrix2 = torch.rand(10, 10, device='cuda')
                    test_result2 = torch.matmul(test_matrix1, test_matrix2)
                    del test_matrix1, test_matrix2, test_result2  # Clean up test tensors
                    
                    logging.info(f"CUDA is available and working with {torch.cuda.device_count()} device(s). Using GPU acceleration.")
                    # Get GPU info
                    for i in range(torch.cuda.device_count()):
                        logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
                        logging.info(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                    print(f"\nGPU ACCELERATION ENABLED: {torch.cuda.get_device_name(0)}")
                    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
                except RuntimeError as e:
                    logging.error(f"CUDA test failed with matrix multiplication: {e}")
                    logging.warning("Falling back to CPU mode")
                    has_cuda = False
                    print("\nGPU ACCELERATION DISABLED: CUDA test failed\n")
            except Exception as e:
                logging.error(f"CUDA is available but failed to initialize: {e}")
                logging.warning("Falling back to CPU mode")
                has_cuda = False
                print("\nGPU ACCELERATION DISABLED: Using CPU only\n")
        else:
            logging.info("CUDA is not available. Using CPU only.")
            print("\nGPU ACCELERATION DISABLED: Using CPU only\n")
    except ImportError:
        logging.error("PyTorch not installed. GPU acceleration will be disabled.")
        has_cuda = False
        print("\nGPU ACCELERATION DISABLED: Using CPU only\n")

# Function to safely handle CUDA operations with fallback
def safe_cuda_operation(cuda_func, cpu_func=None, *args, **kwargs):
    """
    Safely execute a CUDA operation with fallback to CPU if it fails.
    
    Args:
        cuda_func: Function to execute if CUDA is available
        cpu_func: Function to execute if CUDA fails or is not available (optional)
        *args, **kwargs: Arguments to pass to the functions
        
    Returns:
        Result of the function call
    """
    global has_cuda
    
    if has_cuda:
        try:
            return cuda_func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logging.error(f"CUDA operation failed: {e}")
                logging.warning("Disabling CUDA for future operations")
                has_cuda = False
                print(f"\nGPU ACCELERATION DISABLED due to error: {e}\n")
                
                # Try to clean up CUDA memory
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as cleanup_error:
                    logging.error(f"Failed to clean up CUDA memory: {cleanup_error}")
                
                # If CPU fallback is provided, use it
                if cpu_func is not None:
                    logging.info("Falling back to CPU implementation")
                    return cpu_func(*args, **kwargs)
            else:
                # Re-raise if it's not a CUDA error
                raise
    
    # If CUDA is not available or no CPU fallback is provided
    if cpu_func is not None:
        return cpu_func(*args, **kwargs)
    
    # If we get here, both CUDA failed and no CPU fallback was provided
    raise RuntimeError("Operation failed on CUDA and no CPU fallback was provided")

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enrichment.log", mode='w')
    ]
)

# Print a prominent message about Tesseract
print("\n" + "="*80)
print("NOTE: For OCR functionality (extracting text from images in PDFs),")
print("      you need to install Tesseract OCR separately from:")
print("      https://github.com/UB-Mannheim/tesseract/wiki (Windows)")
print("      or use your package manager on Linux/macOS.")
print("      Without Tesseract, the script will still work but will skip OCR.")
print("="*80 + "\n")

# Try to import required packages with fallbacks
try:
    import fitz  # PyMuPDF for PDF extraction
except ImportError:
    logging.error("PyMuPDF (fitz) not installed. Please run install_dependencies.py first.")
    fitz = None

try:
    import stanza
except ImportError:
    logging.error("Stanza not installed. Please run install_dependencies.py first.")
    stanza = None

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

# Define a simple sentence tokenizer that doesn't rely on NLTK
def simple_sentence_tokenizer(text):
    """A simple sentence tokenizer that doesn't rely on NLTK."""
    # Replace common sentence-ending punctuation with periods
    text = text.replace('!', '.').replace('?', '.')
    # Split by periods and create sentences
    sentences = []
    for s in text.split('.'):
        if s.strip():
            sentences.append(s.strip() + '.')
    return sentences

# Define a simple stopwords list
SIMPLE_STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'when', 'where', 'how', 'to', 'of', 'for', 'with', 'in', 'on', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may',
    'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
    'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours',
    'theirs', 'am', 'who', 'whom', 'whose', 'which', 'there', 'here'
])

# Initialize NLTK variables
nltk = None
nltk_stopwords = None
sent_tokenize = simple_sentence_tokenizer
stopwords = SIMPLE_STOPWORDS

# Try to import NLTK, but don't use it directly yet
try:
    import nltk as nltk_module
    nltk = nltk_module
    print("NLTK module imported successfully")
except ImportError:
    logging.error("NLTK not installed. Using simplified text processing.")
    nltk = None

# Only if NLTK import succeeded, try to download and use resources
if nltk:
    try:
        # Download resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK resources downloaded")
        
        # Skip trying to use NLTK's tokenizer directly since it's causing issues
        # Just use our simple tokenizer which is more reliable
        print("Using simple sentence tokenizer to avoid NLTK punkt_tab issues")
        
        # Try to get stopwords
        try:
            from nltk.corpus import stopwords as nltk_stopwords_module
            stopwords_list = nltk_stopwords_module.words('english')
            if stopwords_list:
                nltk_stopwords = set(stopwords_list)
                stopwords = nltk_stopwords
                print("Using NLTK stopwords")
            else:
                print("NLTK stopwords empty, using simple stopwords")
        except Exception as e:
            logging.error(f"Error setting up NLTK stopwords: {e}")
            print(f"NLTK stopwords error: {e}")
            # Keep using the simple stopwords
            
    except Exception as e:
        logging.error(f"Error setting up NLTK: {e}")
        print(f"NLTK setup error: {e}")

# ------------------------------
# Load Stanza model if available - with memory optimization
# ------------------------------
nlp = None
if stanza:
    try:
        # Initialize Stanza pipeline with only necessary processors to reduce memory usage
        # Reduced from full set to only what's needed
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,ner',  # Reduced processor set
            use_gpu=True,  # Force GPU usage as requested by user
            verbose=False,
            # Add memory optimization settings
            tokenize_batch_size=1024,
            pos_batch_size=1024,
            ner_batch_size=1024
        )
        logging.info(f"Loaded Stanza English model with GPU acceleration")
    except Exception as e:
        logging.error(f"Error loading Stanza model with GPU: {e}")
        # Try again with CPU if GPU fails
        try:
            nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,ner',
                use_gpu=False,
                verbose=False,
                tokenize_batch_size=1024,
                pos_batch_size=1024,
                ner_batch_size=1024
            )
            logging.info("Loaded Stanza English model with CPU (GPU failed)")
        except Exception as e2:
            logging.error(f"Error loading Stanza model with CPU: {e2}")
            nlp = None

if not nlp:
    logging.warning("NLP processing will be limited without a Stanza model.")

# ------------------------------
# Load summarization models (GPU and CPU versions)
# ------------------------------
summarizer_gpu = None
summarizer_cpu = None
summarizer = None

if pipeline:
    try:
        # Load CPU model first as fallback
        try:
            summarizer_cpu = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
            logging.info("Loaded CPU summarization pipeline as fallback")
        except Exception as e:
            logging.error(f"Error loading CPU summarization model: {e}")
            summarizer_cpu = None
        
        # Load GPU model if CUDA available
        if has_cuda:
            try:
                # Force GPU usage by explicitly setting device map
                model_name = "facebook/bart-large-cnn"
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                summarizer_gpu = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)
                logging.info(f"Loaded summarization pipeline with {model_name} on GPU")
                print(f"Summarization model loaded on GPU: {torch.cuda.get_device_name(0)}")
                # Set active summarizer to GPU
                summarizer = summarizer_gpu
            except Exception as e:
                logging.error(f"Error initializing GPU summarization pipeline: {e}")
                # Use CPU model as fallback
                if summarizer_cpu:
                    summarizer = summarizer_cpu
                    logging.info("Using CPU summarization model as fallback")
        else:
            # Use CPU model if no GPU
            summarizer = summarizer_cpu
            logging.info("Using CPU summarization model (no GPU available)")
    except Exception as e:
        logging.error(f"Error initializing summarization pipeline: {e}")
        logging.warning("Summarization will be disabled.")
        print(f"Failed to load summarization model: {e}")
else:
    logging.warning("Transformers not available. Summarization will be disabled.")

# ------------------------------
# Load sentence transformer for semantic similarity if available
# ------------------------------
sentence_model = None
if SentenceTransformer:
    try:
        # Use GPU if available
        if has_cuda:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            logging.info(f"Loaded sentence transformer model: all-MiniLM-L6-v2 on GPU")
        else:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logging.info(f"Loaded sentence transformer model: all-MiniLM-L6-v2 on CPU")
    except Exception as e:
        logging.error(f"Error loading sentence transformer model with GPU: {e}")
        # Try with CPU if GPU fails
        try:
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logging.info(f"Loaded sentence transformer model: all-MiniLM-L6-v2 on CPU (GPU failed)")
        except Exception as e2:
            logging.error(f"Error loading sentence transformer model with CPU: {e2}")
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
    Lemmatize text and remove stopwords/punctuation using Stanza if available.
    Returns a cleaned version of the text.
    """
    if not text:
        return ""
        
    if nlp:
        try:
            # Process with Stanza - Ensure text is a non-empty string
            if not isinstance(text, str) or not text.strip():
                return ""
                
            doc = nlp(text)
            
            # Get lemmatized tokens, excluding stopwords and punctuation
            tokens = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    # Skip punctuation and short words
                    if word.upos == 'PUNCT' or len(word.text.strip()) <= 1:
                        continue
                    
                    # Skip common stopwords (Stanza doesn't have built-in stopwords)
                    if word.text.lower() in {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 
                                            'as', 'what', 'when', 'where', 'how', 'to', 'of', 'for',
                                            'with', 'in', 'on', 'by', 'is', 'are', 'was', 'were'}:
                        continue
                    
                    # Skip None values
                    if word.lemma is None:
                        tokens.append(word.text)
                    else:
                        # Add lemmatized form
                        tokens.append(word.lemma)
            
            return " ".join(tokens)
        except Exception as e:
            logging.error(f"Error cleaning text with Stanza: {e}")
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
        if isinstance(stopwords, set):
            words = [word for word in words if word.lower() not in stopwords]
        else:
            try:
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word.lower() not in stop_words]
            except Exception:
                pass
            
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
        # Use the global sent_tokenize function that we've already set up
        sentences = sent_tokenize(text)
            
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
    Extract named entities from text using Stanza with confidence filtering.
    Returns a list of dictionaries with entity text, label, and confidence score.
    """
    if not text or not nlp:
        return []
        
    try:
        doc = nlp(text)
        
        # Extract entities with confidence scores
        entities = []
        char_offset = 0
        
        for sentence in doc.sentences:
            # Track character offset for this sentence
            sentence_text = sentence.text
            
            for ent in sentence.ents:
                # Estimate confidence based on entity length and context
                # This is a simplified approach since Stanza doesn't provide confidence scores directly
                confidence = min(1.0, 0.5 + (len(ent.text.split()) / 10))
                
                if confidence >= confidence_threshold:
                    # Find the start and end character positions
                    start_char = text.find(ent.text, char_offset)
                    if start_char == -1:  # If not found from current offset, search from beginning
                        start_char = text.find(ent.text)
                    
                    if start_char != -1:  # Only add if we found the entity text
                        end_char = start_char + len(ent.text)
                        
                        entities.append({
                            "text": ent.text,
                            "label": ent.type,  # Stanza uses 'type' instead of 'label_'
                            "start": start_char,
                            "end": end_char,
                            "confidence": round(confidence, 2)
                        })
            
            # Update character offset for next sentence
            char_offset += len(sentence_text)
        
        return entities
    except Exception as e:
        logging.error(f"Error extracting entities with Stanza: {e}")
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
def extractive_summarize(text: str, sentence_count: int = 3) -> str:
    """
    Simple extractive summarization as a fallback method.
    Uses word frequency to identify important sentences.
    """
    if not text:
        return ""
    
    # Get sentences using our global sent_tokenize function
    sentences = sent_tokenize(text)
    
    if len(sentences) <= sentence_count:
        return text
    
    # Simple word frequency-based scoring
    word_freq = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            word = word.strip('.,;:!?()[]{}"\'')
            if len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score sentences based on word frequency
    sentence_scores = []
    for sentence in sentences:
        score = 0
        for word in sentence.lower().split():
            word = word.strip('.,;:!?()[]{}"\'')
            if len(word) > 1:
                score += word_freq.get(word, 0)
        sentence_scores.append((sentence, score))
    
    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:sentence_count]
    
    # Sort by original order
    ordered_sentences = [s for s, _ in sorted(top_sentences, key=lambda x: sentences.index(x[0]))]
    
    return " ".join(ordered_sentences)

def safe_summarize(text: str, max_retries: int = 3) -> List[Dict[str, str]]:
    """
    Try to summarize text with retries and fallbacks.
    Returns the summarizer output or None if all attempts fail.
    """
    global summarizer, summarizer_gpu, summarizer_cpu
    
    if not summarizer:
        return None
    
    # Ensure text is not too long for the model
    # If text is very long, truncate it to avoid potential issues
    if len(text) > 10000:  # Arbitrary limit to prevent extremely long inputs
        text = text[:10000]
        logging.warning(f"Text truncated to 10000 characters for summarization")
    
    # Make sure text is not empty
    if not text.strip():
        logging.warning("Empty text provided for summarization")
        return None
    
    # For very short texts, don't even try the transformer
    if len(text.split()) < 50:
        # Create a fake summarizer output structure
        return [{'summary_text': extractive_summarize(text, 2)}]
    
    # Define GPU and CPU summarization functions for our safe_cuda_operation
    def gpu_summarize(attempt):
        if attempt == 0:
            # First attempt with normal parameters
            return summarizer(text, max_length=130, min_length=30, do_sample=False)
        elif attempt == 1:
            # Second attempt with more conservative parameters
            return summarizer(text, max_length=100, min_length=20, do_sample=False)
        else:
            # Third attempt with even more conservative parameters
            return summarizer(text, max_length=80, min_length=10, do_sample=False)
    
    def cpu_summarize(attempt):
        # If we have a CPU summarizer, use it
        if summarizer_cpu:
            if attempt == 0:
                return summarizer_cpu(text, max_length=130, min_length=30, do_sample=False)
            elif attempt == 1:
                return summarizer_cpu(text, max_length=100, min_length=20, do_sample=False)
            else:
                return summarizer_cpu(text, max_length=80, min_length=10, do_sample=False)
        else:
            # No CPU summarizer available, use extractive summarization
            return [{'summary_text': extractive_summarize(text)}]
    
    original_summarizer = summarizer
    
    for attempt in range(max_retries):
        try:
            # Use our safe CUDA operation wrapper
            if has_cuda and summarizer == summarizer_gpu:
                result = safe_cuda_operation(
                    lambda: gpu_summarize(attempt),
                    lambda: cpu_summarize(attempt)
                )
            else:
                # Already using CPU or no GPU available
                result = cpu_summarize(attempt)
            
            # Verify the result structure before returning
            if result and isinstance(result, list) and len(result) > 0:
                return result
            else:
                logging.warning(f"Summarization attempt {attempt+1} returned invalid structure: {result}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # If all attempts failed, create a dummy summarizer output with extractive summary
                    logging.info("Using extractive summarization as fallback")
                    return [{'summary_text': extractive_summarize(text)}]
                
        except IndexError as e:
            # Specifically handle index errors which seem to be common
            logging.warning(f"Summarization attempt {attempt+1} failed with IndexError: {e}. Retrying...")
            if attempt < max_retries - 1:
                continue
            else:
                logging.error(f"All summarization attempts failed with IndexError")
                # If all attempts failed, create a dummy summarizer output with extractive summary
                logging.info("Using extractive summarization as fallback after IndexError")
                return [{'summary_text': extractive_summarize(text)}]
                
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Summarization attempt {attempt+1} failed: {e}. Retrying...")
            else:
                logging.error(f"All summarization attempts failed: {e}")
                # If all attempts failed, create a dummy summarizer output with extractive summary
                logging.info("Using extractive summarization as fallback after exception")
                return [{'summary_text': extractive_summarize(text)}]
    
    # Restore the original summarizer
    summarizer = original_summarizer
    
    # If all retries failed, return extractive summary
    logging.info("Using extractive summarization as fallback after all retries")
    return [{'summary_text': extractive_summarize(text)}]

def summarize_text(text: str, max_chunk_tokens: int = 1024) -> str:
    """
    Generate a summary using the Hugging Face summarization pipeline.
    Uses smart chunking to preserve sentence boundaries.
    """
    # At the beginning of the function
    if not text:
        return ""
    
    if not summarizer:
        return extractive_summarize(text) if text else ""
    
    try:
        if not text.strip():
            return ""
        
        # Use smart chunking to split text
        chunks = smart_text_chunking(text, max_chunk_tokens)
        
        if len(chunks) == 1:
            # For short texts, summarize directly with proper error handling
            try:
                summary_output = safe_summarize(chunks[0])
                # Explicitly check the structure of the output
                if (summary_output and isinstance(summary_output, list) and 
                        len(summary_output) > 0 and 'summary_text' in summary_output[0]):
                    return summary_output[0]['summary_text']
                else:
                    logging.warning("Summarizer returned unexpected output format")
                    # Try extractive summarization as fallback
                    return extractive_summarize(chunks[0])
            except IndexError:
                logging.error("Index error in summarizer output")
                return extractive_summarize(chunks[0])  # Use extractive fallback
            except Exception as e:
                logging.error(f"Error summarizing single chunk: {e}")
                return extractive_summarize(chunks[0])  # Use extractive fallback
        else:
            # For longer texts, summarize each chunk with proper error handling
            chunk_summaries = []
            for idx, chunk in enumerate(chunks, 1):
                if not chunk.strip():
                    continue
                    
                try:
                    summary_output = safe_summarize(chunk)
                    # Explicitly check the structure of the output
                    if (summary_output and isinstance(summary_output, list) and 
                            len(summary_output) > 0 and 'summary_text' in summary_output[0]):
                        chunk_summaries.append(summary_output[0]['summary_text'])
                        logging.info(f"Chunk {idx}/{len(chunks)} summarized successfully.")
                    else:
                        logging.warning(f"Summarizer returned unexpected output format for chunk {idx}")
                        # Add an extractive summary
                        chunk_summaries.append(extractive_summarize(chunk))
                except IndexError:
                    logging.error(f"Index error in summarizer output for chunk {idx}")
                    # Add an extractive summary
                    chunk_summaries.append(extractive_summarize(chunk))
                except Exception as e:
                    logging.error(f"Error summarizing chunk {idx}: {e}")
                    # Add an extractive summary
                    chunk_summaries.append(extractive_summarize(chunk))
            
            # Combine chunk summaries
            if chunk_summaries:
                combined = " ".join(chunk_summaries)
                # If combined summary is still too long, summarize it again
                if len(combined.split()) > max_chunk_tokens:
                    try:
                        final_summary = safe_summarize(combined)
                        if (final_summary and isinstance(final_summary, list) and 
                                len(final_summary) > 0 and 'summary_text' in final_summary[0]):
                            return final_summary[0]['summary_text']
                        else:
                            logging.warning("Summarizer returned unexpected output format for combined chunks")
                            return extractive_summarize(combined)
                    except IndexError:
                        logging.error("Index error in summarizer output for combined chunks")
                        return extractive_summarize(combined)
                    except Exception as e:
                        logging.error(f"Error summarizing combined chunks: {e}")
                        return extractive_summarize(combined)
                return combined
            else:
                # If all summaries failed, return an extractive summary
                return extractive_summarize(text)
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
        stop_words = set()
        if isinstance(stopwords, set):
            stop_words = stopwords
        else:
            try:
                stop_words = set(stopwords.words('english'))
            except Exception as e:
                logging.warning(f"Failed to get stopwords: {e}, using basic stopwords list")
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
                sentences = sent_tokenize(text)
                
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
    
    # Check for tesseract availability
    is_tesseract_available = False
    if pytesseract:
        try:
            pytesseract.get_tesseract_version()
            is_tesseract_available = True
            logging.info("Tesseract OCR is available and will be used for image text extraction")
        except Exception as e:
            logging.warning(f"Tesseract is not properly installed or not in PATH. OCR will be disabled: {e}")
    
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
                if is_tesseract_available:
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
def enrich_record(pdf_data: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """
    Given extracted PDF data and its source path,
    add cleaned text, extracted entities, keywords, and a summary.
    """
    raw_text = pdf_data.get("text", "")
    if not raw_text.strip():
        logging.warning(f"No raw text for PDF: {pdf_path}")
        return {
            "path": pdf_path,
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
            "path": pdf_path,
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
        logging.error(f"Error enriching record for PDF {pdf_path}: {e}")
        return {
            "path": pdf_path,
            "cleaned_text": "",
            "entities": [],
            "keywords": [],
            "summary": "",
            "metadata": pdf_data.get("metadata", {}),
            "error": str(e)
        }

# ------------------------------
# Batch processing functions
# ------------------------------
def process_in_batches(pdf_files: List[str], output_json_path: str, batch_size: int = 5) -> List[Dict[str, Any]]:
    """Process files in small batches with explicit memory cleanup between batches."""
    all_enriched_records = []
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:min(i+batch_size, len(pdf_files))]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(pdf_files)-1)//batch_size + 1} ({len(batch)} files)")
        
        # Process this batch
        batch_records = process_pdf_files_sequentially(batch)
        all_enriched_records.extend(batch_records)
        
        # Save intermediate results
        try:
            temp_path = f"{output_json_path}.batch_{i//batch_size + 1}.temp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(all_enriched_records, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved intermediate results after batch {i//batch_size + 1}")
        except Exception as e:
            logging.error(f"Failed to save intermediate batch results: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Safely clear GPU memory with error handling
        if has_cuda:
            try:
                # Try to clear CUDA cache safely
                torch.cuda.empty_cache()
                logging.info("Cleared GPU cache after batch processing")
            except RuntimeError as e:
                logging.error(f"CUDA error during memory cleanup: {e}")
                logging.warning("Continuing without GPU memory cleanup")
    
    return all_enriched_records

def process_pdf_files_sequentially(pdf_files: List[str]) -> List[Dict[str, Any]]:
    """Process PDF files sequentially to avoid GPU memory sharing issues."""
    enriched_records = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            logging.info(f"Processing PDF file: {pdf_path}")
            pdf_data = extract_text_from_pdf(pdf_path)
            enriched = enrich_record(pdf_data, pdf_path)
            enriched_records.append(enriched)
            logging.info(f"Successfully processed: {pdf_path}")
            
            # Force GPU memory cleanup after each file with error handling
            if has_cuda:
                try:
                    torch.cuda.empty_cache()
                    logging.info("Cleared GPU cache after file processing")
                except RuntimeError as e:
                    logging.error(f"CUDA error during memory cleanup: {e}")
                    logging.warning("Continuing without GPU memory cleanup")
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            enriched_records.append({"path": pdf_path, "error": str(e)})
    
    return enriched_records

# ------------------------------
# Main script logic
# ------------------------------
def main():
    print("\n=== PDF Data Enrichment Tool with GPU Acceleration ===\n")
    print("NOTE: This tool has been optimized to prevent memory issues and crashes.")
    print("      Processing will be done in small batches with memory cleanup between batches.\n")
    
    # Prompt for input folder
    default_input = os.path.join(os.getcwd(), "PDFs")
    input_pdf_folder = input(f"Enter folder containing PDF files (default: {default_input}): ").strip()
    if not input_pdf_folder:
        input_pdf_folder = default_input
    
    # Prompt for output file
    default_output = os.path.join(os.getcwd(), "enriched_data.json")
    output_json_path = input(f"Enter output JSON file path (default: {default_output}): ").strip()
    if not output_json_path:
        output_json_path = default_output
    
    if not os.path.exists(input_pdf_folder):
        logging.error(f"Input PDF folder not found: {input_pdf_folder}")
        sys.exit(1)
    
    # List all PDF files in the folder (recursively if needed)
    pdf_files = []
    for root, dirs, files in os.walk(input_pdf_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        logging.error(f"No PDF files found in {input_pdf_folder}")
        sys.exit(1)
    
    logging.info(f"Found {len(pdf_files)} PDF files in {input_pdf_folder}.")
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    # Process in batches instead of parallel
    batch_size = 5  # Adjust based on available memory
    enriched_records = process_in_batches(pdf_files, output_json_path, batch_size)
    
    # Save enriched data to JSON
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(enriched_records, f, ensure_ascii=False, indent=4)
        logging.info(f"Enriched data successfully saved to {output_json_path}")
        print(f"\nEnriched data successfully saved to {output_json_path}")
        
        # Generate summary statistics
        entity_counts = sum(len(record.get("entities", [])) for record in enriched_records)
        keyword_counts = sum(len(record.get("keywords", [])) for record in enriched_records)
        error_count = sum(1 for record in enriched_records if "error" in record)
        
        print(f"\nProcessing summary:")
        print(f"  - Total PDFs processed: {len(enriched_records)}")
        print(f"  - Successful: {len(enriched_records) - error_count}")
        print(f"  - Failed: {error_count}")
        print(f"  - Total entities extracted: {entity_counts}")
        print(f"  - Total keywords extracted: {keyword_counts}")
        
        logging.info(f"Processing summary:")
        logging.info(f"  - Total PDFs processed: {len(enriched_records)}")
        logging.info(f"  - Successful: {len(enriched_records) - error_count}")
        logging.info(f"  - Failed: {error_count}")
        logging.info(f"  - Total entities extracted: {entity_counts}")
        logging.info(f"  - Total keywords extracted: {keyword_counts}")
        
    except Exception as e:
        logging.error(f"Error saving enriched JSON: {e}")
        print(f"Error saving enriched JSON: {e}")
    
if __name__ == "__main__":
    main()
