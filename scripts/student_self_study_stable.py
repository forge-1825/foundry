import os
import sys
import re
import json
import logging
import argparse
import time
import gc
from datetime import datetime
import torch
import numpy as np
import PyPDF2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# -----------------------------------------------------
# Memory management functions
# -----------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Cleared GPU memory cache")

# -----------------------------------------------------
# Argument parsing
# -----------------------------------------------------
parser = argparse.ArgumentParser(description="Self-study module using distilled student model (Stable Version)")
parser.add_argument("--pdf_folder", type=str, help="Path to folder containing PDF files")
parser.add_argument("--model_path", type=str, default="./distilled_model_final", 
                    help="Path to the distilled student model")
parser.add_argument("--output_dir", type=str, default="./self_study_results", 
                    help="Directory to save results")
parser.add_argument("--num_questions", type=int, default=3, 
                    help="Number of questions to generate per sentence (default: 3, reduced from 10)")
parser.add_argument("--batch_size", type=int, default=1, 
                    help="Batch size for processing (default: 1, reduced from 4)")
parser.add_argument("--min_sentence_length", type=int, default=10, 
                    help="Minimum number of words in a sentence to process")
parser.add_argument("--max_sentence_length", type=int, default=50, 
                    help="Maximum number of words in a sentence to process (reduced from 100)")
parser.add_argument("--temperature", type=float, default=0.5, 
                    help="Temperature for question generation (reduced from 0.7)")
parser.add_argument("--max_length", type=int, default=32, 
                    help="Maximum length for generated text (reduced from 64)")
parser.add_argument("--max_sentences", type=int, default=10, 
                    help="Maximum number of sentences to process per PDF (default: 10)")
parser.add_argument("--use_8bit", action="store_true", 
                    help="Load model in 8-bit precision to save memory")
parser.add_argument("--checkpoint_interval", type=int, default=5, 
                    help="Save checkpoint after processing this many sentences")
parser.add_argument("--verbose", action="store_true", 
                    help="Print detailed output")
args = parser.parse_args()

# -----------------------------------------------------
# Logging configuration
# -----------------------------------------------------
os.makedirs(args.output_dir, exist_ok=True)
log_file = os.path.join(args.output_dir, f"self_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting student self-study (STABLE VERSION)")
logging.info(f"Arguments: {args}")

# -----------------------------------------------------
# Download NLTK resources if not already downloaded
# -----------------------------------------------------
try:
    # Download all required NLTK resources
    logging.info("Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK resources: {e}")
    logging.warning("Some NLTK features may not work properly.")

# -----------------------------------------------------
# Load the student model (our distilled model)
# -----------------------------------------------------
def load_model(model_path, use_8bit=False):
    logging.info(f"Loading student model from {model_path}...")
    try:
        # Check if this is a PEFT adapter model by looking for adapter_config.json
        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_adapter:
            # Get the base model from the adapter config
            import json
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                
                base_model_name = adapter_config.get("base_model_name_or_path", "microsoft/phi-2")
            else:
                base_model_name = "microsoft/phi-2"  # Default if config not found
                
            logging.info(f"This is a PEFT adapter model. Loading base model: {base_model_name}")
            
            # Configure quantization if requested
            if use_8bit:
                logging.info("Loading model in 8-bit precision to save memory")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            # Set pad_token to eos_token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
            
            # Load the PEFT adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Load as a regular model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Set pad_token to eos_token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
            
            # Configure quantization if requested
            if use_8bit:
                logging.info("Loading model in 8-bit precision to save memory")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not use_8bit:  # If using 8-bit, device_map="auto" handles this
            model.to(device)
        model.eval()
        logging.info(f"Student model loaded on device: {next(model.parameters()).device}")
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(f"Detailed error: {str(e)}")
        sys.exit(1)

try:
    student_tokenizer, student_model, device = load_model(args.model_path, args.use_8bit)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

# -----------------------------------------------------
# Load SentenceTransformer for similarity measurement
# -----------------------------------------------------
logging.info("Loading SentenceTransformer for similarity scoring...")
try:
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    sim_model.to(device)
except Exception as e:
    logging.error(f"Error loading SentenceTransformer: {e}")
    logging.warning("Will use a simple word overlap metric for similarity instead")
    sim_model = None

# Fallback similarity function using word overlap
def word_overlap_similarity(text1, text2):
    """Compute similarity based on word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    return len(intersection) / max(len(words1), len(words2))

# -----------------------------------------------------
# PDF extraction and text processing functions
# -----------------------------------------------------
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Limit to first 10 pages for stability
            for i, page in enumerate(reader.pages[:10]):
                if i >= 10:  # Only process first 10 pages
                    break
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
    return text

def clean_text(text):
    """Clean extracted text from common PDF artifacts."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers
    text = re.sub(r'\b\d+\b(?:\s*\|\s*\d+)?', '', text)
    # Remove headers/footers (common patterns)
    text = re.sub(r'(?i)page \d+ of \d+', '', text)
    # Remove URLs (simplified pattern)
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()

def split_into_sentences(text):
    """Split text into sentences using regex."""
    # Use regex for sentence splitting (more reliable than NLTK in this case)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def filter_sentences(sentences, min_words=5, max_words=100):
    """Filter sentences based on length and quality."""
    filtered = []
    for sentence in sentences:
        words = sentence.split()
        if min_words <= len(words) <= max_words:
            # Skip sentences with too many numbers or special characters
            if len(re.findall(r'[a-zA-Z]', sentence)) < len(sentence) * 0.5:
                continue
            filtered.append(sentence)
    return filtered

# -----------------------------------------------------
# Functions using the student model to generate questions and answers
# -----------------------------------------------------
def generate_question(sentence, temperature=0.5, max_length=32):
    """Generate a single question for a sentence."""
    # Create a more focused prompt
    prompt = f"""Generate an insightful question about this sentence: "{sentence}"

Question:"""
    
    # Use simple encoding without padding
    input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            # Use max_new_tokens instead of max_length to avoid input length errors
            outputs = student_model.generate(
                input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                no_repeat_ngram_size=2,
                pad_token_id=student_tokenizer.eos_token_id
            )
        
        # Process the generated question
        question_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated question (remove the prompt)
        question_text = question_text.split("Question:")[-1].strip()
        
        # Clean up the question
        if not question_text.endswith('?'):
            question_text += '?'
        
        return question_text
    except Exception as e:
        logging.error(f"Error generating question for sentence: {sentence[:50]}...: {e}")
        # Return a fallback question
        return f"What does this mean: {sentence[:30]}...?"

def generate_answer(question, max_length=32):
    """Generate an answer for a question."""
    # Create a focused prompt for answering
    prompt = f"""Answer this question concisely:

Question: {question}

Answer:"""
    
    # Process one prompt at a time without padding
    input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            output = student_model.generate(
                input_ids,
                max_new_tokens=max_length,
                do_sample=False,  # Deterministic for stability
                pad_token_id=student_tokenizer.eos_token_id
            )
        
        # Process the generated answer
        answer_text = student_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the generated answer (remove the prompt)
        answer_text = answer_text.split("Answer:")[-1].strip()
        
        return answer_text
    except Exception as e:
        logging.error(f"Error generating answer for question: {question}: {e}")
        # Return fallback answer
        return "Unable to generate an answer for this question."

def compute_similarity(reference, generated):
    """Compute similarity between reference and generated text."""
    try:
        if sim_model is not None:
            # Use SentenceTransformer if available
            ref_emb = sim_model.encode(reference, convert_to_tensor=True)
            gen_emb = sim_model.encode(generated, convert_to_tensor=True)
            
            # Compute cosine similarity
            cosine_sim = util.pytorch_cos_sim(ref_emb, gen_emb).item()
            
            return cosine_sim
        else:
            # Fallback to word overlap
            return word_overlap_similarity(reference, generated)
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        return word_overlap_similarity(reference, generated)  # Fallback

# -----------------------------------------------------
# Main self-study function with checkpointing
# -----------------------------------------------------
def run_self_study(pdf_folder, output_dir):
    """Run the self-study process on all PDFs in the folder with checkpointing."""
    if not os.path.isdir(pdf_folder):
        logging.error(f"Folder {pdf_folder} does not exist.")
        return False
    
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.error("No PDF files found in the specified folder.")
        return False
    
    logging.info(f"Found {len(pdf_files)} PDF file(s) in {pdf_folder}.")
    
    all_results = []
    
    for pdf_file in pdf_files:
        try:
            pdf_results = process_pdf(pdf_file)
            all_results.extend(pdf_results)
            
            # Save checkpoint after each PDF
            checkpoint_file = os.path.join(output_dir, f"checkpoint_{os.path.basename(pdf_file)}.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(pdf_results, f, indent=2)
            logging.info(f"Checkpoint saved to {checkpoint_file}")
            
            # Clear memory
            clear_gpu_memory()
            
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_file}: {e}")
            logging.info("Continuing with next PDF...")
            continue
    
    # Save all results to a single JSON file
    results_file = os.path.join(output_dir, f"self_study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"All results saved to {results_file}")
    
    # Generate summary statistics
    generate_summary(all_results, output_dir)
    
    return True

def process_pdf(pdf_file):
    """Process a single PDF file with sentence-level checkpointing."""
    logging.info(f"\nProcessing PDF file: {pdf_file}")
    
    # Extract and clean text
    text = extract_text_from_pdf(pdf_file)
    if not text:
        logging.warning(f"No text extracted from {pdf_file}.")
        return []
    
    text = clean_text(text)
    
    # Split into sentences and filter
    sentences = split_into_sentences(text)
    logging.info(f"Extracted {len(sentences)} sentence(s).")
    
    filtered_sentences = filter_sentences(
        sentences, 
        min_words=args.min_sentence_length, 
        max_words=args.max_sentence_length
    )
    logging.info(f"After filtering: {len(filtered_sentences)} sentence(s).")
    
    if not filtered_sentences:
        logging.warning("No suitable sentences found after filtering.")
        return []
    
    # Limit the number of sentences to process
    if len(filtered_sentences) > args.max_sentences:
        logging.info(f"Limiting to {args.max_sentences} sentences for stability")
        filtered_sentences = filtered_sentences[:args.max_sentences]
    
    pdf_results = []
    checkpoint_count = 0
    
    # Process one sentence at a time for stability
    for i, sentence in enumerate(filtered_sentences):
        sentence_results = []
        logging.info(f"Processing sentence {i+1}/{len(filtered_sentences)}")
        
        # Generate questions for this sentence
        for q in range(args.num_questions):
            try:
                # Generate a question
                question = generate_question(
                    sentence,
                    temperature=args.temperature,
                    max_length=args.max_length
                )
                
                # Generate an answer
                answer = generate_answer(
                    question,
                    max_length=args.max_length
                )
                
                # Compute similarity
                similarity = compute_similarity(sentence, answer)
                
                # Add to results
                result = {
                    "sentence": sentence,
                    "question": question,
                    "answer": answer,
                    "similarity": similarity,
                    "pdf_file": os.path.basename(pdf_file)
                }
                
                sentence_results.append(result)
                
                # Print if verbose
                if args.verbose:
                    print(f"\nSentence: {sentence}")
                    print(f"Question: {question}")
                    print(f"Answer: {answer}")
                    print(f"Similarity: {similarity:.4f}")
                
                # Clear memory after each question
                clear_gpu_memory()
                
            except Exception as e:
                logging.error(f"Error processing question {q+1} for sentence {i+1}: {e}")
                continue
        
        # Add results for this sentence
        pdf_results.extend(sentence_results)
        
        # Save checkpoint at intervals
        checkpoint_count += 1
        if checkpoint_count >= args.checkpoint_interval:
            checkpoint_file = os.path.join(
                args.output_dir, 
                f"checkpoint_{os.path.basename(pdf_file)}_{i+1}.json"
            )
            with open(checkpoint_file, 'w') as f:
                json.dump(pdf_results, f, indent=2)
            logging.info(f"Intermediate checkpoint saved to {checkpoint_file}")
            checkpoint_count = 0
    
    return pdf_results

def generate_summary(results, output_dir):
    """Generate summary statistics from results."""
    if not results:
        logging.warning("No results to summarize.")
        return
    
    # Calculate overall statistics
    similarities = [r["similarity"] for r in results]
    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    # Group by PDF file
    pdf_stats = {}
    for result in results:
        pdf_file = result.get("pdf_file", "unknown")
        if pdf_file not in pdf_stats:
            pdf_stats[pdf_file] = []
        pdf_stats[pdf_file].append(result["similarity"])
    
    # Calculate per-PDF statistics
    pdf_summaries = {}
    for pdf_file, sims in pdf_stats.items():
        pdf_summaries[pdf_file] = {
            "count": len(sims),
            "avg_similarity": np.mean(sims),
            "median_similarity": np.median(sims),
            "min_similarity": np.min(sims),
            "max_similarity": np.max(sims)
        }
    
    # Create summary dictionary
    summary = {
        "overall": {
            "total_questions": len(results),
            "avg_similarity": avg_similarity,
            "median_similarity": median_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        },
        "per_pdf": pdf_summaries
    }
    
    # Save summary to file
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Summary statistics saved to {summary_file}")
    
    # Print summary to console
    print("\n===== SUMMARY STATISTICS =====")
    print(f"Total questions processed: {len(results)}")
    print(f"Average similarity score: {avg_similarity:.4f}")
    print(f"Median similarity score: {median_similarity:.4f}")
    print(f"Range: {min_similarity:.4f} - {max_similarity:.4f}")
    print("==============================\n")

# -----------------------------------------------------
# Main execution
# -----------------------------------------------------
if __name__ == "__main__":
    # If no PDF folder is provided via command line, ask for it
    if not args.pdf_folder:
        args.pdf_folder = input("Enter the full path to the folder containing the PDF files: ").strip()
    
    start_time = time.time()
    success = run_self_study(args.pdf_folder, args.output_dir)
    end_time = time.time()
    
    if success:
        logging.info(f"Self-study process completed in {end_time - start_time:.2f} seconds.")
    else:
        logging.error("Self-study process failed.")
