import os
import sys
import re
import json
import logging
import argparse
import time
from datetime import datetime
import torch
import numpy as np
import PyPDF2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# -----------------------------------------------------
# Argument parsing
# -----------------------------------------------------
parser = argparse.ArgumentParser(description="Self-study module using distilled student model")
parser.add_argument("--pdf_folder", type=str, help="Path to folder containing PDF files")
parser.add_argument("--model_path", type=str, default="./distilled_model_final", 
                    help="Path to the distilled student model")
parser.add_argument("--output_dir", type=str, default="./self_study_results", 
                    help="Directory to save results")
parser.add_argument("--num_questions", type=int, default=10, 
                    help="Number of questions to generate per sentence")
parser.add_argument("--batch_size", type=int, default=4, 
                    help="Batch size for processing")
parser.add_argument("--min_sentence_length", type=int, default=5, 
                    help="Minimum number of words in a sentence to process")
parser.add_argument("--max_sentence_length", type=int, default=100, 
                    help="Maximum number of words in a sentence to process")
parser.add_argument("--temperature", type=float, default=0.7, 
                    help="Temperature for question generation")
parser.add_argument("--max_length", type=int, default=64, 
                    help="Maximum length for generated text")
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
def load_model(model_path):
    logging.info(f"Loading student model from {model_path}...")
    try:
        # Check if this is a PEFT adapter model by looking for adapter_config.json
        is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_adapter:
            # Get the base model from the adapter config
            import json
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path", "microsoft/phi-2")
            logging.info(f"This is a PEFT adapter model. Loading base model: {base_model_name}")
            
            # Load the base model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            # Set pad_token to eos_token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
            
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
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
                
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logging.info(f"Student model loaded on device: {next(model.parameters()).device}")
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(f"Detailed error: {str(e)}")
        sys.exit(1)

try:
    student_tokenizer, student_model, device = load_model(args.model_path)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

# -----------------------------------------------------
# Load SentenceTransformer for similarity measurement
# -----------------------------------------------------
logging.info("Loading SentenceTransformer for similarity scoring...")
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
sim_model.to(device)

# -----------------------------------------------------
# PDF extraction and text processing functions
# -----------------------------------------------------
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
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
def generate_questions_batch(sentences, num_questions=10, batch_size=4, max_length=64, temperature=0.7):
    """Generate questions for a batch of sentences."""
    all_questions = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_questions = []
        
        for sentence in batch_sentences:
            # Create a more detailed prompt with examples
            prompt = f"""Generate {num_questions} diverse and insightful questions that test understanding of the following sentence. 
Make sure the questions cover different aspects and require deep comprehension.

Sentence: "{sentence}"

Questions:
1."""
            
            # Use simple encoding without padding for single inputs
            input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            try:
                with torch.no_grad():
                    # Use max_new_tokens instead of max_length to avoid input length errors
                    outputs = student_model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        num_return_sequences=1,  # Generate one sequence at a time
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        no_repeat_ngram_size=2,
                        pad_token_id=student_tokenizer.eos_token_id
                    )
                
                # Process the generated questions
                questions_text = student_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the generated questions (remove the prompt)
                questions_text = questions_text[len(prompt):]
                
                # Split into individual questions
                questions = []
                for line in questions_text.split('\n'):
                    # Remove numbering and clean up
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                    if clean_line and not clean_line.isdigit() and len(clean_line) > 10:
                        questions.append(clean_line)
                
                # Ensure we have the requested number of questions
                questions = questions[:num_questions]
                # If we don't have enough questions, add the sentence as a question
                while len(questions) < num_questions:
                    questions.append(f"What does this mean: {sentence[:50]}...?")
                
                batch_questions.append((sentence, questions))
            except Exception as e:
                logging.error(f"Error generating questions for sentence: {sentence[:50]}...: {e}")
                # Add a fallback question
                batch_questions.append((sentence, [f"What does this mean: {sentence[:50]}...?"]))
        
        all_questions.extend(batch_questions)
    
    return all_questions

def answer_questions_batch(sentence_questions_pairs, batch_size=4, max_length=64):
    """Generate answers for batches of questions."""
    all_results = []
    
    # Flatten the list of questions
    flat_questions = []
    for sentence, questions in sentence_questions_pairs:
        for question in questions:
            flat_questions.append((sentence, question))
    
    for i in range(0, len(flat_questions), batch_size):
        batch_items = flat_questions[i:i+batch_size]
        
        for sentence, question in batch_items:
            # Create a more detailed prompt for answering
            prompt = f"""Answer the following question accurately and concisely:

Question: "{question}"

Answer:"""
            
            # Process one prompt at a time without padding
            input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            try:
                with torch.no_grad():
                    output = student_model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        do_sample=False,
                        pad_token_id=student_tokenizer.eos_token_id
                    )
                
                # Process the generated answer
                answer_text = student_tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract just the generated answer (remove the prompt)
                answer_text = answer_text.split("Answer:")[-1].strip()
                
                # Compute similarity
                similarity = compute_similarity(sentence, answer_text)
                
                # Add to results
                all_results.append({
                    "sentence": sentence,
                    "question": question,
                    "answer": answer_text,
                    "similarity": similarity
                })
            except Exception as e:
                logging.error(f"Error generating answer for question: {question[:50]}...: {e}")
                # Add fallback answer
                all_results.append({
                    "sentence": sentence,
                    "question": question,
                    "answer": "Error generating answer.",
                    "similarity": 0.0
                })
    
    return all_results

def compute_similarity(reference, generated):
    """Compute similarity between reference and generated text using multiple metrics."""
    try:
        # Encode texts
        ref_emb = sim_model.encode(reference, convert_to_tensor=True)
        gen_emb = sim_model.encode(generated, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_sim = util.pytorch_cos_sim(ref_emb, gen_emb).item()
        
        return cosine_sim
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        return 0.0

def evaluate_answer_quality(answer):
    """Evaluate the quality of an answer based on length, structure, etc."""
    # Simple heuristics for answer quality
    if len(answer) < 10:
        return 0.2  # Very short answers are likely low quality
    
    if len(answer.split()) > 50:
        return 0.7  # Long answers might be comprehensive
    
    # Check for presence of explanation indicators
    explanation_words = ['because', 'therefore', 'thus', 'since', 'as a result']
    if any(word in answer.lower() for word in explanation_words):
        return 0.8  # Contains explanation
    
    return 0.5  # Default moderate score

# -----------------------------------------------------
# Main self-study function
# -----------------------------------------------------
def run_self_study(pdf_folder, output_dir):
    """Run the self-study process on all PDFs in the folder."""
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
        pdf_results = process_pdf(pdf_file)
        all_results.extend(pdf_results)
    
    # Save all results to a single JSON file
    results_file = os.path.join(output_dir, f"self_study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"All results saved to {results_file}")
    
    # Generate summary statistics
    generate_summary(all_results, output_dir)
    
    return True

def process_pdf(pdf_file):
    """Process a single PDF file."""
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
    
    # Generate questions for all sentences
    logging.info("Generating questions...")
    sentence_questions = generate_questions_batch(
        filtered_sentences, 
        num_questions=args.num_questions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    # Generate answers for all questions
    logging.info("Generating answers and computing similarities...")
    results = answer_questions_batch(
        sentence_questions,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Add PDF filename to results
    for result in results:
        result["pdf_file"] = os.path.basename(pdf_file)
    
    # Print results if verbose
    if args.verbose:
        for result in results:
            print("\n==========================================")
            print(f"Original Sentence: {result['sentence']}")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Similarity Score: {result['similarity']:.4f}")
            print("==========================================\n")
    
    return results

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
