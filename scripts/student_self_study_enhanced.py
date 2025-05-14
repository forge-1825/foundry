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
from openai import OpenAI

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
parser = argparse.ArgumentParser(description="Enhanced Self-study module with domain-specific knowledge")
parser.add_argument("--pdf_folder", type=str, help="Path to folder containing PDF files")
parser.add_argument("--model_path", type=str, default="./distilled_model_phi3_improved/best_checkpoint",
                    help="Path to the distilled student model")
parser.add_argument("--output_dir", type=str, default="./self_study_results",
                    help="Directory to save results")
parser.add_argument("--num_questions", type=int, default=3,
                    help="Number of questions to generate per sentence")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for processing")
parser.add_argument("--min_sentence_length", type=int, default=10,
                    help="Minimum number of words in a sentence to process")
parser.add_argument("--max_sentence_length", type=int, default=50,
                    help="Maximum number of words in a sentence to process")
parser.add_argument("--temperature", type=float, default=0.5,
                    help="Temperature for question generation")
parser.add_argument("--max_length", type=int, default=64,
                    help="Maximum length for generated text")
parser.add_argument("--max_sentences", type=int, default=10,
                    help="Maximum number of sentences to process per PDF")
parser.add_argument("--use_8bit", action="store_true",
                    help="Load model in 8-bit precision to save memory")
parser.add_argument("--checkpoint_interval", type=int, default=5,
                    help="Save checkpoint after processing this many sentences")
parser.add_argument("--use_teacher", action="store_true",
                    help="Use teacher model (Phi-4) for answer verification")
parser.add_argument("--teacher_port", type=int, default=8000,
                    help="Port for the teacher model vLLM server")
parser.add_argument("--domain_context", type=str, default="",
                    help="Path to domain context file for better answers")
parser.add_argument("--iterative_refinement", action="store_true",
                    help="Use iterative refinement for better answers")
parser.add_argument("--use_hierarchical_context", action="store_true",
                    help="Use hierarchical context encoding with paragraph summaries")
parser.add_argument("--max_paragraph_size", type=int, default=5,
                    help="Maximum number of sentences to group into a paragraph")
parser.add_argument("--include_reasoning", action="store_true",
                    help="Include chain-of-thought reasoning in prompts")
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

logging.info("Starting enhanced student self-study with domain-specific knowledge")
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
# Load domain context if provided
# -----------------------------------------------------
domain_context = ""
if args.domain_context and os.path.exists(args.domain_context):
    try:
        with open(args.domain_context, "r", encoding="utf-8") as f:
            domain_context = f.read()
        logging.info(f"Loaded domain context from {args.domain_context} ({len(domain_context)} characters)")
    except Exception as e:
        logging.error(f"Error loading domain context: {e}")

# -----------------------------------------------------
# Initialize teacher model client if requested
# -----------------------------------------------------
teacher_client = None
if args.use_teacher:
    try:
        from openai import OpenAI
        teacher_client = OpenAI(
            base_url=f"http://localhost:{args.teacher_port}/v1",
            api_key="not-needed"  # vLLM doesn't require an API key
        )

        # Test the teacher model
        response = teacher_client.completions.create(
            model="jakiAJK/microsoft-phi-4_GPTQ-int4",  # Model name from Docker logs
            prompt="test",
            max_tokens=1,
            temperature=0.7,
            n=1,
            stop=None
        )
        logging.info("Teacher model (Phi-4) is available and responding")
    except Exception as e:
        logging.error(f"Error connecting to teacher model: {e}")
        logging.warning("Proceeding without teacher model verification")
        args.use_teacher = False

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

                base_model_name = adapter_config.get("base_model_name_or_path", "microsoft/phi-3")
            else:
                base_model_name = "microsoft/phi-3"  # Default if config not found

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
# Hierarchical context and chain-of-thought functions
# -----------------------------------------------------
def group_sentences_into_paragraphs(sentences, max_paragraph_size=5):
    """
    Group sentences into paragraphs or contextual blocks.

    Args:
        sentences (list): List of sentences
        max_paragraph_size (int): Maximum number of sentences per paragraph

    Returns:
        list: List of paragraphs, where each paragraph is a list of sentences
    """
    paragraphs = []
    for i in range(0, len(sentences), max_paragraph_size):
        paragraph = sentences[i:i + max_paragraph_size]
        paragraphs.append(paragraph)
    return paragraphs

def generate_context_summary(paragraph, domain_context="", max_length=100):
    """
    Generate a summary with chain-of-thought reasoning for a paragraph.

    Args:
        paragraph (str): The paragraph or group of sentences to summarize
        domain_context (str): Optional domain context for better understanding
        max_length (int): Maximum length of the generated summary

    Returns:
        tuple: (summary, reasoning) - The generated summary and reasoning
    """
    # Skip if hierarchical context is not enabled
    if not args.use_hierarchical_context:
        return "", ""

    # Create a prompt that encourages chain-of-thought reasoning
    domain_prefix = ""
    if domain_context:
        domain_prefix = f"Using the following domain knowledge:\n{domain_context[:500]}...\n\n"

    prompt = f"""{domain_prefix}Summarize the key technical details in this paragraph.
First, explain your reasoning step-by-step, then provide a concise summary.

Paragraph: "{paragraph}"

Step-by-step reasoning:"""

    # Process the prompt
    input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output = student_model.generate(
                input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=student_tokenizer.eos_token_id
            )

        # Process the generated text
        generated_text = student_tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract the reasoning and summary
        full_output = generated_text.split("Step-by-step reasoning:")[-1].strip()

        # Split into reasoning and summary (assuming the model follows the format)
        parts = re.split(r'Summary:|Concise summary:', full_output, maxsplit=1)

        if len(parts) > 1:
            reasoning = parts[0].strip()
            summary = parts[1].strip()
        else:
            # If the model didn't follow the format exactly
            reasoning = full_output[:len(full_output)//2]
            summary = full_output[len(full_output)//2:]

        return summary, reasoning
    except Exception as e:
        logging.error(f"Error generating context summary: {e}")
        return paragraph[:100] + "...", "Unable to generate reasoning."

# -----------------------------------------------------
# Functions using the student model to generate questions and answers
# -----------------------------------------------------
def generate_domain_specific_question(sentence, context_summary="", context_reasoning="", domain_context="", temperature=0.5, max_length=64):
    """Generate a domain-specific question for a sentence with context."""
    # Create a more focused prompt with domain context and paragraph context
    domain_prefix = ""
    if domain_context:
        domain_prefix = f"Using the following domain knowledge:\n{domain_context[:500]}...\n\n"

    context_prefix = ""
    if context_summary and args.use_hierarchical_context:
        context_prefix = f"Paragraph context: {context_summary}\n\n"
        if context_reasoning and args.include_reasoning:
            context_prefix += f"Reasoning about the paragraph: {context_reasoning}\n\n"

    prompt = f"""{domain_prefix}{context_prefix}Generate a specific, technical question that tests deep understanding of the following sentence.
The question should require domain expertise to answer correctly and should relate to the broader context of the paragraph.

Sentence: "{sentence}"

Technical Question:"""

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
        question_text = question_text.split("Technical Question:")[-1].strip()

        # Clean up the question
        if not question_text.endswith('?'):
            question_text += '?'

        return question_text
    except Exception as e:
        logging.error(f"Error generating question for sentence: {sentence[:50]}...: {e}")
        # Return a fallback question
        return f"What are the technical implications of: {sentence[:30]}...?"

def generate_domain_specific_answer(question, sentence, context_summary="", context_reasoning="", domain_context="", max_length=64):
    """Generate a domain-specific answer for a question with context."""
    # Create a focused prompt for answering with domain context and paragraph context
    domain_prefix = ""
    if domain_context:
        domain_prefix = f"Using the following domain knowledge:\n{domain_context[:500]}...\n\n"

    context_prefix = ""
    if context_summary and args.use_hierarchical_context:
        context_prefix = f"Paragraph context: {context_summary}\n\n"
        if context_reasoning and args.include_reasoning:
            context_prefix += f"Reasoning about the paragraph: {context_reasoning}\n\n"

    # Determine if we should use chain-of-thought prompting
    if args.include_reasoning:
        prompt = f"""{domain_prefix}{context_prefix}Answer this technical question with specific, accurate details.
First, explain your reasoning step-by-step, then provide the final answer.

Context: {sentence}

Question: {question}

Step-by-step reasoning:"""
    else:
        prompt = f"""{domain_prefix}{context_prefix}Answer this technical question with specific, accurate details:

Context: {sentence}

Question: {question}

Technical Answer:"""

    # Process one prompt at a time without padding
    input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output = student_model.generate(
                input_ids,
                max_new_tokens=max_length,
                do_sample=True,  # Use sampling for more diverse answers
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.9,
                pad_token_id=student_tokenizer.eos_token_id
            )

        # Process the generated answer
        answer_text = student_tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the generated answer (remove the prompt)
        answer_text = answer_text.split("Technical Answer:")[-1].strip()

        return answer_text
    except Exception as e:
        logging.error(f"Error generating answer for question: {question}: {e}")
        # Return fallback answer
        return "Unable to generate a technical answer for this question."

def verify_with_teacher(question, sentence, student_answer, max_tokens=100):
    """Verify and improve the student's answer using the teacher model."""
    if not args.use_teacher or teacher_client is None:
        return student_answer, 0.0

    try:
        # Create a prompt for the teacher to verify and improve the answer
        prompt = f"""Verify and improve this technical answer:

Context: {sentence}

Question: {question}

Student's Answer: {student_answer}

Improved Technical Answer:"""

        # Get the teacher's response
        response = teacher_client.completions.create(
            model="jakiAJK/microsoft-phi-4_GPTQ-int4",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            n=1,
            stop=None
        )

        # Extract the improved answer
        improved_answer = response.choices[0].text.strip()

        # Calculate similarity between student and teacher answers
        if sim_model is not None:
            student_emb = sim_model.encode(student_answer, convert_to_tensor=True)
            teacher_emb = sim_model.encode(improved_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(student_emb, teacher_emb).item()
        else:
            similarity = word_overlap_similarity(student_answer, improved_answer)

        return improved_answer, similarity
    except Exception as e:
        logging.error(f"Error verifying with teacher: {e}")
        return student_answer, 0.0

def iterative_refinement(question, sentence, initial_answer, domain_context="", max_length=64, iterations=2):
    """Iteratively refine the answer to improve quality."""
    if not args.iterative_refinement:
        return initial_answer

    current_answer = initial_answer

    for i in range(iterations):
        try:
            # Create a prompt for refinement
            prompt = f"""Improve this technical answer to be more specific, accurate, and detailed:

Context: {sentence}

Question: {question}

Current Answer: {current_answer}

Improved Answer:"""

            # Process the prompt
            input_ids = student_tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = student_model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    do_sample=False,  # Deterministic for refinement
                    temperature=0.3,
                    pad_token_id=student_tokenizer.eos_token_id
                )

            # Process the refined answer
            refined_text = student_tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract just the refined answer
            refined_text = refined_text.split("Improved Answer:")[-1].strip()

            # Update the current answer if the refinement is better
            if len(refined_text) > len(current_answer) * 0.8:  # Ensure it's not too short
                current_answer = refined_text

            # Clear memory
            clear_gpu_memory()

        except Exception as e:
            logging.error(f"Error in refinement iteration {i+1}: {e}")
            break

    return current_answer

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

def evaluate_answer_quality(answer, question):
    """Evaluate the quality of an answer based on multiple metrics."""
    # Length-based score (0.0-0.3)
    length_score = min(len(answer.split()) / 50, 1.0) * 0.3

    # Specificity score (0.0-0.3)
    specificity_words = ['specifically', 'precisely', 'exactly', 'particularly', 'notably',
                         'especially', 'primarily', 'mainly', 'chiefly', 'principally']
    specificity_score = min(sum(1 for word in specificity_words if word.lower() in answer.lower()) / 3, 1.0) * 0.3

    # Technical terms score (0.0-0.4)
    # Extract potential technical terms (capitalized words, words with numbers, etc.)
    technical_terms = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b|\b[a-z]+[0-9]+[a-zA-Z0-9]*\b|\b[a-z]+\-[a-zA-Z0-9]+\b', answer)
    technical_score = min(len(technical_terms) / 5, 1.0) * 0.4

    # Combine scores
    quality_score = length_score + specificity_score + technical_score

    return quality_score

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
    """Process a single PDF file with paragraph-level context."""
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

    # Group sentences into paragraphs if hierarchical context is enabled
    if args.use_hierarchical_context:
        paragraphs = group_sentences_into_paragraphs(filtered_sentences, max_paragraph_size=args.max_paragraph_size)
        logging.info(f"Grouped sentences into {len(paragraphs)} paragraph(s).")
    else:
        # If hierarchical context is not enabled, treat each sentence as its own "paragraph"
        paragraphs = [[sentence] for sentence in filtered_sentences]

    pdf_results = []
    checkpoint_count = 0

    # Process one paragraph at a time
    for p_idx, paragraph_sentences in enumerate(paragraphs):
        # Generate context summary for this paragraph if hierarchical context is enabled
        context_summary = ""
        context_reasoning = ""
        if args.use_hierarchical_context:
            paragraph_text = " ".join(paragraph_sentences)
            logging.info(f"Processing paragraph {p_idx+1}/{len(paragraphs)} with {len(paragraph_sentences)} sentence(s)")

            # Generate context summary with reasoning
            context_summary, context_reasoning = generate_context_summary(
                paragraph_text,
                domain_context=domain_context,
                max_length=150
            )

            if context_summary:
                logging.info(f"Generated context summary: {context_summary[:100]}...")
                if args.verbose:
                    print(f"\nParagraph: {paragraph_text[:200]}...")
                    print(f"Context Summary: {context_summary}")
                    if args.include_reasoning:
                        print(f"Reasoning: {context_reasoning[:200]}...")

        # Process each sentence in the paragraph
        for s_idx, sentence in enumerate(paragraph_sentences):
            sentence_results = []
            if args.use_hierarchical_context:
                logging.info(f"Processing sentence {s_idx+1}/{len(paragraph_sentences)} in paragraph {p_idx+1}")
            else:
                logging.info(f"Processing sentence {p_idx+1}/{len(paragraphs)}")

            # Generate questions for this sentence
            for q in range(args.num_questions):
                try:
                    # Generate a domain-specific question with context
                    question = generate_domain_specific_question(
                        sentence,
                        context_summary=context_summary,
                        context_reasoning=context_reasoning,
                        domain_context=domain_context,
                        temperature=args.temperature,
                        max_length=args.max_length
                    )

                    # Generate a domain-specific answer with context
                    student_answer = generate_domain_specific_answer(
                        question,
                        sentence,
                        context_summary=context_summary,
                        context_reasoning=context_reasoning,
                        domain_context=domain_context,
                        max_length=args.max_length
                    )

                    # Iteratively refine the answer if requested
                    if args.iterative_refinement:
                        student_answer = iterative_refinement(
                            question,
                            sentence,
                            student_answer,
                            domain_context=domain_context,
                            max_length=args.max_length
                        )

                    # Verify with teacher if requested
                    final_answer = student_answer
                    teacher_similarity = 0.0
                    if args.use_teacher:
                        teacher_answer, teacher_similarity = verify_with_teacher(
                            question,
                            sentence,
                            student_answer
                        )
                        if teacher_similarity < 0.7:  # If teacher's answer is significantly different
                            final_answer = teacher_answer

                    # Compute similarity and quality metrics
                    similarity = compute_similarity(sentence, final_answer)
                    quality_score = evaluate_answer_quality(final_answer, question)

                    # Add to results
                    result = {
                        "sentence": sentence,
                        "question": question,
                        "student_answer": student_answer,
                        "final_answer": final_answer,
                        "similarity": similarity,
                        "quality_score": quality_score,
                        "teacher_similarity": teacher_similarity,
                        "pdf_file": os.path.basename(pdf_file),
                        "used_teacher": args.use_teacher and teacher_similarity < 0.7,
                        "used_refinement": args.iterative_refinement,
                        "paragraph_context": context_summary if args.use_hierarchical_context else ""
                    }

                    sentence_results.append(result)

                    # Print if verbose
                    if args.verbose:
                        print(f"\nSentence: {sentence}")
                        print(f"Question: {question}")
                        print(f"Student Answer: {student_answer}")
                        if args.use_teacher and teacher_similarity < 0.7:
                            print(f"Teacher Answer: {final_answer}")
                        print(f"Similarity: {similarity:.4f}")
                        print(f"Quality Score: {quality_score:.4f}")

                    # Clear memory after each question
                    clear_gpu_memory()

                except Exception as e:
                    logging.error(f"Error processing question {q+1} for sentence {s_idx+1} in paragraph {p_idx+1}: {e}")
                    continue

        # Add results for this sentence
        pdf_results.extend(sentence_results)

        # Save checkpoint at intervals
        checkpoint_count += 1
        if checkpoint_count >= args.checkpoint_interval:
            checkpoint_file = os.path.join(
                args.output_dir,
                f"checkpoint_{os.path.basename(pdf_file)}_{p_idx+1}.json"
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
    quality_scores = [r["quality_score"] for r in results]

    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)

    avg_quality = np.mean(quality_scores)
    median_quality = np.median(quality_scores)
    min_quality = np.min(quality_scores)
    max_quality = np.max(quality_scores)

    # Calculate teacher usage statistics if applicable
    teacher_used = [r for r in results if r.get("used_teacher", False)]
    teacher_usage_percent = len(teacher_used) / len(results) * 100 if results else 0

    # Group by PDF file
    pdf_stats = {}
    for result in results:
        pdf_file = result.get("pdf_file", "unknown")
        if pdf_file not in pdf_stats:
            pdf_stats[pdf_file] = {
                "similarities": [],
                "quality_scores": []
            }
        pdf_stats[pdf_file]["similarities"].append(result["similarity"])
        pdf_stats[pdf_file]["quality_scores"].append(result["quality_score"])

    # Calculate per-PDF statistics
    pdf_summaries = {}
    for pdf_file, stats in pdf_stats.items():
        pdf_summaries[pdf_file] = {
            "count": len(stats["similarities"]),
            "avg_similarity": np.mean(stats["similarities"]),
            "avg_quality": np.mean(stats["quality_scores"]),
            "min_similarity": np.min(stats["similarities"]),
            "max_similarity": np.max(stats["similarities"]),
            "min_quality": np.min(stats["quality_scores"]),
            "max_quality": np.max(stats["quality_scores"])
        }

    # Create summary dictionary
    summary = {
        "overall": {
            "total_questions": len(results),
            "avg_similarity": avg_similarity,
            "median_similarity": median_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "avg_quality": avg_quality,
            "median_quality": median_quality,
            "min_quality": min_quality,
            "max_quality": max_quality,
            "teacher_usage_percent": teacher_usage_percent if args.use_teacher else "N/A"
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
    print(f"Average quality score: {avg_quality:.4f}")
    print(f"Teacher model usage: {teacher_usage_percent:.1f}%" if args.use_teacher else "Teacher model not used")
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
        logging.info(f"Enhanced self-study process completed in {end_time - start_time:.2f} seconds.")
    else:
        logging.error("Enhanced self-study process failed.")
