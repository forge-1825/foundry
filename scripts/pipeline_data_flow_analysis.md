# Pipeline Data Flow Analysis

## Overview
This document analyzes the data flow between pipeline steps in the Model Distillation UI, including output formats, input expectations, file paths, and data compatibility.

## Pipeline Steps and Data Flow

### 1. Content Extraction & Enrichment
**Script**: `data_enrichment.py` (or enhanced versions)
**Input**: 
- PDF files from a specified folder
- OR JSON file from manual extractor
- Optional: Domain context file

**Output**: 
- **File**: `Output/enriched_data.json`
- **Format**: JSON array of enriched records
```json
[
  {
    "url": "path/to/file.pdf",
    "text": "extracted text content",
    "content": "processed content",
    "summary": "generated summary",
    "entities": ["entity1", "entity2"],
    "keywords": ["keyword1", "keyword2"],
    "metadata": {...}
  }
]
```

**Key Parameters**:
- `--input-file`: Input JSON from manual extractor
- `--output-file`: Output JSON path (default: `/data/enriched/enriched_data.json`)
- `--input-folder`: PDF folder path

### 2. Teacher Pair Generation
**Script**: `teacher_pair_generation_vllm_hierarchical.py`
**Input**: 
- **File**: `Output/enriched_data.json` (from step 1)
- **Expected Format**: JSON array with enriched records containing:
  - `content`: Main text content
  - `summary`: Summary text
  - `entities`: List of entities
  - `keywords`: List of keywords

**Output**:
- **File**: `Output/teacher_pairs.json`
- **Format**: JSON array of Q&A pairs
```json
[
  {
    "input": "question or prompt",
    "target": "expected answer/response"
  }
]
```

**Key Parameters**:
- `--input_file`: Path to enriched data (default: `Output/enriched_data.json`)
- `--output_file`: Path to save teacher pairs (default: `Output/teacher_pairs.json`)
- `--max_pairs`: Maximum number of pairs to generate

### 3. Distillation Training
**Script**: `distillation_vllm_faster_improved.py`
**Input**:
- **File**: `Output/teacher_pairs.json` (from step 2)
- **Expected Format**: JSON array with fields:
  - `input`: Question/prompt text
  - `target`: Answer/response text

**Output**:
- **Directory**: `./distilled_model_phi2_improved/` (or similar)
- **Contents**:
  - `best_checkpoint/`: Best model checkpoint
  - `final/`: Final model checkpoint
  - Model config, tokenizer files, LoRA adapters

**Key Parameters**:
- Teacher pairs file paths (hardcoded, tries multiple locations)
- Model architecture configurations

### 4. Model Merging
**Script**: `merge_model.py`
**Input**:
- **Directory**: `./distilled_model_phi2/best_checkpoint` or `./distilled_model_phi2/final`
- **Expected**: LoRA adapter files and configuration

**Output**:
- **Directory**: `./merged_distilled_phi2/`
- **Contents**: Merged model files ready for serving

**Key Parameters**:
- Adapter path (hardcoded)
- Base model: `microsoft/phi-2`

### 5. Student Self-Study
**Script**: `student_self_study_enhanced.py`
**Input**:
- **PDF Folder**: Path to PDF files for self-study
- **Model Path**: `./distilled_model_phi3_improved/best_checkpoint` (or merged model)

**Output**:
- **Directory**: `./self_study_results/`
- **Files**:
  - `self_study_results_YYYYMMDD_HHMMSS.json`: Main results
  - `checkpoint_*.json`: Per-PDF checkpoints
  - `summary_YYYYMMDD_HHMMSS.json`: Summary statistics

**Output Format**:
```json
[
  {
    "sentence": "original sentence",
    "questions": [
      {
        "question": "generated question",
        "student_answer": "student model response",
        "teacher_answer": "teacher model response (if enabled)",
        "similarity_score": 0.85,
        "quality_score": 0.90
      }
    ],
    "context": "paragraph context",
    "summary": "paragraph summary"
  }
]
```

**Key Parameters**:
- `--pdf_folder`: Input PDF folder
- `--model_path`: Path to distilled model
- `--output_dir`: Results directory
- `--use_teacher`: Enable teacher verification

### 6. Model Evaluation
**Script**: `run_openevals_langchain.py`
**Input**:
- Model endpoints (via vLLM servers)
- Evaluation datasets in `datasets/` directory
- Configuration files

**Output**:
- **Directory**: `results/`
- **Files**: Evaluation results in JSON format

## Data Flow Issues and Compatibility

### Critical Path Dependencies:
1. **Enrichment → Teacher Generation**: 
   - Teacher generation expects `content`, `summary`, `entities`, and `keywords` fields
   - If enrichment fails or produces incomplete data, teacher generation may skip records

2. **Teacher Generation → Distillation**:
   - Distillation expects `input` and `target` fields
   - Empty or missing targets will be regenerated using ensemble approach

3. **Distillation → Merge**:
   - Merge expects specific directory structure with LoRA adapters
   - Hardcoded paths may cause issues if output directories change

4. **Model Path Inconsistencies**:
   - Different scripts use different model architectures (Phi-2 vs Phi-3)
   - Path conventions vary between scripts

### File Format Compatibility:
- All JSON files use UTF-8 encoding
- Array structure is consistent across pipeline
- Field names must match exactly between steps

### Path Configuration Issues:
1. **Hardcoded Paths**: Many scripts have hardcoded paths that may not match actual locations
2. **Docker vs Local**: Some scripts assume Docker container paths (`/data/`), others use local paths
3. **Windows vs Linux**: Path separators and drive letters may cause issues

## Recommendations

1. **Standardize Output Paths**: Use consistent output directory structure across all steps
2. **Add Path Validation**: Check if input files exist before processing
3. **Implement Schema Validation**: Validate JSON structure between steps
4. **Use Configuration Files**: Move hardcoded paths to configuration
5. **Add Data Flow Logging**: Log input/output file paths and record counts
6. **Handle Missing Fields**: Add defaults for optional fields
7. **Consistent Model Naming**: Use same model architecture throughout pipeline

## Pipeline Configuration Reference

From `pipeline_routes.py`, the expected flow is:
1. `content_extraction_enrichment` → 
2. `teacher_pair_generation` → 
3. `distillation` → 
4. `merge_model` → 
5. `student_self_study` → 
6. `evaluation`

Each step should produce output that exactly matches the input requirements of the next step.