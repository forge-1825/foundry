# Pipeline Workflow Guide

This guide explains how the Foundry model distillation pipeline works, from data extraction to final evaluation.

## Pipeline Overview

The Foundry pipeline implements knowledge distillation - a process where a large "teacher" model trains a smaller "student" model to achieve similar performance with fewer resources.

```
[PDFs/Data] → [Extraction] → [Enrichment] → [Teacher Pairs] → [Distillation] → [Merge] → [Self-Study] → [Evaluation]
```

## Detailed Workflow

### Step 1: Content Extraction & Enrichment

**Purpose**: Extract text from various sources and enrich it with metadata

**Process**:
1. **Input Sources**:
   - PDF files from specified folders
   - Code datasets from HuggingFace
   - CVE data for cybersecurity contexts
   - Manual JSON extracts

2. **Extraction**:
   - PDFs: Uses PyPDF2/pdfplumber for text extraction
   - Code: Downloads and processes programming datasets
   - CVE: Parses vulnerability databases

3. **Enrichment**:
   - Entity extraction (people, organizations, locations)
   - Keyword identification
   - Summary generation
   - Metadata tagging

**Output**: `Output/enriched_data.json`
```json
[{
  "url": "source_file.pdf",
  "text": "raw extracted text",
  "content": "processed content",
  "summary": "AI-generated summary",
  "entities": ["entity1", "entity2"],
  "keywords": ["keyword1", "keyword2"]
}]
```

### Step 2: Teacher Pair Generation

**Purpose**: Create high-quality Q&A pairs using the teacher model

**Process**:
1. **Hierarchical Context Building**:
   - Document level: Overall document understanding
   - Section level: Chapter/section context
   - Paragraph level: Detailed content
   - Sentence level: Specific facts

2. **Question Generation**:
   - Factual questions (What, When, Where)
   - Conceptual questions (How, Why)
   - Application questions (scenarios)
   - Analysis questions (compare/contrast)

3. **Answer Generation**:
   - Teacher model generates comprehensive answers
   - Includes reasoning and explanations
   - Cross-references related content

**Output**: `Output/teacher_pairs.json`
```json
[{
  "input": "What are the key features of network scanning?",
  "target": "Network scanning key features include: 1) Port discovery..."
}]
```

### Step 3: Distillation Training

**Purpose**: Train the student model using teacher-generated pairs

**Process**:
1. **Data Preparation**:
   - Load teacher pairs
   - Tokenize inputs/outputs
   - Create training batches

2. **Training Configuration**:
   - LoRA (Low-Rank Adaptation) for efficient fine-tuning
   - Ensemble learning with multiple teacher responses
   - Adaptive learning rate scheduling

3. **Training Loop**:
   - Forward pass through student model
   - Calculate loss against teacher outputs
   - Backpropagation and weight updates
   - Checkpoint saving at intervals

**Output**: `distilled_model_phi3_improved/`
- `best_checkpoint/`: Best performing model
- `final/`: Final trained model
- Training metrics and logs

### Step 4: Model Merging

**Purpose**: Merge LoRA adapters with base model for deployment

**Process**:
1. Load base model (Phi-3 Mini)
2. Load trained LoRA adapters
3. Merge weights into single model
4. Convert to deployment format
5. Optimize for inference

**Output**: `Output/merged/`
- Complete model ready for serving
- Optimized for inference speed

### Step 5: Student Self-Study

**Purpose**: Further improve model through self-directed learning

**Process**:
1. **Document Analysis**:
   - Student reads new documents
   - Generates questions about content
   - Attempts to answer own questions

2. **Teacher Verification** (optional):
   - Teacher model reviews student Q&A
   - Provides feedback and corrections
   - Student learns from feedback

3. **Iterative Refinement**:
   - Multiple passes over content
   - Progressive difficulty increase
   - Focus on weak areas

**Output**: `Output/self_study_results/`
```json
[{
  "sentence": "Original sentence from document",
  "questions": [{
    "question": "Student-generated question",
    "student_answer": "Student's attempt",
    "teacher_answer": "Teacher's correction",
    "similarity_score": 0.85
  }]
}]
```

### Step 6: Model Evaluation

**Purpose**: Assess model performance across various metrics

**Process**:
1. **Benchmark Datasets**:
   - Simple Q&A tasks
   - Command extraction
   - Error analysis
   - Domain-specific tests

2. **Evaluation Metrics**:
   - Accuracy/F1 scores
   - Perplexity
   - Response quality
   - Inference speed

3. **Comparison**:
   - Student vs Teacher performance
   - Before/after distillation
   - Domain-specific improvements

**Output**: `results/evaluation_results.json`
- Detailed metrics per dataset
- Comparative analysis
- Performance visualizations

## Data Flow

### File Dependencies
```
PDFs → enriched_data.json → teacher_pairs.json → model_checkpoint → merged_model → study_results → evaluation_metrics
```

### Key Transformations
1. **Raw Text → Structured Data**: Extraction adds metadata
2. **Content → Q&A Pairs**: Teacher creates training examples  
3. **Training Data → Model Weights**: Distillation updates parameters
4. **Adapters → Full Model**: Merging creates deployment model
5. **Model → Performance Metrics**: Evaluation quantifies success

## Configuration Options

### Global Settings
- `output_dir`: Where to save all outputs
- `use_gpu`: Enable GPU acceleration
- `batch_size`: Processing batch size
- `max_samples`: Limit data for testing

### Per-Step Settings

**Enrichment**:
- Entity extraction models
- Summary length
- Keyword count

**Teacher Generation**:
- Number of Q&A pairs
- Question types
- Context window size

**Distillation**:
- Learning rate
- Number of epochs
- LoRA rank
- Gradient accumulation

**Self-Study**:
- Topics of interest
- Question difficulty
- Iteration count

## Best Practices

1. **Data Quality**:
   - Ensure PDFs are text-based (not scanned images)
   - Remove duplicates before processing
   - Validate extraction quality

2. **Resource Management**:
   - Monitor GPU memory usage
   - Use appropriate batch sizes
   - Enable gradient checkpointing for large models

3. **Training Strategy**:
   - Start with small datasets for testing
   - Use validation sets to prevent overfitting
   - Save checkpoints frequently

4. **Evaluation**:
   - Use diverse test sets
   - Compare against baseline models
   - Consider domain-specific metrics

## Troubleshooting Common Issues

### Pipeline Failures

1. **Out of Memory**:
   - Reduce batch size
   - Enable gradient accumulation
   - Use model quantization

2. **Poor Quality Outputs**:
   - Check teacher model connectivity
   - Verify data enrichment quality
   - Adjust generation parameters

3. **Training Instability**:
   - Lower learning rate
   - Increase warm-up steps
   - Check for data anomalies

### Data Issues

1. **Empty Outputs**:
   - Verify input file paths
   - Check file permissions
   - Validate JSON structure

2. **Encoding Errors**:
   - Ensure UTF-8 encoding
   - Handle special characters
   - Clean text preprocessing

## Advanced Usage

### Custom Pipelines
- Skip steps with direct file inputs
- Combine multiple data sources
- Implement custom evaluators

### Distributed Training
- Multi-GPU support via accelerate
- Distributed data parallel training
- Model parallel for large models

### Production Deployment
- Model quantization for efficiency
- API endpoint configuration
- Monitoring and logging setup

## Next Steps

1. Review [MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md) for model configuration
2. Check [SCRIPT_VERSIONS_GUIDE.md](SCRIPT_VERSIONS_GUIDE.md) for script selection
3. See [BETA_RELEASE_NOTES.md](BETA_RELEASE_NOTES.md) for known limitations
4. Run tests with `tests/run_tests.sh`