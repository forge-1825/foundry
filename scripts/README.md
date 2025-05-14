# Model Distillation Pipeline Scripts

This directory contains the Python scripts that are executed by the Model Distillation Pipeline Web UI.

## Scripts

1. **manual_extractor.py** - Data extraction script
   - Downloads web documents and extracts content to PDF and JSON formats

2. **data_enrichment.py** - Data enrichment script
   - Processes the extracted JSON file to clean text, extract entities, and summarize content

3. **teacher_pair_generation.py** - Teacher pair generation script
   - Queries a teacher model to generate "soft" target outputs from enriched records

4. **distillation.py** - Distillation training script
   - Trains the student model using the teacher pairs

5. **evaluate_distilled.py** - Model evaluation script
   - Evaluates the distilled model against the original model

## Usage

These scripts are executed by the web UI and should not be run directly unless for testing purposes.

When the application is started, the scripts from the parent directory are copied to this directory using the `copy_scripts.py` script.
