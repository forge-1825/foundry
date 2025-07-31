# Script Versions Guide

This guide explains the different versions of scripts in the Foundry pipeline and their purposes.

## Overview
Multiple script versions exist to support different features, optimizations, and testing approaches. This guide helps you understand which version to use for your specific needs.

## Data Enrichment Scripts

### `data_enrichment.py` (ACTIVE - Default)
- **Purpose**: Standard data enrichment script
- **Status**: Production ready
- **Features**: Basic PDF extraction, text enrichment, entity extraction
- **Use When**: Running standard pipeline

### `data_enrichment_enhanced_gpu_fixed_v2.py`
- **Purpose**: GPU-optimized version with enhanced features
- **Status**: Experimental
- **Features**: GPU acceleration, enhanced entity extraction, improved summarization
- **Use When**: Have GPU available and need better quality enrichment

### `data_enrichment_improved.py`
- **Purpose**: Improved version with better error handling
- **Status**: Testing
- **Features**: Better PDF handling, robust error recovery
- **Use When**: Processing problematic PDFs

## Teacher Pair Generation Scripts

### `teacher_pair_generation_vllm_hierarchical.py` (ACTIVE - Default)
- **Purpose**: Hierarchical context-aware pair generation
- **Status**: Production ready
- **Features**: Multi-level context, better question quality
- **Use When**: Standard pipeline execution

### `teacher_pair_generation_vllm.py`
- **Purpose**: Basic vLLM-based pair generation
- **Status**: Deprecated
- **Features**: Simple Q&A generation
- **Use When**: Need baseline comparison

### `teacher_pair_generation_vllm_improved.py`
- **Purpose**: Improved version with better prompts
- **Status**: Testing
- **Features**: Enhanced prompt engineering, quality checks
- **Use When**: Experimenting with prompt variations

### `teacher_pair_generation_vllm_ssh.py`
- **Purpose**: SSH-aware version for remote model access
- **Status**: Production ready
- **Features**: SSH tunnel support, remote model connectivity
- **Use When**: Using models on remote servers via SSH

### `teacher_pair_generation_vllmMITRE.py`
- **Purpose**: MITRE-specific cybersecurity pair generation
- **Status**: Specialized
- **Features**: MITRE ATT&CK framework integration
- **Use When**: Creating cybersecurity-focused training data

### `teacher_pair_generation_vllm_fixed.py`
- **Purpose**: Bug fixes for edge cases
- **Status**: Maintenance
- **Features**: Stability improvements
- **Use When**: Encountering issues with main version

### `teacher_pair_generation.py`
- **Purpose**: Original non-vLLM version
- **Status**: Legacy
- **Features**: Uses HuggingFace transformers directly
- **Use When**: vLLM is not available

## Distillation Scripts

### `distillation_vllm_faster_improved.py` (ACTIVE - Default)
- **Purpose**: Fast distillation with vLLM optimization
- **Status**: Production ready
- **Features**: Speed optimizations, LoRA support, ensemble learning
- **Use When**: Standard pipeline execution

### `distillation.py`
- **Purpose**: Original distillation implementation
- **Status**: Legacy
- **Features**: Basic distillation without optimizations
- **Use When**: Baseline comparison needed

### `distillation_enhanced.py`
- **Purpose**: Enhanced features for distillation
- **Status**: Experimental
- **Features**: Advanced loss functions, curriculum learning
- **Use When**: Research and experimentation

## Student Self-Study Scripts

### `student_self_study_enhanced.py` (ACTIVE - Default)
- **Purpose**: Enhanced self-study with multiple features
- **Status**: Production ready
- **Features**: RAG, TabOut, hierarchical context, teacher verification
- **Use When**: Standard pipeline execution

### `student_self_study_stable.py`
- **Purpose**: Stable version without experimental features
- **Status**: Fallback option
- **Features**: Core self-study functionality only
- **Use When**: Enhanced version has issues

### `student_self_study_improved.py`
- **Purpose**: Improved version with better question generation
- **Status**: Testing
- **Features**: Better question quality, improved context handling
- **Use When**: Testing new features

## Evaluation Scripts

### `run_openevals_langchain.py` (ACTIVE - Default)
- **Purpose**: Comprehensive evaluation using LangChain
- **Status**: Production ready
- **Features**: Multiple evaluators, dataset support, comparison metrics
- **Use When**: Full model evaluation needed

### `evaluate_model.py`
- **Purpose**: Simple model evaluation
- **Status**: Utility
- **Features**: Basic perplexity and accuracy metrics
- **Use When**: Quick model checks

### `evaluate_distilled.py`
- **Purpose**: Specific evaluation for distilled models
- **Status**: Specialized
- **Features**: Distillation-specific metrics
- **Use When**: Analyzing distillation quality

## Utility Scripts

### `manual_extractor.py`
- **Purpose**: Manual PDF text extraction
- **Status**: Utility
- **Features**: Simple PDF to JSON conversion
- **Use When**: Need to manually extract PDFs

### `query_model.py`
- **Purpose**: Interactive model querying
- **Status**: Utility
- **Features**: Command-line model interaction
- **Use When**: Testing models interactively

### `merge_model.py`
- **Purpose**: Merge LoRA adapters with base model
- **Status**: Production ready
- **Features**: Adapter merging, safetensor support
- **Use When**: Preparing model for deployment

### `version_manager.py`
- **Purpose**: Manage script versions
- **Status**: Utility
- **Features**: Version tracking and switching
- **Use When**: Managing multiple versions

### `memory_monitor.py`
- **Purpose**: Monitor GPU/CPU memory usage
- **Status**: Utility
- **Features**: Real-time memory tracking
- **Use When**: Debugging memory issues

### `check_cuda.py`
- **Purpose**: Verify CUDA installation
- **Status**: Utility
- **Features**: CUDA availability check
- **Use When**: Setting up GPU environment

### `create_mock_datasets.py`
- **Purpose**: Generate test datasets
- **Status**: Testing
- **Features**: Mock data generation
- **Use When**: Testing without real data

## Legacy Directory
The `legacy/` directory contains older versions of scripts that are no longer actively maintained but kept for reference.

## Recommendations

1. **For Production Use**: Stick to scripts marked as "ACTIVE - Default"
2. **For Testing**: Use "Testing" or "Experimental" versions
3. **For Debugging**: Use "Stable" or "Fixed" versions as fallbacks
4. **For Remote Setup**: Use SSH-aware versions when applicable

## Version Selection Strategy

When running the pipeline:
1. Start with default (ACTIVE) versions
2. If issues occur, check this guide for alternative versions
3. Use specialized versions for specific use cases (e.g., MITRE for cybersecurity)
4. Fall back to stable versions if enhanced versions fail

## Future Consolidation

In future releases, we plan to:
- Consolidate similar versions into single configurable scripts
- Move experimental features to configuration flags
- Archive truly deprecated versions
- Maintain backward compatibility during transition