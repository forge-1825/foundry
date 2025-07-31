# BETA Release Notes

**Version**: 1.0.0-beta  
**Release Date**: January 31, 2025  
**Status**: BETA - Not for production use

## Overview

This is the initial BETA release of Foundry, an AI model distillation pipeline that enables training smaller, efficient models from larger teacher models. This release is intended for testing and feedback purposes.

## What's Included

### Core Features
- PDF text extraction and enrichment
- Teacher-student model distillation
- Hierarchical Q&A pair generation  
- LoRA-based fine-tuning
- Student self-study capabilities
- Model evaluation framework
- Web-based UI for pipeline management

### Model Support
- Teacher: Llama 3 8B Instruct
- Student: Phi-3 Mini 4k Instruct
- vLLM inference engine
- Remote model access via SSH

### Data Sources
- PDF document processing
- Code dataset integration (HuggingFace)
- CVE database parsing
- Manual JSON input support

## Known Limitations

### Functionality
1. **Limited Error Recovery**: Pipeline may halt on errors without graceful recovery
2. **Memory Intensive**: Requires significant RAM/VRAM for model operations
3. **Single GPU Only**: Multi-GPU support is experimental
4. **Hardcoded Paths**: Some scripts contain hardcoded file paths
5. **Windows Compatibility**: Primarily tested on Linux; Windows support is limited

### Performance
1. **Processing Speed**: Large PDF batches may take hours to process
2. **Model Loading**: Initial model loading can take 5-10 minutes
3. **Memory Leaks**: Long-running processes may accumulate memory
4. **No Streaming**: Responses are not streamed, causing UI delays

### User Interface
1. **Basic Error Messages**: Limited error details in UI
2. **No Progress Bars**: Long operations lack progress indicators
3. **Manual Refresh**: Some status updates require page refresh
4. **Limited Validation**: Input validation is minimal

### Data Handling
1. **PDF Limitations**: 
   - Scanned PDFs (images) not supported
   - Complex layouts may extract poorly
   - Tables and figures are skipped

2. **Size Constraints**:
   - Maximum 1000 pages per PDF
   - Maximum 100MB per file
   - Batch processing limited to 50 files

### Model Limitations
1. **Context Length**: Limited to 4k tokens for Phi-3
2. **Language Support**: Optimized for English only
3. **Domain Specificity**: Best for technical/cybersecurity content
4. **Quantization**: Some accuracy loss with quantized models

## Setup Requirements

### Hardware
- **Minimum**: 16GB RAM, 8GB VRAM (GPU)
- **Recommended**: 32GB RAM, 16GB VRAM
- **Storage**: 50GB free space

### Software
- Docker and Docker Compose
- NVIDIA drivers (for GPU)
- Python 3.8+
- CUDA 11.8+ (for GPU)

### Network
- Internet for model downloads
- Ports 8000-8003 available
- SSH access (for remote models)

## Installation Issues

### Common Problems
1. **Docker Permission Errors**: Run with sudo or add user to docker group
2. **CUDA Not Found**: Ensure NVIDIA Container Toolkit is installed
3. **Port Conflicts**: Change ports in docker-compose.yml if needed
4. **Model Download Failures**: Check HuggingFace access and tokens

### Workarounds
- Use CPU mode if GPU unavailable (slower)
- Pre-download models to avoid timeouts
- Use remote models via SSH forwarding
- Reduce batch sizes for memory issues

## Testing Recommendations

### Quick Test
1. Start with small PDF set (5-10 files)
2. Use default parameters
3. Monitor resource usage
4. Check output quality

### Validation Steps
1. Verify model connectivity: Check /api/models endpoint
2. Test extraction: Process single PDF first
3. Validate enrichment: Check enriched_data.json
4. Confirm training: Monitor loss curves
5. Evaluate output: Run evaluation suite

## Known Bugs

1. **Pipeline Hangs**: May freeze on certain PDF types
2. **Memory Errors**: OOM not always caught gracefully  
3. **UI Disconnects**: WebSocket may timeout on long operations
4. **Path Issues**: Windows paths may cause failures
5. **Encoding Bugs**: Unicode in PDFs may cause errors

## Planned Improvements

### Next Release (v1.0.0)
- Improved error handling and recovery
- Progress indicators for all operations
- Multi-GPU support
- Streaming responses
- Better memory management

### Future Releases
- Additional model architectures
- Multi-language support
- Cloud deployment options
- Advanced evaluation metrics
- Plugin system for extensions

## Security Considerations

⚠️ **BETA Security Notice**:
- Default API keys are used (change in production)
- No authentication on web UI
- Models accessible without authorization
- File uploads not sanitized
- Container runs with elevated privileges

## Feedback and Support

### Reporting Issues
- Check existing issues first
- Include full error logs
- Specify environment details
- Provide minimal reproduction steps

### Contributing
- See CONTRIBUTING.md for guidelines
- Test changes thoroughly
- Document new features
- Follow code style conventions

## Migration Notes

When upgrading from this BETA:
1. Backup all generated models
2. Export evaluation results
3. Save configuration files
4. Note custom modifications
5. Plan for potential breaking changes

## Disclaimer

This BETA release is provided "as-is" for testing purposes. It is not recommended for production use. Data loss, system instability, or unexpected behavior may occur. Always backup important data before use.

## Acknowledgments

Thanks to all early testers and contributors who helped shape this BETA release. Your feedback is invaluable for improving Foundry.

---

For detailed setup instructions, see:
- [MODEL_SETUP_GUIDE.md](MODEL_SETUP_GUIDE.md)
- [PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md)
- [SCRIPT_VERSIONS_GUIDE.md](scripts/SCRIPT_VERSIONS_GUIDE.md)