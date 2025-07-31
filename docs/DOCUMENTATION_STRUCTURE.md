# Foundry Documentation Architecture

## Overview
This document outlines the comprehensive documentation structure for the Foundry AI system.

## Documentation Structure

```
/home/tachyon/Foundry_entry/Foundry/foundry/docs/
├── README.md                    # Main documentation index
├── getting-started/
│   ├── installation.md          # Installation guide
│   ├── quick-start.md          # Quick start tutorial
│   ├── system-requirements.md   # Hardware/software requirements
│   └── troubleshooting.md      # Common issues and solutions
├── architecture/
│   ├── overview.md             # System architecture overview
│   ├── pipeline-design.md      # Pipeline architecture
│   ├── component-diagram.md    # Visual component relationships
│   └── data-flow.md           # Data flow through the system
├── deployment/
│   ├── local-deployment.md     # Local setup guide
│   ├── docker-deployment.md    # Docker deployment
│   ├── remote-models.md        # Remote model configuration
│   ├── ssh-tunneling.md        # SSH port forwarding setup
│   └── production-guide.md     # Production deployment best practices
├── api-reference/
│   ├── backend-api.md          # Backend API documentation
│   ├── vllm-client.md         # vLLM client reference
│   ├── pipeline-api.md        # Pipeline control API
│   └── websocket-api.md       # WebSocket events reference
├── pipeline-guides/
│   ├── manual-extraction.md    # Manual extractor guide
│   ├── data-enrichment.md     # Data enrichment process
│   ├── teacher-generation.md   # Teacher pair generation
│   ├── distillation.md        # Model distillation
│   ├── student-self-study.md  # Student self-study phase
│   └── evaluation.md          # Model evaluation
├── development/
│   ├── contributing.md        # Contribution guidelines
│   ├── coding-standards.md    # Code style and standards
│   ├── testing-guide.md       # Testing procedures
│   └── debugging.md          # Debugging techniques
└── templates/
    ├── script-template.py     # Standard script template
    ├── test-template.py       # Test file template
    └── api-doc-template.md    # API documentation template
```

## Documentation Templates

### Script Documentation Template

```python
"""
Script Name: {script_name}
Version: {version}
Author: {author}
Date: {date}

Description:
    {detailed_description}

Dependencies:
    - {dependency1}: {version}
    - {dependency2}: {version}

Configuration:
    Environment Variables:
        - {VAR_NAME}: {description}
    
    Config Files:
        - {config_file}: {purpose}

Usage:
    Basic usage:
        python {script_name}.py [options]
    
    Options:
        --input: Input file path
        --output: Output directory
        --model: Model name or path

Examples:
    Example 1: Basic usage
        python {script_name}.py --input data.json --output results/
    
    Example 2: With custom model
        python {script_name}.py --input data.json --model custom-model

Error Handling:
    Common errors and solutions:
    - {Error1}: {Solution1}
    - {Error2}: {Solution2}

Performance Notes:
    - GPU Memory: {requirements}
    - Processing Time: {estimates}
    - Optimization Tips: {tips}
"""
```

### API Documentation Template

```markdown
# {API Endpoint Name}

## Overview
{Brief description of the endpoint's purpose}

## Endpoint
```
{METHOD} /api/{path}
```

## Request

### Headers
| Header | Type | Required | Description |
|--------|------|----------|-------------|
| Authorization | string | Yes | Bearer token |
| Content-Type | string | Yes | application/json |

### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| {param1} | {type} | {Yes/No} | {description} |
| {param2} | {type} | {Yes/No} | {description} |

### Request Body
```json
{
    "field1": "value1",
    "field2": "value2"
}
```

## Response

### Success Response
**Code:** 200 OK

**Content:**
```json
{
    "status": "success",
    "data": {
        "result": "value"
    }
}
```

### Error Responses
**Code:** 400 Bad Request

**Content:**
```json
{
    "error": "Invalid input",
    "details": "Specific error message"
}
```

## Examples

### cURL
```bash
curl -X POST \
  http://localhost:5000/api/{path} \
  -H 'Authorization: Bearer {token}' \
  -H 'Content-Type: application/json' \
  -d '{"field1": "value1"}'
```

### Python
```python
import requests

response = requests.post(
    'http://localhost:5000/api/{path}',
    headers={'Authorization': 'Bearer {token}'},
    json={'field1': 'value1'}
)
```

## Notes
- {Additional notes or considerations}
```

### Pipeline Documentation Template

```markdown
# {Pipeline Step Name}

## Overview
{Comprehensive description of the pipeline step}

## Purpose
{Why this step exists and what problem it solves}

## Input Requirements
- **Data Format**: {JSON/CSV/etc}
- **Schema**: 
  ```json
  {
      "field1": "string",
      "field2": "array"
  }
  ```
- **Preprocessing**: {Any required preprocessing}

## Configuration
```yaml
pipeline_step:
  name: {step_name}
  script: {script_file}
  parameters:
    param1: value1
    param2: value2
  requirements:
    gpu: true/false
    memory: 16GB
    models: [model1, model2]
```

## Process Flow
1. {Step 1 description}
2. {Step 2 description}
3. {Step 3 description}

## Output Format
- **Data Format**: {JSON/CSV/etc}
- **Schema**:
  ```json
  {
      "results": [],
      "metadata": {}
  }
  ```
- **Files Generated**: {List of output files}

## Performance Considerations
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)
- **GPU Memory**: {requirements}
- **Optimization Tips**: {tips}

## Error Handling
| Error Code | Description | Solution |
|------------|-------------|----------|
| E001 | Invalid input format | Check input schema |
| E002 | Model not found | Verify model path |

## Dependencies
- **Python Libraries**: {list}
- **System Requirements**: {list}
- **Model Requirements**: {list}

## Integration Points
- **Previous Step**: {step_name}
- **Next Step**: {step_name}
- **Side Effects**: {any side effects}

## Monitoring and Logging
- **Log Location**: `/path/to/logs`
- **Key Metrics**: {metrics to monitor}
- **Alerts**: {conditions for alerts}

## Examples
```bash
# Example 1: Basic usage
python {script_name}.py --input input.json --output output/

# Example 2: With custom parameters
python {script_name}.py --input input.json --batch-size 64 --gpu 0
```

## Troubleshooting
### Common Issues
1. **Issue**: {description}
   **Solution**: {solution}

2. **Issue**: {description}
   **Solution**: {solution}

## References
- [Related Documentation]({link})
- [API Reference]({link})
```