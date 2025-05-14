# Manual Extractor Script

## Overview

The `manual_extractor.py` script is a versatile tool for extracting content from various sources:

1. **Web URLs**: Extract content from web pages, including metadata and links
2. **Local Files**: Process files from a local directory
3. **Docker Folders**: Access files inside Docker containers

The script generates structured data in JSON format and creates PDF versions of web pages or copies existing PDFs.

## Features

- **Multiple Input Sources**: Process content from URLs, local folders, or Docker containers
- **PDF Generation**: Create PDF versions of web pages using headless Chrome
- **Link Extraction**: Optionally extract and process links from web pages
- **Local File Processing**: Support for various file types (PDF, TXT, MD, JSON, CSV, HTML, XML)
- **Progress Reporting**: Real-time progress updates during processing
- **Structured Output**: Organize extracted content into structured JSON format

## Usage

### Command Line Arguments

```
python manual_extractor.py [options]
```

Options:
- `--url URL`: URL to extract content from
- `--source-folder PATH`: Local folder to extract content from
- `--docker-folder PATH`: Docker container folder to extract content from
- `--output-dir DIR`: Directory to save output files (default: "Output")
- `--extract-links`: Extract links from web pages

### Examples

1. Extract from a URL:
   ```
   python manual_extractor.py --url https://example.com --output-dir Output --extract-links
   ```

2. Process local files:
   ```
   python manual_extractor.py --source-folder C:\Documents\Data --output-dir Output
   ```

3. Access Docker container files:
   ```
   python manual_extractor.py --docker-folder /data --output-dir Output
   ```

## Output

The script generates the following outputs:

1. **JSON Data**: A structured JSON file containing all extracted content
2. **PDF Files**: PDF versions of web pages or copies of existing PDFs
3. **Log File**: Detailed log of the extraction process

## Integration with UI

This script is integrated with the Model Distillation UI and can be executed from the Scripts page. The UI provides a user-friendly interface for configuring and running the script with different input sources.

## Requirements

- Python 3.6+
- Chrome browser (for PDF generation)
- Required Python packages:
  - requests
  - beautifulsoup4
  - selenium
  - webdriver-manager
  - tqdm

## Notes

- When processing local files, the script will recursively search for supported file types
- For Docker folder processing, the script needs access to the Docker container filesystem
- PDF generation requires a working Chrome installation
