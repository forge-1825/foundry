import requests
from bs4 import BeautifulSoup
import json
import os
import sys
import time
import argparse
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging
import glob
import re
from pathlib import Path
import base64

# ---------------------------------------------------------------------
# Logging configuration: log to both console and file.
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("manual_extraction.log", mode='w')
    ]
)

# Set up Chrome options for headless PDF generation
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--window-size=1920,1080')
chrome_options.add_experimental_option('prefs', {
    'download.default_directory': os.path.abspath(os.path.join(os.path.dirname(__file__), "Output", "pdfs")),
    'download.prompt_for_download': False,
    'download.directory_upgrade': True,
    'safebrowsing.enabled': True,
    'plugins.always_open_pdf_externally': True
})

def sanitize_filename(name):
    """Sanitize the filename to remove problematic characters."""
    return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name).strip().replace(" ", "_")

def extract_additional_metadata(soup):
    """Attempt to extract extra metadata like last modified, authors, or version info."""
    metadata = {}
    meta_last = soup.find('meta', attrs={'name': 'last-modified'})
    if meta_last and meta_last.get('content'):
        metadata['last_modified'] = meta_last.get('content')
    meta_author = soup.find('meta', attrs={'name': 'author'})
    if meta_author and meta_author.get('content'):
        metadata['author'] = meta_author.get('content')
    version_tag = soup.find(class_='version')
    if version_tag:
        metadata['version'] = version_tag.get_text(strip=True)
    return metadata

def structure_content(content, headers):
    """
    Split the content into labeled sections based on headers.
    This naive approach splits content by line and checks for header matches.
    """
    sections = {}
    if not headers:
        sections['full_content'] = content
        return sections

    lines = content.splitlines()
    current_section = "Introduction"  # default section
    sections[current_section] = []
    for line in lines:
        stripped = line.strip()
        if stripped in headers:
            current_section = stripped
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(stripped)
    for key in sections:
        sections[key] = "\n".join(sections[key]).strip()
    return sections

def fetch_and_parse(url):
    """Fetch a URL and extract its content, title, metadata, and structure."""
    logging.info("Fetching URL: %s", url)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        logging.info("Successfully fetched URL with status code: %d", response.status_code)
        logging.debug("Response content length: %d", len(response.text))
    except Exception as e:
        logging.error("Error fetching %s: %s", url, e)
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.get_text(strip=True) if soup.title else "No Title"
    pub_date = None
    for meta_name in ['pubdate', 'publish-date', 'date']:
        meta_tag = soup.find('meta', attrs={'name': meta_name})
        if meta_tag and meta_tag.get('content'):
            pub_date = meta_tag.get('content')
            break
    if not pub_date:
        date_tag = soup.find(class_='date')
        if date_tag:
            pub_date = date_tag.get_text(strip=True)
    section_headers = [tag.get_text(strip=True) for tag in soup.find_all(['h1', 'h2', 'h3'])]
    content_div = soup.find('div', id='main-content')
    if not content_div:
        content_div = soup.find('div', class_='content')
    content = content_div.get_text(separator="\n", strip=True) if content_div else ""
    structured_content = structure_content(content, section_headers)
    additional_metadata = extract_additional_metadata(soup)
    data = {
        "url": url,
        "title": title,
        "publication_date": pub_date,
        "section_headers": section_headers,
        "structured_content": structured_content,
        "raw_content": content
    }
    data.update(additional_metadata)
    logging.info("Parsed data for URL: %s", url)
    return data

def extract_links(url, extract_links=True):
    """
    Extract all links from the given URL.
    Returns a list of dictionaries with keys: "url" and "text" (anchor text).
    """
    if not extract_links:
        logging.info("Link extraction disabled, skipping for URL: %s", url)
        return []

    logging.info("Extracting links from URL: %s", url)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        logging.info("Successfully fetched URL with status code: %d", response.status_code)
        logging.debug("Response content length: %d", len(response.text))
    except Exception as e:
        logging.error("Error fetching %s: %s", url, e)
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    container = soup.find('div', id='main-content')
    if not container:
        container = soup
    links = []
    for a_tag in container.find_all('a', href=True):
        link = a_tag['href']
        text = a_tag.get_text(strip=True)
        if not link.startswith("http"):
            link = requests.compat.urljoin(url, link)
        links.append({"url": link, "text": text})
    unique_links = list({item['url']: item for item in links}.values())
    logging.info("Extracted %d unique links from %s", len(unique_links), url)
    return unique_links

def save_pdf(url, pdf_dir, driver=None):
    """Convert a URL to PDF using Chrome's built-in PDF printer."""
    logging.info("Attempting PDF conversion for URL: %s", url)

    if driver is None:
        driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        # Set up PDF parameters
        filename = sanitize_filename(url.split("/")[-1] or "page") + ".pdf"
        pdf_path = os.path.join(pdf_dir, filename)

        # Use Chrome's built-in PDF printer
        print_options = {
            'landscape': False,
            'displayHeaderFooter': False,
            'printBackground': True,
            'preferCSSPageSize': True,
        }

        # Generate PDF using Chrome's Page.printToPDF
        pdf_data = driver.execute_cdp_cmd('Page.printToPDF', print_options)

        # Save PDF file
        import base64
        pdf_bytes = base64.b64decode(pdf_data['data'])
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)

        logging.info("PDF saved successfully: %s", pdf_path)
        return pdf_path
    except Exception as e:
        logging.error("Error converting %s to PDF: %s", url, e)
        return None

def process_local_file(file_path, output_dir):
    """Process a local file and extract its content."""
    logging.info("Processing local file: %s", file_path)

    try:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        # Create data structure for the file
        data = {
            "file_path": file_path,
            "title": file_name,
            "file_type": file_ext[1:] if file_ext.startswith('.') else file_ext,
            "category": "local_file"
        }

        # For PDF files, copy to output directory
        if file_ext == '.pdf':
            pdf_dir = os.path.join(output_dir, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)

            # Copy the PDF file to the output directory
            output_pdf_path = os.path.join(pdf_dir, file_name)
            with open(file_path, 'rb') as src_file, open(output_pdf_path, 'wb') as dst_file:
                dst_file.write(src_file.read())

            data["pdf_path"] = os.path.relpath(output_pdf_path, output_dir)
            logging.info("PDF copied to: %s", output_pdf_path)

            # Try to extract text content from PDF (simplified)
            data["raw_content"] = f"PDF file: {file_name}"
            data["structured_content"] = {"full_content": f"PDF file: {file_name}"}

        # For text files, read the content
        elif file_ext in ['.txt', '.md', '.json', '.csv', '.html', '.xml']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                data["raw_content"] = content

                # Simple structure for text content
                lines = content.splitlines()
                headers = []

                # Try to identify headers (lines that are shorter and possibly all caps or have special chars)
                for line in lines:
                    line = line.strip()
                    if line and (line.isupper() or re.match(r'^[A-Z][\w\s]+:?$', line)):
                        headers.append(line)

                data["structured_content"] = structure_content(content, headers)
                data["section_headers"] = headers

                logging.info("Successfully extracted content from text file: %s", file_path)
            except Exception as e:
                logging.error("Error reading text file %s: %s", file_path, e)
                data["raw_content"] = f"Error reading file: {str(e)}"
                data["structured_content"] = {"error": f"Error reading file: {str(e)}"}

        # For other file types, just record basic metadata
        else:
            data["raw_content"] = f"Unsupported file type: {file_ext}"
            data["structured_content"] = {"full_content": f"Unsupported file type: {file_ext}"}

        return data

    except Exception as e:
        logging.error("Error processing local file %s: %s", file_path, e)
        return None

def process_local_directory(directory_path, output_dir, extract_links=False):
    """Process all files in a local directory."""
    logging.info("Processing local directory: %s", directory_path)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    # Find all files in the directory
    file_paths = []
    for ext in ['*.pdf', '*.txt', '*.md', '*.json', '*.csv', '*.html', '*.xml']:
        pattern = os.path.join(directory_path, '**', ext)
        file_paths.extend(glob.glob(pattern, recursive=True))

    logging.info("Found %d files in directory: %s", len(file_paths), directory_path)

    # Process each file
    dataset = []
    progress_bar = tqdm(total=len(file_paths),
                        desc="Processing local files",
                        unit="file",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for file_path in file_paths:
        progress_bar.set_postfix_str(f"File: {os.path.basename(file_path)}")
        file_data = process_local_file(file_path, output_dir)

        if file_data:
            dataset.append(file_data)
            logging.info("Added data for file: %s", file_path)
        else:
            logging.warning("Failed to process file: %s", file_path)

        progress_bar.update(1)

    progress_bar.close()
    return dataset

def report_progress(current, total, message="Processing"):
    """Report progress to both console and log file."""
    progress_percent = (current / total) * 100 if total > 0 else 0
    logging.info("[PROGRESS] %s: %.2f%% (%d/%d)", message, progress_percent, current, total)
    # In a real implementation, this would send progress to the UI via an API call
    print(f"\r[PROGRESS] {message}: {progress_percent:.2f}% ({current}/{total})", end="")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract content from web pages or local files")
    parser.add_argument("--url", help="URL to extract content from")
    parser.add_argument("--source-folder", help="Local folder to extract content from")
    parser.add_argument("--docker-folder", help="Docker container folder to extract content from")
    parser.add_argument("--output-dir", default="Output", help="Directory to save output files")
    parser.add_argument("--extract-links", action="store_true", help="Extract links from web pages")
    args = parser.parse_args()

    # Validate arguments
    if not args.url and not args.source_folder and not args.docker_folder:
        logging.error("Error: Either URL, source folder, or Docker folder must be provided")
        parser.print_help()
        return

    # Initialize Chrome driver with automatic ChromeDriver installation
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    logging.info("Chrome driver initialized for PDF generation")

    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Output directory created at: %s", output_dir)

    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    logging.info("PDF output directory created at: %s", pdf_dir)

    dataset = []

    try:
        # Process URL if provided
        if args.url:
            logging.info("Starting extraction process for URL: %s", args.url)

            # Test PDF generation with the provided URL
            pdf_path = save_pdf(args.url, pdf_dir, driver)
            if pdf_path:
                logging.info("Test PDF generated successfully at: %s", pdf_path)
            else:
                logging.warning("PDF generation test failed. Continuing with content extraction...")

            # Extract and process the main URL
            page_data = fetch_and_parse(args.url)
            if page_data:
                page_data["category"] = "main_url"
                if pdf_path:
                    page_data["pdf_path"] = os.path.relpath(pdf_path, output_dir)
                dataset.append(page_data)
                logging.info("Added data for URL: %s", args.url)
            else:
                logging.error("Failed to process URL: %s", args.url)

            # Extract and process links if enabled
            if args.extract_links:
                links = extract_links(args.url, extract_links=True)
                logging.info("Found %d links to process", len(links))

                progress_bar = tqdm(total=len(links),
                                    desc="Processing links",
                                    unit="link",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

                for i, link_data in enumerate(links):
                    link_url = link_data["url"]
                    progress_bar.set_postfix_str(f"Link: {link_url}")
                    report_progress(i+1, len(links), "Processing links")

                    page_data = fetch_and_parse(link_url)
                    if page_data:
                        page_data["category"] = "linked_page"
                        dataset.append(page_data)
                        logging.info("Added data for linked URL: %s", link_url)

                        # Generate PDF for the linked page
                        pdf_path = save_pdf(link_url, pdf_dir, driver)
                        if pdf_path:
                            page_data["pdf_path"] = os.path.relpath(pdf_path, output_dir)
                            logging.info("PDF generated successfully at: %s", pdf_path)
                        else:
                            logging.warning("Failed to generate PDF for %s", link_url)
                    else:
                        logging.warning("Failed to process linked URL: %s", link_url)

                    time.sleep(0.5)  # Pause briefly to avoid overloading the server
                    progress_bar.update(1)

                progress_bar.close()

        # Process local folder if provided
        elif args.source_folder:
            logging.info("Processing local folder: %s", args.source_folder)
            local_dataset = process_local_directory(args.source_folder, output_dir, args.extract_links)
            dataset.extend(local_dataset)

        # Process Docker folder if provided
        elif args.docker_folder:
            logging.info("Processing Docker folder: %s", args.docker_folder)
            # In a real implementation, this would need to access files inside the Docker container
            # For now, we'll just log a message
            logging.warning("Docker folder processing is not fully implemented")
            # Placeholder for Docker folder processing
            dataset.append({
                "title": "Docker Folder",
                "category": "docker_folder",
                "path": args.docker_folder,
                "raw_content": f"Docker folder: {args.docker_folder}",
                "structured_content": {"full_content": f"Docker folder: {args.docker_folder}"}
            })

        # Save the extracted data to JSON
        logging.info("Saving extracted JSON data...")
        json_path = os.path.join(output_dir, "extracted_data.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            logging.info("Data successfully saved to %s", json_path)
        except Exception as e:
            logging.error("Error saving JSON data: %s", e)

    finally:
        # Ensure Chrome driver is closed
        try:
            driver.quit()
        except:
            pass
        "https://attack.mitre.org/versions/v16/",
        "https://attack.mitre.org/versions/v16/matrices/",
        "https://attack.mitre.org/versions/v16/tactics/",
        "https://attack.mitre.org/versions/v16/techniques/",
        "https://attack.mitre.org/versions/v16/datasources",
        "https://attack.mitre.org/versions/v16/mitigations/",
        "https://attack.mitre.org/versions/v16/assets",
        "https://attack.mitre.org/versions/v16/groups",
        "https://attack.mitre.org/versions/v16/software",
        "https://attack.mitre.org/versions/v16/campaigns",
        "https://attack.mitre.org/versions/v16/resources/",
        "https://attack.mitre.org/resources/versions/",
        "https://medium.com/mitre-attack/",
        "https://github.com/mitre/cti/releases/tag/ATT%26CK-v16.1",
        "https://na.eventscloud.com/attackcon6",
        "https://attack.mitre.org/versions/v16/datasources/DS0026/",
        "https://attack.mitre.org/versions/v16/datasources/DS0015/",
        "https://attack.mitre.org/versions/v16/datasources/DS0041/",
        "https://attack.mitre.org/versions/v16/datasources/DS0039/",
        "https://attack.mitre.org/versions/v16/datasources/DS0037/",
        "https://attack.mitre.org/versions/v16/datasources/DS0025/",
        "https://attack.mitre.org/versions/v16/datasources/DS0010/",
        "https://attack.mitre.org/versions/v16/datasources/DS0017/",
        "https://attack.mitre.org/versions/v16/datasources/DS0032/",
        "https://attack.mitre.org/versions/v16/datasources/DS0038/",
        "https://attack.mitre.org/versions/v16/datasources/DS0016/",
        "https://attack.mitre.org/versions/v16/datasources/DS0027/",
        "https://attack.mitre.org/versions/v16/datasources/DS0022/",
        "https://attack.mitre.org/versions/v16/datasources/DS0018/",
        "https://attack.mitre.org/versions/v16/datasources/DS0001/",
        "https://attack.mitre.org/versions/v16/datasources/DS0036/",
        "https://attack.mitre.org/versions/v16/datasources/DS0007/",
        "https://attack.mitre.org/versions/v16/datasources/DS0030/",
        "https://attack.mitre.org/versions/v16/datasources/DS0035/",
        "https://attack.mitre.org/versions/v16/datasources/DS0008/",
        "https://attack.mitre.org/versions/v16/datasources/DS0028/",
        "https://attack.mitre.org/versions/v16/datasources/DS0004/",
        "https://attack.mitre.org/versions/v16/datasources/DS0011/",
        "https://attack.mitre.org/versions/v16/datasources/DS0023/",
        "https://attack.mitre.org/versions/v16/datasources/DS0033/",
        "https://attack.mitre.org/versions/v16/datasources/DS0029/",
        "https://attack.mitre.org/versions/v16/datasources/DS0040/",
        "https://attack.mitre.org/versions/v16/datasources/DS0021/",
        "https://attack.mitre.org/versions/v16/datasources/DS0014/",
        "https://attack.mitre.org/versions/v16/datasources/DS0009/",
        "https://attack.mitre.org/versions/v16/datasources/DS0003/",
        "https://attack.mitre.org/versions/v16/datasources/DS0012/",
        "https://attack.mitre.org/versions/v16/datasources/DS0013/",
        "https://attack.mitre.org/versions/v16/datasources/DS0019/",
        "https://attack.mitre.org/versions/v16/datasources/DS0020/",
        "https://attack.mitre.org/versions/v16/datasources/DS0002/",
        "https://attack.mitre.org/versions/v16/datasources/DS0042/",
        "https://attack.mitre.org/versions/v16/datasources/DS0034/",
        "https://attack.mitre.org/versions/v16/datasources/DS0006/",
        "https://attack.mitre.org/versions/v16/datasources/DS0024/",
        "https://attack.mitre.org/versions/v16/datasources/DS0005/",
        "https://attack.mitre.org/versions/v16/techniques/T1649",
        "https://attack.mitre.org/versions/v16/techniques/T1558",
        "https://attack.mitre.org/versions/v16/techniques/T1550/003",
        "https://attack.mitre.org/versions/v16/techniques/T1558/001",
        "https://attack.mitre.org/versions/v16/techniques/T1558/003",
        "https://attack.mitre.org/versions/v16/techniques/T1558/004",
        "https://attack.mitre.org/versions/v16/techniques/T1550",
        "https://attack.mitre.org/versions/v16/techniques/T1550/002",
        "https://attack.mitre.org/versions/v16/techniques/T1615",
        "https://attack.mitre.org/versions/v16/techniques/T1003",
        "https://attack.mitre.org/versions/v16/techniques/T1003/006",
        "https://attack.mitre.org/versions/v16/techniques/T1033",
        "https://attack.mitre.org/versions/v16/techniques/T1098",
        "https://attack.mitre.org/versions/v16/techniques/T1098/001",
        "https://attack.mitre.org/versions/v16/techniques/T1098/005",
        "https://attack.mitre.org/versions/v16/techniques/T1484",
        "https://attack.mitre.org/versions/v16/techniques/T1484/001",
        "https://attack.mitre.org/versions/v16/techniques/T1484/002",
        "https://attack.mitre.org/versions/v16/techniques/T1207",
        "https://attack.mitre.org/versions/v16/techniques/T1134",
        "https://attack.mitre.org/versions/v16/techniques/T1134/005",
        "https://attack.mitre.org/versions/v16/techniques/T1531",
        "https://attack.mitre.org/versions/v16/techniques/T1037",
        "https://attack.mitre.org/versions/v16/techniques/T1037/003",
        "https://attack.mitre.org/versions/v16/techniques/T1222",
        "https://attack.mitre.org/versions/v16/techniques/T1222/001",
        "https://attack.mitre.org/versions/v16/techniques/T1556",
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Unhandled exception: %s", e, exc_info=True)
