# pyright: reportMissingImports=false
import subprocess
import sys
import os

def install_package(package_name, prefer_binary=False, version=None):
    """Install a Python package using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if prefer_binary:
        cmd.append("--prefer-binary")
    
    if version:
        cmd.append(f"{package_name}=={version}")
    else:
        cmd.append(package_name)
    
    package_display = f"{package_name}" + (f" (version {version})" if version else "") + (" (binary only if available)" if prefer_binary else "")
    print(f"Installing {package_display}...")
    print(f"$ {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print(f"Successfully installed {package_display}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_display}: {e}")
        return False

def main():
    # List of required packages
    packages = [
        "pymupdf",  # for fitz module
        "pytesseract",
        "transformers",
        "sentence-transformers",
        "nltk",
        "scikit-learn",
        "numpy",
        "tqdm",
        "pillow",
    ]
    
    # Install packages
    for package in packages:
        install_package(package, prefer_binary=True)
    
    # Try different approaches for installing spaCy
    print("\n=== Attempting to install spaCy ===")
    
    # First attempt: Try with prefer-binary flag
    if install_package("spacy", prefer_binary=True):
        print("Successfully installed spaCy with prefer-binary flag.")
    else:
        # Second attempt: Try with a specific older version
        print("Trying to install an older version of spaCy...")
        if install_package("spacy", prefer_binary=True, version="3.5.3"):
            print("Successfully installed spaCy 3.5.3.")
        else:
            # Third attempt: Try with an even older version
            print("Trying to install an even older version of spaCy...")
            if install_package("spacy", prefer_binary=True, version="3.4.4"):
                print("Successfully installed spaCy 3.4.4.")
            else:
                # Fourth attempt: Try with stanza (Stanford NLP) as an alternative
                print("Trying to install stanza (Stanford NLP) as an alternative to spaCy...")
                if install_package("stanza", prefer_binary=True):
                    print("Successfully installed stanza as an alternative to spaCy.")
                    print("Note: You'll need to modify the code to use stanza instead of spaCy.")
                    print("See https://stanfordnlp.github.io/stanza/ for documentation.")
                else:
                    print("\n=== Alternative Installation Options ===")
                    print("1. Try using conda instead of pip:")
                    print("   conda install -c conda-forge spacy")
                    print("2. Use a Docker container with spaCy pre-installed")
                    print("3. Use a virtual environment with an older Python version (e.g., 3.9 or 3.10)")
                    print("\nThe script will continue, but NLP functionality will be limited.")
    
    # Check if spaCy is installed before trying to install language models
    spacy_installed = False
    stanza_installed = False
    
    try:
        import spacy
        spacy_installed = True
        print("\nspaCy is installed. Checking version...")
        print(f"spaCy version: {spacy.__version__}")
    except ImportError:
        print("\nspaCy is not installed.")
        # Check if stanza was installed as an alternative
        try:
            import stanza
            stanza_installed = True
            print("Stanza is installed as an alternative to spaCy. Checking version...")
            print(f"Stanza version: {stanza.__version__}")
        except ImportError:
            print("Neither spaCy nor Stanza is installed. NLP functionality will be limited.")
    
    # Install spaCy language models if spaCy is installed
    if spacy_installed:
        print("\nInstalling spaCy language models...")
        try:
            # Try to install the transformer-based model first (most accurate)
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
            print("Successfully installed en_core_web_trf")
        except subprocess.CalledProcessError:
            print("Failed to install en_core_web_trf, trying en_core_web_lg...")
            try:
                # Try to install the large model as fallback
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
                print("Successfully installed en_core_web_lg")
            except subprocess.CalledProcessError:
                print("Failed to install en_core_web_lg, trying en_core_web_sm...")
                try:
                    # Install the small model as last resort
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    print("Successfully installed en_core_web_sm")
                except subprocess.CalledProcessError:
                    print("Failed to install any spaCy language model")
    
    # Download stanza models if stanza is installed
    if stanza_installed:
        print("\nDownloading Stanza English model...")
        try:
            # Use Python code to download the model
            import stanza
            stanza.download('en')
            print("Successfully downloaded Stanza English model")
        except Exception as e:
            print(f"Failed to download Stanza English model: {e}")
            print("You can manually download it later using: stanza.download('en')")
    
    # Download NLTK resources
    print("\nDownloading NLTK resources...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Successfully downloaded NLTK resources")
    except Exception as e:
        print(f"Failed to download NLTK resources: {e}")
    
    # Note about neuralcoref
    print("\nNOTE: neuralcoref is optional and not directly installable via pip.")
    print("The script is designed to work without it, but if you want to use it,")
    print("you would need to install it from source following instructions at:")
    print("https://github.com/huggingface/neuralcoref")
    
    # Note about Tesseract OCR
    print("\nNOTE: pytesseract requires Tesseract OCR to be installed on your system.")
    print("If you want to use OCR functionality, please install Tesseract OCR from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki (Windows)")
    print("or use your package manager on Linux/macOS.")
    
    print("\nDependency installation completed.")

if __name__ == "__main__":
    main()
