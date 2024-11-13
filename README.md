# OCR and De-identification Pipeline
This repository contains a Python script that performs Optical Character Recognition (OCR) and de-identification on images containing text, such as scanned medical reports. The script processes images, extracts text, and de-identifies sensitive information using NLP models while attempting to preserve the layout of the original document.

## Example Reports
Example inputs, in the form of medical reports, and outputs are provided in the ``example`` directory.

## Requirements
- Python 3.11 is recommended.
- Tesseract OCR binaries need to be installed separately. You can download Tesseract from [here](https://tesseract-ocr.github.io/tessdoc/Installation.html).
- Conda Environment: It is recommended to use a Conda environment.

## Installation
After cloning the repository, install the required dependencies using the following commands:

### Step 1: Create a New Conda Environment
To ensure dependency compatibility, it’s best to create a new environment:

```
conda create -n ocr_deid python=3.11
conda activate ocr_deid
```

### Step 2: Install Python libraries with Conda
```
conda install -y pandas "numpy<2" scikit-image pillow pytorch pytorch-cuda=12.4 torchvision transformers huggingface_hub fontconfig gtk3 -c conda-forge -c pytorch -c nvidia
```

### Step 3: Install additional libraries with pip
```
python -m pip install "numpy<2" python-doctr==0.8.0 deskew==1.4.3 pytesseract
python -m pip install "numpy<2" presidio-analyzer[transformers] presidio-anonymizer
```

### Step 4: Tesseract OCR
The pipeline was designed and tested with Tesseract version 5.4. While other versions may work, their compatibility is unverified. You need to install Tesseract OCR binaries separately. Download them [here](https://tesseract-ocr.github.io/tessdoc/Installation.html). Ensure that the Tesseract executable is accessible to the script.

## Usage
The script accepts the following arguments:

``-i``, ``--input_dir``: Required. Path to the input directory containing images with text to be de-identified.

``-o``, ``--output_dir``: Required. Path to the output directory for de-identified text files.

``-t``, ``--tesseract``: Optional. Path to the Tesseract OCR executable. Default is ``"tesseract"`` (if already in PATH).

``-f``, ``--overwrite``: Optional. Force overwrite of output files if they already exist.

### Input Image Format
The input images should follow the naming convention:
```
<alphanumeric_report_identifier>.<page_number>.<image_extension>
```

- ``<alphanumeric_report_identifier>``: A unique alphanumeric identifier for each report (e.g., report123, patientABC).
- ``<page_number>``: A three-digit page number (e.g., ``001``, ``002``).
- ``<image_extension>``: The image file extension (e.g., ``png``, ``jpg``).

**Examples:**

- ``report123.001.png``
- ``patientABC.002.jpg``

### Output Format
The output will be in the format:
```
<alphanumeric_report_identifier>.txt
```

All pages corresponding to that report identifier will be combined into a single text file.

**Example:**

Input files: ``report123.001.png``, ``report123.002.png``

Output file: ``report123.txt``

## Configuration Files
There are two configuration files in the config/ directory:
- ``deid_config_stanford.yaml``
- ``deid_config_roberta.yaml``
These files contain configurations for the de-identifiers used in the script.

## Running the Script
Example command to run the script:

```
python ocr_and_deid.py -i input_dir -o output_dir -t /path/to/tesseract.exe -f
```

- Replace ``input_dir`` with the path to your input directory containing images.
- Replace ``output_dir`` with the path to your desired output directory.
- Replace ``/path/to/tesseract.exe`` with the actual path to the Tesseract OCR executable.
- Include ``-f`` if you want to force overwrite of existing output files.
