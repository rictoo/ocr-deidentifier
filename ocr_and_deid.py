import sys
import os
import re
import copy
import argparse
import logging
from tqdm import tqdm
import warnings

os.environ["FONTCONFIG_FILE"] = os.path.join(os.getcwd(), "config", "fontconfig.cfg")
os.environ["FONTCONFIG_PATH"] = os.path.join(os.getcwd(), "config")

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

warnings.filterwarnings("ignore", message="No fonts configured in FontConfig", category=UserWarning, module="weasyprint")

logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Skipping annotation",
    category=UserWarning,
    module="spacy_huggingface_pipelines.token_classification"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="doctr.models.utils.pytorch"
)

warnings.filterwarnings(
    "ignore",
    message="Skipping annotation",
    category=UserWarning,
    module="spacy_huggingface_pipelines.token_classification"
)

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from deskew import determine_skew

from transformers import AutoTokenizer, AutoModelForTokenClassification
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from huggingface_hub import snapshot_download

from skimage import color, filters
from skimage.io import imsave, imread
from skimage.transform import rotate

from doctr.models import (
    ocr_predictor,
    detection_predictor,
    builder,
    crnn_vgg16_bn,
    crnn_mobilenet_v3_large,
    vitstr_small,
    from_hub,
    parseq
)
from doctr.io import DocumentFile

# Set paths
parser = argparse.ArgumentParser(description="OCR and de-identification pipeline")
parser.add_argument("-i", "--input_dir", required=True, help="Path to the input directory containing images with text to be de-identified")
parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory for de-identified text files")
parser.add_argument(
    "-t", "--tesseract",
    required=False,
    default="tesseract",
    help="Path to the Tesseract OCR executable"
)
parser.add_argument(
    "-f", "--overwrite",
    action="store_true",
    help="Force overwrite of output files if they already exist"
)
args = parser.parse_args()
input_directory = args.input_dir
output_directory = args.output_dir
overwrite_flag = args.overwrite

pytesseract.pytesseract.tesseract_cmd = args.tesseract

# Load text detection model (for bounding boxes)
logging.info("Loading the text detection model.")
model = detection_predictor(arch='db_resnet50', pretrained=True)  # Run your code that triggers the warning

# De-identification
conf_file_stanford = 'config/deid_config_stanford.yaml'
conf_file_roberta = 'config/deid_config_roberta.yaml'

transformers_model_stanford = "StanfordAIMI/stanford-deidentifier-base"
transformers_model_roberta = "obi/deid_roberta_i2b2"

logging.info("Loading the de-identification models.")
nlp_engine_stanford = NlpEngineProvider(conf_file=conf_file_stanford).create_engine()
nlp_engine_roberta = NlpEngineProvider(conf_file=conf_file_roberta).create_engine()

analyzer_stanford = AnalyzerEngine(
    nlp_engine = nlp_engine_stanford,
    supported_languages=["en"]
)

analyzer_roberta = AnalyzerEngine(
    nlp_engine = nlp_engine_roberta,
    supported_languages=["en"]
)

postcode_pattern = Pattern(name="postcode_pattern", regex='([A-Z][A-HJ-Y]?\d[A-Z\d]? ?\d[A-Z]{2}|GIR ?0A{2})', score = 1.0)
postcode_recognizer = PatternRecognizer(supported_entity="UK_POSTCODE", patterns = [postcode_pattern])

engine = AnonymizerEngine()

if not os.path.exists(output_directory):
    os.makedirs(path)

# Function to extract report identifier and page number from filename
def extract_info(filename):
    match = re.match(r"([0-9a-zA-Z]+).(\d{3}).[a-zA-Z]+", filename)
    if match:
        identifier, page = match.groups()
        return identifier, int(page)
    else:
        raise IOError('Input images must match naming convention: alphanumeric.pagenumber.extension')
        return None, None
        
def remove_uk_postcodes(text):
    pattern = r'([A-Z][A-HJ-Y]?\d[A-Z\d]? ?\d[A-Z]{2}|GIR ?0A{2})'
    return re.sub(pattern, '[redacted]', text, flags=re.MULTILINE)

def merge_overlapping_entities(entities):
    """
    Merges adjacent/overlapping de-identification start/end indices
    """
    entities.sort(key=lambda x: x.start)
    
    merged = []
    current = None
    
    for entity in entities:
        if not current or entity.start > current.end:
            if current:
                merged.append(current)
            current = entity
        else:
            current.end = max(current.end, entity.end)
            current.score = max(current.score, entity.score)
    if current:
        merged.append(current)
    return merged

def fuzzy_replace(text, base_identifier):
    ocr_errors = {
        '0': '[O0]',
        '1': '[1lI]',
        '2': '[2Z]',
        '3': '[3E]',
        '4': '[4A]',
        '5': '[5S]',
        '6': '[6G]',
        '7': '[7T]',
        '8': '[8B]',
        '9': '[9g]'
    }
    
    fuzzy_pattern = ''.join(ocr_errors.get(char, char) for char in str(base_identifier))
    
    return re.sub(fuzzy_pattern, "[redacted]", text, flags=re.IGNORECASE)
    
def find_word_indices_for_char_positions(words, x, y):
    """
    Takes: a string of words, and absolute indices x and y
    Returns: the word idx and position within word of start (x) and end index (y)
    """
    accumulated_length = 0
    start_word, end_word = None, None
    start_index, end_index = None, None
    for i, word in enumerate(words):
        word_length = len(word)
        # Check for the start index
        if start_word is None and accumulated_length + word_length >= x:
            start_word = i
            start_index = x - accumulated_length
        # Check for the end index
        if end_word is None and accumulated_length + word_length >= y:
            end_word = i
            end_index = y - accumulated_length
        accumulated_length += word_length + 1  # Add 1 for the space
        if start_word is not None and end_word is not None:
            break
    return (start_word, start_index), (end_word, end_index)
    
def calculate_avg_char_size(words):
    total_width = 0
    total_height = 0
    total_chars = 0

    for word in words:
        word_length = len(word['value'])
        if word_length > 0:
            xmin, ymin = word['geometry'][0]
            xmax, ymax = word['geometry'][1]
            total_width += (xmax - xmin)
            total_height += (ymax - ymin)
            total_chars += word_length

    avg_char_width = total_width / total_chars if total_chars > 0 else 0
    avg_char_height = total_height / total_chars if total_chars > 0 else 0

    return avg_char_width, avg_char_height

def merge_overlapping_redaction_segments(segments):
    """
    Merges adjacent or overlapping redaction segments
    """
    if not segments:
        return []
    segments.sort()
    merged_segments = [segments[0]]

    for current in segments[1:]:
        last = merged_segments[-1]
        if current[0] <= last[2] and current[1] <= last[3] + 1:
            merged_segments[-1] = (last[0], min(last[1], current[1]), current[2], max(last[3], current[3]))
        else:
            merged_segments.append(current)
    return merged_segments

def redact_word_and_adjust_geometry(global_index, word, merged_segments, avg_char_width):
    """
    Takes a word idx and word within text block, the redaction dictionary, and avg char width
    Returns redacted text along with repositioning of the geometry along the x axis due to change in word length
    """
    original_word_text = word['value']
    original_word_geometry = word['geometry']
    redacted_word = original_word_text
    total_shift = 0

    redaction_positions = []
    for start_word, start_idx, end_word, end_idx in merged_segments:
        if start_word <= global_index <= end_word: # If input word is contained within redaction
            start_pos = start_idx if global_index == start_word else 0
            end_pos = end_idx if global_index == end_word else len(original_word_text)
            # To handle if > 1 redaction affects different parts of the same word
            redaction_positions.append((start_pos, end_pos))
    #redaction_positions = merge_positions(redaction_positions)
    last_end = 0
    redacted_text = ""
    for start, end in redaction_positions:
        redacted_text += original_word_text[last_end:start] + '[redacted]'
        last_end = end
    redacted_text += original_word_text[last_end:]
    original_length = len(original_word_text)
    redacted_length = len(redacted_text)
    geometry_shift = (redacted_length - original_length) * avg_char_width

    return {
        'text': redacted_text,
        'shift': geometry_shift,
        'geometry': (
            original_word_geometry[0],
            (original_word_geometry[1][0] + geometry_shift, original_word_geometry[1][1])
        )
    }

# def merge_positions(positions):
#     if not positions:
#         return []
#
#     positions.sort()
#     merged = [positions[0]]
#     for current in positions[1:]:
#         last = merged[-1]
#         if current[0] <= last[1]:
#             merged[-1] = (last[0], max(last[1], current[1]))
#         else:
#             merged.append(current)
#     return merged

def redact_words_and_adjust_geometries(words, line_indices, redaction_segments):
    """
    Takes words, corresponding line indices, and redaction elements (with redaction word idx and idx within word)
    Returns redacted words with corresponding new line indices.
    """
    new_words = []
    new_line_indices = []
    # Merge adjacent segments
    global_merged_segments = merge_overlapping_redaction_segments(redaction_segments)
    for line_index in set(line_indices):
        line_words = [word for word, index in zip(words, line_indices) if index == line_index]
        # Calculate average character size to determine how to update geometry
        # depending on # of characters redaction increased / decreased length
        avg_char_width, _ = calculate_avg_char_size(line_words)
        updated_line_words = []
        for i, word in enumerate(line_words):
            global_index = line_indices.index(line_index) + i # Tracks word index within the block
            redacted_word_info = redact_word_and_adjust_geometry(global_index, word, global_merged_segments, avg_char_width)
            if redacted_word_info['text'] != word['value']:
                updated_word = {
                    'value': redacted_word_info['text'],
                    'confidence': word['confidence'],
                    'geometry': redacted_word_info['geometry']
                }
                updated_line_words.append(updated_word)
            else:
                updated_line_words.append(word)

        new_words.extend(updated_line_words)
        new_line_indices.extend([line_index] * len(updated_line_words))
    return new_words, new_line_indices

def redact_and_update_geometries(result, redaction_info):
    """
    Updates the result dictionary with redacted elements and updated geometries (to maintain text flow).
    """
    new_result = copy.deepcopy(result)  # Create a deep copy of the result dictionary

    #for page_idx, page in enumerate(new_result['pages']):
    page = new_result['pages'][0]
    for block_idx, block in enumerate(page['blocks']):
        all_words = []
        line_indices = []
        for line_idx, line in enumerate(block['lines']):
            all_words.extend(line['words'])
            line_indices.extend([line_idx] * len(line['words']))  # Apply line index for each word on line

        # Retrieve the redaction indices for this specific block
        redaction_indices = redaction_info[block_idx]
        # Adjust the geometry for redaction
        updated_words, new_line_indices = redact_words_and_adjust_geometries(all_words, line_indices, redaction_indices)

        # Redistribute these words (with updated geometries) back into their respective lines
        current_word_idx = 0
        for line_idx, line in enumerate(block['lines']):
            line['words'] = [word for word, index in zip(updated_words, new_line_indices) if index == line_idx]

    return new_result
    
def calculate_char_dimensions(ocr_results):
    """
    Calculates median character dimensions across entire page
    """
    line_widths = []
    line_heights = []
    total_chars = 0

    page_width, page_height = ocr_results['dimensions']

    for block in ocr_results['blocks']:
        for line in block['lines']:


            for word in line['words']:
                word_length = len(word['value'])
                if word_length == 0: continue

                (_, word_ymin), (_, word_ymax) = word['geometry'] 
                line_height = (word_ymax - word_ymin) * page_height
                line_heights.append(line_height)

                (xmin, _), (xmax, _) = word['geometry']
                word_width = (xmax - xmin) * page_width

                line_widths.append(word_width/word_length)
                total_chars += word_length

    if total_chars == 0: return 10, 10
    median_char_height = np.median(line_heights) if line_heights else 10
    median_char_width = np.median(line_widths) if line_widths else 10

    return median_char_width, median_char_height
    
def convert_ocr_to_text_with_normalized_spacing(results):
    """
    Converts the OCR result into a text format while preserving the layout,
    with normalized spacing between words.
    """

    # Extracting page dimensions
    page_width, page_height = results['dimensions']

    #char_width, char_height = (30*0.72), (34*0.72)
    char_width, char_height = calculate_char_dimensions(results)
    char_width *= 0.7
    char_height *= 0.7
    page_array = np.full((int(page_height // char_height), int(page_width // char_width)), ' ', dtype='<U1')

    # Process each line
    for block in results['blocks']:
        for line in block['lines']:
            (line_xmin, ymin), _ = line['geometry']
            y_start = int(ymin * page_height // char_height)

            x_cursor = int(line_xmin * page_width // char_width)  # Starting x position for each line
            line_start = True
            for word in line['words']:
                (word_xmin, _), _ = word['geometry']
                word_start = int(word_xmin * page_width // char_width)

                # Maybe just try always adding a space?
                if not line_start:
                    x_cursor += 1
                else:
                    line_start = False

                # Place the word on the page array
                for char in word['value']:
                    if 0 <= x_cursor < page_array.shape[1] and 0 <= y_start < page_array.shape[0]:
                        page_array[y_start, x_cursor] = char
                        x_cursor += 1

    # Convert the array to a string with newlines, removing excess blank lines
    text_lines = [''.join(row).rstrip() for row in page_array]
    text_lines = [line for line in text_lines if line.strip()]  # Remove empty lines

    return '\n'.join(text_lines)

def perform_tesseract_ocr_on_bbox(bbox, image):
    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin * w)
    xmax = int(xmax * w)
    ymin = int(ymin * h)
    ymax = int(ymax * h)
    cropped_image = image[ymin:ymax, xmin:xmax]
    cropped_image = cropped_image.astype(np.uint8) * 255
    border_size = [int(0.2 * dim) for dim in cropped_image.shape[:2]]
    bordered_image = np.pad(cropped_image, ((border_size[0], border_size[0]), (border_size[1], border_size[1]), (0, 0)), mode='constant', constant_values=255)
    return (pytesseract.image_to_string(Image.fromarray(bordered_image), lang='eng', config='--psm 7')).replace('\n', '').strip()

all_files = {}
for f in os.listdir(input_directory):
    all_files[extract_info(f)[0]] = [f] if extract_info(f)[0] not in all_files else all_files[extract_info(f)[0]] + [f]

# For tqdm purposes, to update progress bar on each page of each report


# Initialize a counter for processed reports
processed_report_count = 0
processed_identifiers = set()

complete_text = []  # This list will hold the OCR'd pages (as Document objects)
redaction_info = []  # This list will hold our redaction indices

#for file in all_files:
for report_name, page_file in tqdm([(key, value) for key, values in all_files.items() for value in values]):
    if report_name not in processed_identifiers:
        if not overwrite_flag and os.path.exists(os.path.join(output_directory, f"{report_name}.txt")):
            continue
        processed_identifiers.add(report_name)
        complete_text = []  # Zero'ing out for the next document
        redaction_info = []  # Zero'ing out for the next document

    # Process each page of the report
    file_path = os.path.join(input_directory, page_file)

    # Read and process the image
    try:
        image = Image.open(file_path).convert("L")
    except Exception as e:
        print(f"An error occured with {full_identifier}: {e}")
        continue
    gray_image = np.array(image)

    # Deskew image
    angle = determine_skew(gray_image)
    gray_image = rotate(gray_image, angle, resize=True)

    # Convert pixel information to 0-255 range
    gray_image = (gray_image * 255).astype("uint8")
    #threshold_value = filters.threshold_otsu(gray_image)

    # Apply local thresholding for conversion to binary image
    threshold_value = filters.threshold_local(gray_image, 21, offset=25)
    binary_image = gray_image > threshold_value

    # Conversion to three-channel image, since this is needed for input into the text detector
    three_channel_image = np.stack([binary_image] * 3, axis=-1)

    # Running text detection + Tesseract OCR
    detector_output = model([three_channel_image.astype(np.uint8) * 255])

    boxes = [np.array(detector_output[0]['words'])]
    text_preds = [[(None, 1.0) for bbox in detector_output[0]['words']]]
    page_shapes = [three_channel_image.shape[:2]]
    doc_builder = builder.DocumentBuilder(resolve_blocks=False) # Don't heuristically make blocks.

    # Assemble Document object
    doc = doc_builder(
        [three_channel_image.astype(np.uint8) * 255],
        boxes,
        text_preds,
        page_shapes
    )

    # Iterate over detected text bounding boxes and perform OCR on line elements.
    line_boxes = [] # Bounding boxes of line elements
    line_text_preds = [] # OCR'd text in corresponding bounding boxes
    for _, page in enumerate(doc.export()['pages']): # Can probably just take the first Page idx.
        page_line_boxes = []
        page_line_text_preds = []
        for block_idx, block in enumerate(page['blocks']):
            for line_idx, line in enumerate(block['lines']):
                recognized_text = perform_tesseract_ocr_on_bbox([coordinate for bbox in line['geometry'] for coordinate in bbox], three_channel_image)
                line_box = [coordinate for bbox in line['geometry'] for coordinate in bbox]
                page_line_text_preds.append((recognized_text, 1.0))
                page_line_boxes.append(line_box)
        line_text_preds.append(page_line_text_preds)
        line_boxes.append(page_line_boxes)

    # Re-assemble Document object with OCR'd text
    result = doc_builder(
        [three_channel_image.astype(np.uint8) * 255],
        np.array(line_boxes),
        line_text_preds,
        page_shapes
    )

    complete_text.append(result)

    # De-identification, performed on each page-block element combination
    for page_idx, page_text in enumerate(complete_text):
        page = page_text.pages[0]
        redaction_info.append({})
        # Loop through each block in the page
        for block_idx, block in enumerate(page.blocks):
            block_contents = [] # Stores all words within this block
            for line_idx, line in enumerate(block.lines):
                for word in line.words:
                    block_contents.append(word.value)
            redaction_info[page_idx][block_idx] = []

            # Run both de-identification models on text block
            results_english_stanford = analyzer_stanford.analyze(text=' '.join(block_contents), language="en")
            results_english_roberta = analyzer_roberta.analyze(text=' '.join(block_contents), language="en")

            # De-identify UK postcodes
            results_postcode = postcode_recognizer.analyze(text=' '.join(block_contents), entities=["UK_POSTCODE"])

            # Merge all de-identifications
            results_english_merged = results_english_roberta + results_english_stanford + results_postcode

            # Drop these two PII categories, as they seem to result in lots of false positives
            results_english = [x for x in results_english_merged if x.entity_type not in ['US_DRIVER_LICENSE', 'IN_PAN']]

            # Merge overlapping/adjacent de-identification indices
            results_english = merge_overlapping_entities(results_english)

            # Prevent erroneous de-identification of ICD codes and marker status (ER/PR, etc.)
            pattern = r"\b[TMP](-?\d{5})(-?[A-Za-z])?\b"
            results_english = [x for x in results_english if not re.search(pattern, ' '.join(block_contents)[x.start:x.end])]
            pattern = r"^\d/8$"
            results_english = [x for x in results_english if not re.search(pattern, ' '.join(block_contents)[x.start:x.end])]

            # Replace text falling within de-identification indices with hashes (#)
            anonymized_text = engine.anonymize(' '.join(block_contents), results_english, operators =
                                              {"DEFAULT":OperatorConfig("custom",
                                                                       {"lambda": lambda x: "#"*len(x)})})

            # Process the anonymized results to get the start and end indices
            for item in anonymized_text.items:
                (start_word_idx, start_idx_in_word), (end_word_idx, end_idx_in_word) = (
                    find_word_indices_for_char_positions(block_contents, item.start, item.end)
                )

                redaction_info[page_idx][block_idx].append(
                    (start_word_idx, start_idx_in_word, end_word_idx, end_idx_in_word)
                )

    anonymized_text = ''

    # For every page (Document object) and corresponding redaction dictionary (with keys corresponding to block elements)
    for page_text, page_redactions in zip(complete_text, redaction_info):
        redacted_text = redact_and_update_geometries(page_text.export(), page_redactions)
        redacted_plaintext = convert_ocr_to_text_with_normalized_spacing(redacted_text['pages'][0])
        # Further check to remove postcodes and identifiers:
        redacted_plaintext = remove_uk_postcodes(redacted_plaintext)
        anonymized_text += ('\n' if len(anonymized_text) != 0 else '') + redacted_plaintext

    # Save the de-identified text with report name as the filename
    output_file_path = os.path.join(output_directory, f"{report_name}.txt")
    with open(output_file_path, 'w') as f:
        f.write(anonymized_text)

logging.info(f"Processed {len(processed_identifiers)}/{len(all_files)} reports.")
