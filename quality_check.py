import yaml
import re
import os
import logging # Recommended for logging errors/info

# --- Configuration Loading ---
# Assumes the config file is named 'config.yaml' and is in the same directory as the script
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = {}

try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f: # Added encoding
        config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from: {CONFIG_FILE_PATH}")
except FileNotFoundError:
    logging.error(f"CRITICAL: Configuration file not found at {CONFIG_FILE_PATH}. Using empty config.")
    # Depending on the application, you might exit or use hardcoded defaults
    # exit()
except yaml.YAMLError as e:
    logging.error(f"CRITICAL: Error parsing configuration file {CONFIG_FILE_PATH}: {e}. Using empty config.")
    # Depending on the application, you might exit or use hardcoded defaults
    # exit()
except Exception as e:
    logging.error(f"CRITICAL: An unexpected error occurred loading config {CONFIG_FILE_PATH}: {e}. Using empty config.")
    # exit()

# Helper function to safely get config values with defaults
def get_config_value(keys, default=None):
    """Safely navigates the config dictionary."""
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logging.warning(f"Config path '{'.'.join(keys)}' not found or invalid. Using default: {default}")
        return default

# --- Placeholder Functions (These need proper implementation based on your data) ---

def get_baseline_data(page_segments):
    """
    Placeholder: Calculates baseline paragraph properties (font size, indent, etc.)
    Requires analyzing segments to find the most common paragraph style.
    Needs segment data including position (bbox) and font info.
    """
    # Example structure - needs real logic
    logging.debug("Calculating baseline data (placeholder)")
    # Simplified: find most common font size, assume baseline indent is min x0?
    # This is complex and domain-specific.
    return {
        "font_size": 10.0, # Placeholder
        "indent": 72.0, # Placeholder (e.g., 1 inch in points)
        "avg_char_width": 6.0 # Placeholder
    }

def get_page_data(page_number):
    """ Placeholder: Returns page dimensions (width, height) """
    logging.debug(f"Getting page data for page {page_number} (placeholder)")
    # Needs access to document structure info
    return {"width": 612, "height": 792} # Example: US Letter

# --- Modified Heuristic Functions (Using YAML Config and Assumed Metadata) ---

def is_likely_page_number_or_noise_yaml(segment_data, config):
    """
    Heuristic using YAML config and segment metadata.
    Assumes segment_data contains keys like: 'text', 'font_size', 'font_flags', 'font_name', 'bbox'
    """
    text = segment_data.get('text', '').strip()
    if not text:
        return False

    # 1. Purely numeric or Roman numeral
    if re.fullmatch(r'\d+', text): return True # Keep simple check
    if re.fullmatch(r'\s*[ivxlcdm]+\s*', text.lower()): return True # Keep simple check

    # 2. Number surrounded by specific symbols (using config regex if available)
    # Note: YAML has no direct equivalent for this specific check in the provided config
    # Keep the original logic or adapt if needed
    if re.fullmatch(r'[-*•\s]*\d+[-*•\s]*', text): return True

    # 3. Specific header/footer patterns (using config regex if available)
    # The YAML config doesn't have an exact match for '^[A-Z\W]\s*•\s*.+\s*\d+$'
    # Keep original or decide if another config regex applies
    if re.search(r'^[A-Z\W]\s*•\s*.+\s*\d+$', text): return True

    # Chapter/Section/Appendix with page number at end
    # Using a generic noise regex from config, if defined, otherwise keep specific one
    noise_regex = get_config_value(['heuristics', 'noise', 'page_number_header_regex']) # Example hypothetical config path
    if noise_regex:
        if re.search(noise_regex, text): return True
    elif re.search(r'^(Chapter|Section|Appendix)\s+[\w\d\.]+\s+\d+$', text): return True

    # 4. Keyword checks
    # Make keywords configurable if desired, e.g., get_config_value(['heuristics', 'noise', 'keywords'], [])
    if "openstax.org" in text.lower(): return True
    if any(kw in text for kw in ["Additional Topics", "Selected Hints", "Selected Solutions"]): return True

    # 5. Copyright/Permission notices (configurable length limit)
    max_len_copyright = get_config_value(['heuristics', 'noise', 'max_copyright_length'], 300)
    if "permission is granted" in text.lower() or "copyright" in text.lower():
        if len(text) < max_len_copyright: return True

    # 6. Font checks (Example using math/superscript flags from config)
    font_flags = segment_data.get('font_flags', 0)
    superscript_bit = get_config_value(['heuristics', 'math_formula', 'superscript_flag_bit'])
    if superscript_bit and (font_flags & superscript_bit):
        # If it's just a number and superscript, likely noise/footnote marker
        if re.fullmatch(r'\d+', text):
             logging.debug(f"Flagging as noise (superscript number): {text}")
             return True

    # Add checks for specific math font names if text is short/symbolic?
    math_fonts = get_config_value(['heuristics', 'math_formula', 'math_font_names'], [])
    font_name = segment_data.get('font_name', '').lower()
    if any(mf in font_name for mf in math_fonts):
         # Add more logic: maybe if it contains specific symbols and is short?
         pass # Example placeholder

    return False

def is_likely_heading_yaml(segment_data, baseline_data, page_data, config):
    """
    Heuristic using YAML config and segment metadata.
    Assumes segment_data contains: 'text', 'font_size', 'font_flags', 'bbox'
    Assumes baseline_data contains: 'font_size'
    Assumes page_data contains: 'width'
    """
    text = segment_data.get('text', '').strip()
    font_size = segment_data.get('font_size')
    font_flags = segment_data.get('font_flags', 0)
    bbox = segment_data.get('bbox') # Expected: [x0, y0, x1, y1]

    if not text or font_size is None:
        return None # Cannot determine without text or font size

    baseline_font_size = baseline_data.get('font_size', font_size) # Default to self if no baseline
    if baseline_font_size == 0: baseline_font_size = font_size # Avoid division by zero

    # --- Primary Checks using Config ---
    size_ratio = font_size / baseline_font_size
    bold_flag_bit = get_config_value(['heuristics', 'heading', 'bold_flag_bit'])
    is_bold = bold_flag_bit and (font_flags & bold_flag_bit) > 0

    # Check defined heading level tiers
    level_tiers = get_config_value(['heuristics', 'heading', 'level_tiers'], [])
    for tier in sorted(level_tiers, key=lambda x: x.get('level', 99)): # Check stricter levels first
        min_ratio = tier.get('min_ratio', 0)
        required_flags_bits = tier.get('flags_must_include', []) # Assuming these are bit values
        # Check if size ratio meets minimum
        if size_ratio >= min_ratio:
             # Check if all required flags are present
             flags_met = True
             if required_flags_bits: # Only check if flags are specified
                for flag_bit in required_flags_bits:
                    # Special handling if flag_bit list is empty? Maybe means just ratio? Check config definition.
                    # Assuming list contains integer bit values like [16] for bold based on YAML.
                    # If YAML uses [6], this might mean (Bold=2? Italic=4?) - needs clarification
                    if not (font_flags & flag_bit):
                        flags_met = False
                        break
             else:
                 # If flags_must_include is empty or missing, does it mean *any* flag or *no* flags?
                 # Assume for now empty means no flag requirement beyond ratio.
                 # Or maybe it implies bold is required for level 4? Clarify intent.
                 # Let's assume level 4 requires bold if flags are empty in YAML.
                 if tier.get('level') == 4 and not is_bold:
                    flags_met = False


             if flags_met:
                 logging.debug(f"Matched Heading Level {tier.get('level')} (Ratio: {size_ratio:.2f}, Bold: {is_bold}): {text}")
                 return tier.get('level') # Return the heading level

    # Check for numbering patterns using config regex
    numbering_regex = get_config_value(['heuristics', 'heading', 'numbering_regex'])
    if numbering_regex and re.match(numbering_regex, text):
        # If it matches numbering, consider it a heading, maybe default level? Needs refinement.
        # Let's assign level 5 if it matches numbering but not tiers? Or requires bold?
        if is_bold or size_ratio > 1.0: # Add some minimal criteria
             logging.debug(f"Matched Heading (Numbering Regex, Ratio: {size_ratio:.2f}, Bold: {is_bold}): {text}")
             return 5 # Example: Assign a default level for numbered headings

    # --- Fallback/Additional Checks (Can be made configurable too) ---
    max_len = get_config_value(['heuristics', 'heading', 'max_length'], 100) # Example default
    min_len = get_config_value(['heuristics', 'heading', 'min_length'], 5)   # Example default

    if min_len < len(text) < max_len:
        is_title_case = text.istitle()
        is_upper_case = text.isupper()
        ends_punctuation = text.endswith('.') or text.endswith('!') or text.endswith('?')

        # Title or Upper case, not ending like a sentence
        if (is_title_case or is_upper_case) and not ends_punctuation:
            # Maybe require bold or slightly larger size as qualifier?
            if is_bold or size_ratio > 1.0:
                 logging.debug(f"Matched Heading (Case/Length, Ratio: {size_ratio:.2f}, Bold: {is_bold}): {text}")
                 return 6 # Example: Assign another default level

    # Check for specific keywords
    heading_keywords = get_config_value(['heuristics', 'heading', 'keywords'], ["Contents", "Preface", "Acknowledgements", "References", "Index", "Appendix"])
    if text in heading_keywords:
        logging.debug(f"Matched Heading (Keyword): {text}")
        return 7 # Example: Assign level based on keyword match

    # Centered check (simplified example)
    centered_tolerance_ratio = get_config_value(['heuristics', 'heading', 'centered_tolerance'], 0.1)
    if bbox and page_data:
        page_width = page_data['width']
        block_width = bbox[2] - bbox[0]
        block_center = bbox[0] + block_width / 2
        page_center = page_width / 2
        deviation = abs(block_center - page_center)
        allowed_deviation = page_width * centered_tolerance_ratio
        if deviation < allowed_deviation:
             # Is being centered enough? Maybe combine with size/bold?
             if is_bold or size_ratio > 1.05:
                 logging.debug(f"Matched Heading (Centered, Ratio: {size_ratio:.2f}, Bold: {is_bold}): {text}")
                 return 8 # Example: Assign level for centered items

    # Max lines check would require knowing if the segment_data represents multiple lines

    return None # Not identified as a heading by any criteria

def is_likely_list_item_yaml(segment_data, baseline_data, config):
    """
    Heuristic using YAML config and segment metadata.
    Assumes segment_data contains: 'text', 'bbox'
    Assumes baseline_data contains: 'indent', 'avg_char_width'
    """
    text = segment_data.get('text', '').strip()
    bbox = segment_data.get('bbox')

    if not text or not bbox:
        return None # Need text and position

    # --- Check 1: Indentation (Primary indicator if available) ---
    baseline_indent = baseline_data.get('indent')
    avg_char_width = baseline_data.get('avg_char_width')
    indent_factor = get_config_value(['heuristics', 'list_item', 'indentation_factor'], 1.5)

    if baseline_indent is not None and avg_char_width is not None and avg_char_width > 0:
        segment_indent = bbox[0] # Left coordinate
        required_indent = baseline_indent + (indent_factor * avg_char_width)
        if segment_indent >= required_indent:
             logging.debug(f"Potential list item (Indentation {segment_indent:.1f} >= {required_indent:.1f}): {text}")
             # Indentation suggests it *could* be a list item. Now check for markers.
        else:
            # If not significantly indented relative to baseline, less likely a list item unless marker is very strong?
            # For now, let's rely on marker regex if not indented.
            pass


    # --- Check 2: Markers (Using config regex) ---
    # Max length check
    max_len_list = get_config_value(['heuristics', 'list_item', 'max_length'], 300)
    if len(text) >= max_len_list:
        return None # Too long

    # Bullet points
    bullet_regex = get_config_value(['heuristics', 'list_item', 'marker_regex', 'bullet'])
    if bullet_regex and re.match(bullet_regex, text):
        logging.debug(f"Matched List Item (Bullet Regex): {text}")
        return "bullet"

    # Numbered/Lettered/Roman
    numbered_regex = get_config_value(['heuristics', 'list_item', 'marker_regex', 'numbered'])
    if numbered_regex and re.match(numbered_regex, text):
        logging.debug(f"Matched List Item (Numbered Regex): {text}")
        # Could try to determine specific type (numeric, alpha, roman) from match group 1 if needed
        match = re.match(numbered_regex, text)
        marker = match.group(1) # Get the captured marker part
        if marker.isdigit(): return "numbered_arabic"
        if re.fullmatch(r'[a-zA-Z]', marker): return "numbered_latin"
        if re.fullmatch(r'[ivxlcdmIVXLCDM]+', marker): return "numbered_roman"
        return "numbered" # Fallback

    # If only indentation matched, but no standard marker regex, what is it?
    # Could be a hanging indent paragraph or just an indented block.
    # Let's return None if no marker is found, even if indented, to be conservative.
    # if segment_indent >= required_indent: # Revisit this condition
    #    logging.debug(f"Indented block, but no standard list marker found: {text}")
    #    return "indented_block_no_marker" # Or None?

    return None # Not identified as a list item

# --- Example Main Processing Logic ---

def process_document(document_data):
    """
    Example of how the checks might be called.
    Assumes document_data is structured, e.g., by page, then by segment,
    with each segment having text and metadata.
    """
    all_flagged_segments = {}

    for page_num, segments in document_data.items():
        logging.info(f"Processing Page {page_num}")
        page_flagged = []

        # --- These need proper implementation ---
        baseline_data = get_baseline_data(segments)
        page_data = get_page_data(page_num)
        # ---

        for segment in segments:
            # --- Run Checks ---
            # Check 1: Noise/Page Number
            # Pass relevant segment metadata and config
            if is_likely_page_number_or_noise_yaml(segment, config):
                segment['flag_reason'] = 'noise_or_page_num'
                page_flagged.append(segment)
                logging.info(f"Flagged [Noise]: {segment.get('text', '')[:50]}...")
                continue # Skip other checks if noise

            # Check 2: Heading
            heading_level = is_likely_heading_yaml(segment, baseline_data, page_data, config)
            if heading_level is not None:
                segment['likely_type'] = f'H{heading_level}'
                logging.info(f"Identified [H{heading_level}]: {segment.get('text', '')[:50]}...")
                # Potentially flag headings based on other rules? For now, just identify.
                # If you want to flag *bad* headings (e.g., inconsistent numbering), add logic here.
                continue # Assume headings aren't flagged unless specific checks fail

            # Check 3: List Item
            list_type = is_likely_list_item_yaml(segment, baseline_data, config)
            if list_type is not None:
                segment['likely_type'] = f'list_{list_type}'
                logging.info(f"Identified [List-{list_type}]: {segment.get('text', '')[:50]}...")
                # Flag inconsistent lists? (e.g., mixing markers, bad indentation)
                continue

            # Check 4: Code Block (Example - Needs Implementation)
            # code_block_heuristic(segment, config) -> needs font name/flag checks

            # Check 5: Math Formula (Example - Needs Implementation)
            # math_formula_heuristic(segment, config) -> needs font name/flag checks

            # Check 6: Caption (Example - Needs Implementation)
            # caption_heuristic(segment, page_layout, config) -> needs layout context

            # --- Default / Other Flagging Logic ---
            # If a segment wasn't identified as a specific type, apply other quality checks
            # (e.g., very short lines, lines with odd characters, etc.)
            # These checks could also be made configurable via YAML
            if 'likely_type' not in segment:
                 if len(segment.get('text', '').strip()) < 5: # Example: Flag very short segments
                      segment['flag_reason'] = 'too_short'
                      page_flagged.append(segment)
                      logging.info(f"Flagged [Too Short]: {segment.get('text', '')[:50]}...")


        if page_flagged:
            all_flagged_segments[page_num] = page_flagged

    return all_flagged_segments

# --- Main Execution Area ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Load your structured data (JSONL or other format)
    #    This data MUST include text AND metadata (font size, flags, bbox etc.)
    #    Example: input_data = load_data_with_metadata('C:\\apex_project\\structured_jsonl_by_category\\math.jsonl')
    #    The structure of input_data needs to be known (e.g., dict mapping page_num to list of segment dicts)
    logging.info("Loading data (placeholder)...")
    # dummy_document_data = {
    #     1: [
    #         {'text': 'Chapter 1: Introduction', 'font_size': 18.0, 'font_flags': 16, 'bbox': [72, 700, 400, 718], 'font_name': 'Arial-BoldMT'},
    #         {'text': 'This is the first paragraph.', 'font_size': 10.0, 'font_flags': 0, 'bbox': [72, 680, 540, 690], 'font_name': 'TimesNewRomanPSMT'},
    #         {'text': '1.', 'font_size': 10.0, 'font_flags': 0, 'bbox': [72, 665, 85, 675], 'font_name': 'TimesNewRomanPSMT'},
    #         {'text': 'First list item', 'font_size': 10.0, 'font_flags': 0, 'bbox': [90, 665, 200, 675], 'font_name': 'TimesNewRomanPSMT'},
    #         {'text': '123', 'font_size': 8.0, 'font_flags': 0, 'bbox': [300, 50, 320, 58], 'font_name': 'TimesNewRomanPSMT'},
    #     ]
    # }
    # flagged_results = process_document(dummy_document_data)

    # 2. Run the processing
    #    flagged_results = process_document(input_data)
    logging.warning("Using dummy data for demonstration. Replace with actual data loading and processing.")


    # 3. Save the flagged segments
    #    save_flagged_data(flagged_results, 'C:\\apex_project\\flagged_structured_data\\...')
    logging.info("Processing complete (dummy run).")