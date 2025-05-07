# Full updated script for Phase 2 Post-Processing (v15 - Multi-Pass Refinement)
# Uses config.yaml (v7) for identification rules.
# Prioritizes Math ID, Stricter Code ID, Improved LaTeX script detection.
# Adds a second pass for refining ambiguous segment classifications.

import json
import os
import traceback
from tqdm import tqdm
import re
import uuid
import statistics
import yaml # Requires PyYAML: pip install pyyaml
import math # For sqrt check

# --- Load Configuration ---
CONFIG_FILE = 'config.yaml'
CONFIG = None
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    if CONFIG is None: CONFIG = {}
    print(f"Successfully loaded configuration from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"ERROR: Configuration file '{CONFIG_FILE}' not found. Using default fallbacks.")
    CONFIG = {}
except yaml.YAMLError as e:
    print(f"ERROR: Error parsing configuration file '{CONFIG_FILE}': {e}")
    CONFIG = {}
except Exception as e:
    print(f"ERROR: Unexpected error loading configuration: {e}")
    CONFIG = {}

# --- Get rules from config or use defaults ---
CODE_RULES = CONFIG.get('code_rules', {})
MONO_FONT_PATTERNS = [p.lower() for p in CODE_RULES.get('monospace_font_patterns', ["Mono", "Typewriter", "Courier", "Consola", "Menlo", "SFTT", "Fixed", "Code", "CMTT", "Lucida Console", "Hack"])]
CODE_KEYWORDS = set(CODE_RULES.get('code_keywords', ["def", "class", "import", "from", "for", "while", "if", "else", "elif", "try", "except", "return", "yield", "lambda", "print", "self", "true", "false", "none", "and", "or", "not", "in", "is", "with", "as", "assert", "break", "continue", "pass", "raise", "global", "nonlocal"]))
MONO_CHAR_THRESHOLD = CODE_RULES.get('monospace_char_threshold', 0.6) # Stricter default
KEYWORD_DENSITY_THRESHOLD = CODE_RULES.get('keyword_density_threshold', 0.15)
OPERATOR_DENSITY_THRESHOLD = CODE_RULES.get('operator_density_threshold', 0.05)
SHORT_BLOCK_MONO_THRESHOLD = CODE_RULES.get('short_block_monospace_threshold', 50)
CODE_SCORE_THRESHOLD = CODE_RULES.get('code_score_threshold', 3)
NEGATIVE_CODE_PATTERNS = CODE_RULES.get('negative_code_patterns', ["^<Figure size .*>$", "^Listing \\d+\\.\\d+:", "^Example \\d+\\.\\d+:", "^Algorithm \\d+:", "^Figure \\d+\\.\\d+:", "^Table \\d+\\.\\d+:"])

MATH_RULES = CONFIG.get('math_rules', {})
MATH_FONT_PATTERNS = [p.lower() for p in MATH_RULES.get('math_font_patterns', ["math", "cmsy", "cmmi", "symbol"])]
MATH_CHARS = set(MATH_RULES.get('math_chars', [ '≈' ,  '−' ,  '≠' ,  '≤' ,  '≥' ,  '∑' ,  '∫' ,  '∂' ,  '∇' ,  '√' ,  '∞' , '±', '×', '÷',  '∈' ,  '∉' ,  '⊂' ,  '⊃' ,  '⊆' ,  '⊇' ,  '∧' ,  '∨' , '¬',  '∀' ,  '∃' ,  '∴' ,  '∵' , 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω', '^', '_', '{', '}']))
MATH_FONT_THRESHOLD = MATH_RULES.get('math_font_threshold', 0.15)
MATH_SYMBOL_THRESHOLD = MATH_RULES.get('math_symbol_threshold', 0.08)
ITALIC_VAR_THRESHOLD = MATH_RULES.get('italic_variable_threshold', 0.10)
VARIED_SIZE_THRESHOLD = MATH_RULES.get('varied_size_threshold', 1.5)
NEGATIVE_PROSE_KEYWORDS = set(MATH_RULES.get('negative_prose_keywords', ['figure', 'table', 'example', 'note', 'source', 'caution', 'chapter', 'section', 'equation', 'theorem', 'lemma', 'proof', 'corollary', 'definition', 'proposition', 'remark', 'exercise']))
MAX_PROSE_WORD_RATIO = MATH_RULES.get('max_prose_word_ratio', 0.5) # Stricter default
MATH_SCORE_THRESHOLD = MATH_RULES.get('math_score_threshold', 3) # Back to 3, rely on stricter scoring

# New configuration for ambiguity thresholds (add these to your config.yaml)
AMBIGUITY_RULES = CONFIG.get('ambiguity_rules', {})
AMBIGUOUS_CODE_SCORE_MIN = AMBIGUITY_RULES.get('ambiguous_code_score_min', 1) # Minimum score to be considered potentially ambiguous code
AMBIGUOUS_MATH_SCORE_MIN = AMBIGUITY_RULES.get('ambiguous_math_score_min', 1) # Minimum score to be considered potentially ambiguous math
# Note: Segments with scores between MIN and THRESHOLD for either type can be ambiguous

GROUPING_RULES = CONFIG.get('grouping_rules', {})
CODE_MAX_V_DISTANCE = GROUPING_RULES.get('code_max_v_distance', 20)
GAP_CODE_MAX_V_DISTANCE = GROUPING_RULES.get('gap_code_max_v_distance', 30)
INDENT_WIDTH_ESTIMATE = GROUPING_RULES.get('indent_width_estimate', 18)
MIN_INDENT_DIFF = GROUPING_RULES.get('min_indent_diff', 5)
MAX_INDENT_WIDTH = GROUPING_RULES.get('max_indent_width', 50)

INPUT_JSONL_DIR = CONFIG.get('input_jsonl_dir', r'C:\apex_project\final_processed_jsonl_by_category_detailed')
OUTPUT_PROCESSED_DIR = CONFIG.get('output_processed_dir', r'C:\apex_project\structured_jsonl_by_category')
INPUT_FILENAME = CONFIG.get('input_filename', None)

# --- Constants ---
CODE_BLOCK_TYPE = 10
MATH_BLOCK_TYPE = 11 # Keeping this for potential intermediate math blocks
LATEX_MATH_TYPE = 12
GAP_BLOCK_TYPE = 5
TABLE_BLOCK_TYPE = 2
TEXT_BLOCK_TYPE = 0
IMAGE_BLOCK_TYPE = 1
# Define a new type for segments that remain ambiguous after refinement, if needed
# AMBIGUOUS_TEXT_TYPE = 13 # Example

# --- Pre-compile Regex for efficiency ---
NEGATIVE_CODE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_CODE_PATTERNS]
CODE_OPERATOR_REGEX = re.compile(r'[=+\-*/%<>&|^~:]')
CODE_BRACKET_REGEX = re.compile(r'[][(){}]')
WORD_REGEX = re.compile(r'\b\w+\b')
ALPHA_WORD_REGEX = re.compile(r'\b[a-z]{3,}\b', re.IGNORECASE)
LIST_ITEM_START_REGEX = re.compile(r"^\s*(?:(?:\d+\.)|[-*•+]|[a-z]\)|[ivx]+\.)\s+", re.IGNORECASE)
COMMON_WORDS = {'the', 'a', 'is', 'of', 'in', 'it', 'and', 'to', 'that', 'this', 'for', 'with', 'are', 'be', 'as', 'on', 'at', 'by', 'we', 'can', 'or', 'an', 'if', 'then', 'else', 'which', 'where', 'when', 'how', 'why'}

# =============================================
# --- ALL HELPER FUNCTIONS DEFINED HERE FIRST ---
# =============================================

def calculate_vertical_distance(bbox1, bbox2):
    """ Calculates vertical distance between bottom of bbox1 and top of bbox2 """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4: return float('inf')
    return bbox2[1] - bbox1[3]

# Modified classification functions to return detailed results
def is_likely_code_segment_detailed(segment):
    """Checks if a segment is likely code using refined rules and returns detailed results."""
    block_type = segment.get("block_type")
    block_text_content = segment.get("text", "")
    score = 0
    details = {}

    # --- Negative Pattern Check ---
    if block_text_content:
        for compiled_regex in NEGATIVE_CODE_REGEXES:
            if compiled_regex.match(block_text_content.strip()):
                details['negative_pattern_match'] = True
                # A strong negative match immediately reduces score or flags as not code
                # return {'is_code': False, 'is_ambiguous': False, 'score': -10, 'details': details} # Example of early exit

        if LIST_ITEM_START_REGEX.match(block_text_content.strip()):
             temp_mono_chars = 0
             temp_operator_hits = 0
             temp_keyword_hits = 0
             temp_total_chars = 0
             for line in segment.get("lines", []):
                 for span in line.get("spans", []):
                     span_text = span.get("text", "")
                     temp_total_chars += len(span_text)
                     font_name = span.get("font_name", "").lower()
                     if span.get("is_monospaced") or any(pattern in font_name for pattern in MONO_FONT_PATTERNS):
                         temp_mono_chars += len(span_text)
                 temp_operator_hits += len(CODE_OPERATOR_REGEX.findall(line.get("text","")))
                 temp_keyword_hits += sum(1 for word in WORD_REGEX.findall(line.get("text","").lower()) if word in CODE_KEYWORDS)
             mono_ratio = temp_mono_chars / temp_total_chars if temp_total_chars > 0 else 0
             details['list_item_mono_ratio'] = mono_ratio
             details['list_item_operator_hits'] = temp_operator_hits
             details['list_item_keyword_hits'] = temp_keyword_hits
             if mono_ratio < 0.7 and (temp_operator_hits < 2 and temp_keyword_hits < 2):
                 details['list_item_heuristic_fail'] = True
                 # return {'is_code': False, 'is_ambiguous': False, 'score': -10, 'details': details} # Example of early exit


    if block_type == GAP_BLOCK_TYPE:
        text = segment.get("text", "")
        if CODE_OPERATOR_REGEX.search(text) or text.strip().startswith(tuple(CODE_KEYWORDS) + ('#',)):
             score += 2 # Strong indicator for gap code
             details['gap_code_indicator'] = True

    elif block_type == TEXT_BLOCK_TYPE:
        lines = segment.get("lines", [])
        if not lines:
            return {'is_code': False, 'is_ambiguous': False, 'score': 0, 'details': details}

        monospace_chars = 0
        keyword_hits = 0
        operator_hits = 0
        total_chars = 0
        line_texts = []
        starts_with_comment_or_keyword = False
        ends_with_prose_punctuation = False
        has_colon_ending = False
        indentation_levels = set()
        min_line_x0 = float('inf')
        max_line_len = 0
        num_lines = len(lines)

        for i, line in enumerate(lines):
            line_text = line.get("text", "")
            line_text_stripped = line_text.strip()
            line_texts.append(line_text_stripped)
            max_line_len = max(max_line_len, len(line_text_stripped))

            INTERACTIVE_PROMPTS = ['>>>', '...', 'In [', 'Out [']  # Example interactive prompts
            if any(line_text_stripped.startswith(p) for p in INTERACTIVE_PROMPTS):
                 score += 5 # Very strong indicator
                 details['interactive_prompt'] = True

            first_span = line.get("spans", [{}])[0]
            x0 = first_span.get("origin", [0,0])[0] if first_span else line.get("bbox", [0,0,0,0])[0]
            if line_text_stripped:
                min_line_x0 = min(min_line_x0, x0)
                indentation_levels.add(round(x0))

            if i == 0 and (line_text_stripped.startswith('#') or line_text_stripped.startswith(tuple(CODE_KEYWORDS))): starts_with_comment_or_keyword = True
            if line_text_stripped.endswith(':'): has_colon_ending = True

            words = WORD_REGEX.findall(line_text_stripped.lower())
            for word in words:
                if word in CODE_KEYWORDS:
                    # Refined keyword check: avoid common words that are also keywords if in a prose-like context
                    if word in ['if', 'for', 'in', 'is', 'and', 'or', 'not'] and len(words) > 4 and len(CODE_OPERATOR_REGEX.findall(line_text)) < 1: continue
                    keyword_hits += 1

            operator_hits += len(CODE_OPERATOR_REGEX.findall(line_text))
            operator_hits += len(CODE_BRACKET_REGEX.findall(line_text))

            spans = line.get("spans", [])
            for span in spans:
                span_text = span.get("text", "")
                span_len = len(span_text)
                total_chars += span_len
                font_name = span.get("font_name", "").lower()
                if span.get("is_monospaced") or any(pattern in font_name for pattern in MONO_FONT_PATTERNS):
                    monospace_chars += span_len

            if line_text_stripped.endswith(('.', '?', '!')): ends_with_prose_punctuation = True

        # --- Refined Heuristics with Scoring ---
        if total_chars == 0:
             return {'is_code': False, 'is_ambiguous': False, 'score': 0, 'details': details}

        mono_ratio = monospace_chars / total_chars if total_chars > 0 else 0
        details['mono_ratio'] = mono_ratio
        if mono_ratio > MONO_CHAR_THRESHOLD: score += 3
        elif mono_ratio > 0.1: score += 1

        total_words = sum(len(WORD_REGEX.findall(lt)) for lt in line_texts)
        keyword_ratio = keyword_hits / total_words if total_words > 0 else 0
        details['keyword_ratio'] = keyword_ratio
        details['keyword_hits'] = keyword_hits
        if keyword_ratio > KEYWORD_DENSITY_THRESHOLD: score += 2
        elif keyword_hits > 1: score += 1

        operator_ratio = operator_hits / total_chars if total_chars > 0 else 0
        details['operator_ratio'] = operator_ratio
        details['operator_hits'] = operator_hits
        if operator_ratio > OPERATOR_DENSITY_THRESHOLD: score += 1
        if operator_hits > 2: score += 1

        if starts_with_comment_or_keyword:
             score += 1
             details['starts_with_comment_or_keyword'] = True
        if has_colon_ending and num_lines > 1:
             score += 1
             details['has_colon_ending'] = True
        if len(indentation_levels) > 1:
             score += 1 # Multiple indent levels suggest structure
             details['multiple_indent_levels'] = True

        # Penalties
        if max_line_len > 100 and mono_ratio < 0.2:
             score -= 1
             details['long_line_low_mono_penalty'] = True
        if ends_with_prose_punctuation and keyword_hits < 2 and mono_ratio < 0.1 and operator_hits < 2:
            score -= 4 # Increased penalty
            details['prose_punctuation_penalty'] = True

        # Short block adjustments
        if total_chars < SHORT_BLOCK_MONO_THRESHOLD:
            details['short_block'] = True
            if monospace_chars > 0: score += 1
            if operator_hits > 0: score += 1
            if keyword_hits > 0: score += 1 # Added keyword check for short blocks

    # --- Decision ---
    is_code = score >= CODE_SCORE_THRESHOLD
    # Define ambiguity: score is above a minimum but below the main threshold
    is_ambiguous = score >= AMBIGUOUS_CODE_SCORE_MIN and score < CODE_SCORE_THRESHOLD and not is_code # Ensure not already classified as code

    # Add checks for conflicting indicators that might make it ambiguous even if score is high
    if is_code and details.get('prose_punctuation_penalty', False) and score < (CODE_SCORE_THRESHOLD + 2): # Example: penalize if high score but also looks like prose
         is_code = False
         is_ambiguous = True
         details['reclassified_to_ambiguous_due_to_prose'] = True


    return {'is_code': is_code, 'is_ambiguous': is_ambiguous, 'score': score, 'details': details}


def segment_is_math_detailed(segment):
    """Checks if a segment (type 0) is likely math using refined rules and returns detailed results."""
    if segment.get("block_type") != TEXT_BLOCK_TYPE:
        return {'is_math': False, 'is_ambiguous': False, 'score': 0, 'details': {}}

    lines = segment.get("lines", [])
    if not lines:
        return {'is_math': False, 'is_ambiguous': False, 'score': 0, 'details': {}}

    # --- Feature Extraction ---
    total_spans = 0
    total_chars = 0
    math_font_chars = 0
    math_symbol_chars = 0
    italic_variable_chars = 0
    varied_size_spans = 0
    operator_chars = 0
    bracket_chars = 0
    numeric_chars = 0
    alpha_chars = 0
    max_span_len = 0
    distinct_math_symbols = set()
    text_chars = 0
    has_equals_sign = False
    has_fraction_bar_candidate = False
    common_word_count = 0
    total_word_count = 0
    num_lines = len(lines)
    ends_with_punctuation = False

    all_sizes = [s.get("font_size", 0) for ln in lines for s in ln.get("spans", []) if s.get("font_size", 0) > 0]
    base_font_size = statistics.median(all_sizes) if all_sizes else None

    block_text = segment.get("text", "")
    block_text_lower = block_text.lower()
    block_words = ALPHA_WORD_REGEX.findall(block_text_lower)
    total_word_count = len(block_words)
    common_word_count = sum(1 for word in block_words if word in COMMON_WORDS)


    details = {} # Store heuristic results

    # --- Negative Constraints ---
    negative_prose_match = any(neg_word in block_words for neg_word in NEGATIVE_PROSE_KEYWORDS)
    if negative_prose_match:
         details['negative_prose_keyword'] = True
         # Re-evaluate if it's still math despite negative keyword
         temp_math_symbols = sum(1 for char in block_text_lower if char in MATH_CHARS)
         temp_total_chars = len(block_text_lower) if block_text_lower else 0
         if temp_total_chars > 0 and (temp_math_symbols / temp_total_chars < 0.05) and len(CODE_OPERATOR_REGEX.findall(block_text)) < 2:
              details['strong_negative_prose_match'] = True
              # return {'is_math': False, 'is_ambiguous': False, 'score': -10, 'details': details} # Example of early exit


    common_text_fonts = ['times', 'arial', 'helvetica', 'calibri', 'verdana', 'roman', 'sans', 'serif']
    text_font_chars = 0
    contains_long_word = False
    for line in lines:
        for span in line.get("spans", []):
             font_name = span.get("font_name", "").lower()
             span_text = span.get("text","")
             if any(pattern in font_name for pattern in common_text_fonts) and not span.get("is_italic"):
                 text_font_chars += len(span_text)
             if len(span_text) > 15: contains_long_word = True

    if total_chars > 0 and text_font_chars / total_chars > 0.85 and not contains_long_word:
         details['dominant_text_font'] = True
         # return {'is_math': False, 'is_ambiguous': False, 'score': -10, 'details': details} # Example of early exit


    # --- Detailed Feature Calculation ---
    for line in lines:
        line_text_stripped = line.get("text", "").strip()
        if line_text_stripped.endswith(('.', ',', ';', '?', '!')):
            ends_with_punctuation = True

        for span in line.get("spans", []):
            total_spans += 1
            text = span.get("text", "")
            span_len = len(text)
            total_chars += span_len
            max_span_len = max(max_span_len, span_len)

            font_name = span.get("font_name", "").lower()
            font_size = span.get("font_size", 0)
            is_italic = span.get("is_italic", False)

            if base_font_size and font_size > 0 and abs(font_size - base_font_size) > VARIED_SIZE_THRESHOLD: varied_size_spans += 1

            is_math_font = any(pattern in font_name for pattern in MATH_FONT_PATTERNS)
            if is_math_font: math_font_chars += span_len

            current_math_chars = 0
            for char in text:
                if char in MATH_CHARS:
                    current_math_chars += 1
                    distinct_math_symbols.add(char)
                if char == '=': has_equals_sign = True
                if char in '+-*/': operator_chars += 1
                if char in '()[]{}': bracket_chars += 1
                if char.isdigit(): numeric_chars += 1
                if char.isalpha(): alpha_chars += 1
                if char ==  '−'  or char == '-':
                    if span_len == 1 or all(c == '-' for c in text): has_fraction_bar_candidate = True
            math_symbol_chars += current_math_chars

            if is_italic and not is_math_font and span_len == 1 and text.isalpha(): italic_variable_chars += 1

    # --- Calculate Ratios ---
    math_font_ratio = math_font_chars / total_chars if total_chars > 0 else 0
    math_symbol_ratio = math_symbol_chars / total_chars if total_chars > 0 else 0
    italic_variable_ratio = italic_variable_chars / total_chars if total_chars > 0 else 0
    varied_size_ratio = varied_size_spans / total_spans if total_spans > 0 else 0
    operator_ratio = operator_chars / total_chars if total_chars > 0 else 0
    prose_word_ratio = common_word_count / total_word_count if total_word_count > 0 else 0
    numeric_ratio = numeric_chars / total_chars if total_chars > 0 else 0
    alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0

    details['math_font_ratio'] = math_font_ratio
    details['math_symbol_ratio'] = math_symbol_ratio
    details['distinct_math_symbols_count'] = len(distinct_math_symbols)
    details['italic_variable_ratio'] = italic_variable_ratio
    details['varied_size_ratio'] = varied_size_ratio
    details['operator_ratio'] = operator_ratio
    details['bracket_chars'] = bracket_chars
    details['has_equals_sign'] = has_equals_sign
    details['has_fraction_bar_candidate'] = has_fraction_bar_candidate
    details['prose_word_ratio'] = prose_word_ratio
    details['numeric_ratio'] = numeric_ratio
    details['alpha_ratio'] = alpha_ratio
    details['ends_with_punctuation'] = ends_with_punctuation


    # --- Apply Refined Heuristics with Scoring ---
    score = 0
    # Strong indicators
    if math_font_ratio > MATH_FONT_THRESHOLD: score += 3
    if math_symbol_ratio > MATH_SYMBOL_THRESHOLD: score += 3
    elif len(distinct_math_symbols) >= 3: score += 2
    elif math_symbol_chars > 0: score += 1

    # Medium indicators
    medium_indicators = 0
    if varied_size_ratio > 0.05: medium_indicators += 1
    if italic_variable_ratio > ITALIC_VAR_THRESHOLD and (operator_chars > 0 or math_symbol_chars > 0): medium_indicators += 1
    if operator_ratio > 0.05: medium_indicators += 1
    if bracket_chars > 1: medium_indicators += 1
    if has_equals_sign: medium_indicators += 1
    if has_fraction_bar_candidate and varied_size_ratio > 0: medium_indicators += 1

    if medium_indicators >= 3: score += 2
    elif medium_indicators >= 1: score += 1

    # Contextual adjustments
    if max_span_len < 6 and total_spans > 3: score += 1
    if numeric_ratio > 0.7 and alpha_chars == 0 and operator_chars < 1 and not has_equals_sign: score -= 2 # Penalize purely numeric blocks without operators/equals
    if prose_word_ratio > MAX_PROSE_WORD_RATIO: score -= 3 # Increased penalty for high prose content
    if num_lines == 1 and total_chars > 100 and score < 5: score -= 2 # Increased penalty for long single lines
    if ends_with_punctuation and score < 4 and prose_word_ratio > 0.3: score -= 2 # Penalize text ending in punctuation

    # --- Decision ---
    is_math = score >= MATH_SCORE_THRESHOLD
    # Define ambiguity: score is above a minimum but below the main threshold
    is_ambiguous = score >= AMBIGUOUS_MATH_SCORE_MIN and score < MATH_SCORE_THRESHOLD and not is_math # Ensure not already classified as math

    # Add checks for conflicting indicators that might make it ambiguous even if score is high
    if is_math and (details.get('strong_negative_prose_match', False) or details.get('dominant_text_font', False)) and score < (MATH_SCORE_THRESHOLD + 2): # Example: penalize if high score but also looks like prose/text
         is_math = False
         is_ambiguous = True
         details['reclassified_to_ambiguous_due_to_prose_or_font'] = True


    return {'is_math': is_math, 'is_ambiguous': is_ambiguous, 'score': score, 'details': details}


def _spans_to_latex(spans, base_font_size, symbol_map, func_names):
    """Helper to convert a list of spans (potentially a fraction part) to LaTeX."""
    line_str = ""
    script_active = None
    prev_span_bbox = None
    # last_x1 = spans[0].get("bbox", [0]*4)[0] if spans and spans[0].get("bbox") else 0 # Not used

    for span in spans:
        text = span.get("text", "")
        if not text: continue

        font_size = span.get("font_size", 0)
        # is_super_flag = span.get("is_superscript") # Relying more on vertical position
        span_bbox = span.get("bbox")
        origin_y = span.get("origin", [0,0])[1]
        span_height = span_bbox[3] - span_bbox[1] if span_bbox else font_size

        # --- Refined Script Detection based on Vertical Position ---
        current_script = None
        if base_font_size and font_size > 0 and span_bbox and prev_span_bbox:
            size_diff = font_size - base_font_size
            prev_baseline_y = prev_span_bbox[3] # Use previous span's bottom as baseline estimate
            vertical_offset = origin_y - prev_baseline_y

            # Check for significant vertical offset AND smaller font size
            # Thresholds adjusted for stricter vertical check
            if size_diff < -VARIED_SIZE_THRESHOLD: # Font is significantly smaller
                if vertical_offset < -span_height * 0.4: # Clearly above previous baseline
                    current_script = 'super'
                elif vertical_offset > span_height * 0.3: # Clearly below previous baseline
                     current_script = 'sub'
            # Consider is_superscript flag only if strong vertical offset confirms it
            # elif is_super_flag and origin_y < prev_baseline_y - span_height * 0.3:
            #      current_script = 'super'
        # --- End Refined Script Detection ---

        # Add space based on horizontal distance
        if span_bbox and prev_span_bbox and span_bbox[0] > prev_span_bbox[2] + 1.5:
             line_str += " "

        # Handle script transitions
        if current_script != script_active:
            if script_active: line_str += "}"
            if current_script == 'super': line_str += "^{"
            elif current_script == 'sub': line_str += "_{"
            script_active = current_script

        # --- Text Processing ---
        processed_text = ""
        for char in text:
            if char in symbol_map:
                processed_text += symbol_map[char]
            else:
                # Escape special LaTeX characters
                if char in ['{', '}', '_', '^', '%', '$', '#', '&']:
                    processed_text += "\\" + char
                elif char == '\\': # Handle backslash itself
                     processed_text += "\\\\"
                else:
                    processed_text += char

        stripped_text = processed_text.strip()
        if stripped_text in func_names:
             processed_text = f"\\{stripped_text} " # Add a space after function names

        line_str += processed_text
        # --- End Text Processing ---

        prev_span_bbox = span_bbox
        # if span_bbox: last_x1 = span_bbox[2] # Not used

    if script_active: line_str += "}"
    return line_str.strip()


def convert_math_to_latex(segment):
    """Attempts to convert a math-flagged segment into a LaTeX string, with basic fraction and sqrt detection."""
    # (Uses updated _spans_to_latex helper)
    # This function can now be called on segments initially flagged as MATH_BLOCK_TYPE or even ambiguous ones
    # It returns None if conversion is not possible or results in empty LaTeX.

    symbol_map = {
         '≈' : '\\approx',  '−' : '-',  '≠' : '\\neq',  '≤' : '\\leq',  '≥' : '\\geq',
         '∑' : '\\sum',  '∫' : '\\int',  '∂' : '\\partial',  '∇' : '\\nabla',  '√' : '\\sqrt',
         '∞' : '\\infty', '±': '\\pm', '×': '\\times', '÷': '\\div',  '∈' : '\\in',
         '∉' : '\\notin',  '⊂' : '\\subset',  '⊃' : '\\supset',  '⊆' : '\\subseteq',  '⊇' : '\\supseteq',
         '∧' : '\\land',  '∨' : '\\lor', '¬': '\\neg',  '∀' : '\\forall',  '∃' : '\\exists',
         '∴' : '\\therefore',  '∵' : '\\because',
        'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon',
        'ζ': '\\zeta', 'η': '\\eta', 'θ': '\\theta', 'ι': '\\iota', 'κ': '\\kappa',
        'λ': '\\lambda', 'μ': '\\mu', 'ν': '\\nu', 'ξ': '\\xi', 'ο': 'o', 'π': '\\pi',
        'ρ': '\\rho', 'ς': '\\varsigma', 'σ': '\\sigma', 'τ': '\\tau', 'υ': '\\upsilon',
        'φ': '\\phi', 'χ': '\\chi', 'ψ': '\\psi', 'ω': '\\omega',
        'Γ': '\\Gamma', 'Δ': '\\Delta', 'Θ': '\\Theta', 'Λ': '\\Lambda', 'Ξ': '\\Xi',
        'Π': '\\Pi', 'Σ': '\\Sigma', 'Φ': '\\Phi', 'Ψ': '\\Psi', 'Ω': '\\Omega',
    }
    func_names = {'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'det', 'lim', 'sum', 'prod', 'min', 'max'}

    latex_str = ""
    base_font_size = None
    all_sizes = [s.get("font_size", 0) for ln in segment.get("lines", []) for s in ln.get("spans", []) if s.get("font_size", 0) > 0]
    if all_sizes: base_font_size = statistics.median(all_sizes)

    processed_parts = []
    all_spans_with_line_info = []
    for line_idx, line in enumerate(segment.get("lines", [])):
        line_bbox = line.get("bbox")
        if not line_bbox: continue
        line_center_y = (line_bbox[1] + line_bbox[3]) / 2
        for span_idx, span in enumerate(line.get("spans", [])):
            all_spans_with_line_info.append({
                'span': span, 'line_idx': line_idx, 'span_idx': span_idx,
                'line_center_y': line_center_y, 'processed': False
            })

    all_spans_with_line_info.sort(key=lambda x: (x['line_idx'], x['span'].get('bbox', [0]*4)[0]))

    i = 0
    while i < len(all_spans_with_line_info):
        item = all_spans_with_line_info[i]
        if item['processed']:
            i += 1
            continue

        span = item['span']
        text = span.get("text", "")
        is_fraction_bar = False
        is_sqrt_symbol = False

        # --- Fraction Detection ---
        # Check if the current span is a potential fraction bar ('-' or '−')
        if text.strip() and all(c in ('-', '−') for c in text.strip()):
            # Look for potential numerator (previous span) and denominator (next span)
            if i > 0 and i < len(all_spans_with_line_info) - 1:
                prev_item = all_spans_with_line_info[i-1]
                next_item = all_spans_with_line_info[i+1]
                span_bbox = span.get("bbox")
                prev_bbox = prev_item['span'].get("bbox")
                next_bbox = next_item['span'].get("bbox")

                if span_bbox and prev_bbox and next_bbox:
                    bar_center_y = (span_bbox[1] + span_bbox[3]) / 2
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    next_center_y = (next_bbox[1] + next_bbox[3]) / 2

                    # Check vertical alignment relative to the bar and horizontal overlap
                    # Adjusted vertical thresholds for stricter check
                    if prev_center_y < bar_center_y - (span_bbox[3]-span_bbox[1])*0.2 and \
                       next_center_y > bar_center_y + (span_bbox[3]-span_bbox[1])*0.2 and \
                       max(prev_bbox[0], span_bbox[0]) < min(prev_bbox[2], span_bbox[2]): # Check for horizontal overlap

                        # Recursively process potential numerator and denominator spans
                        # This basic implementation only handles single spans as num/den.
                        # A more robust version would group adjacent spans horizontally.
                        num_latex = _spans_to_latex([prev_item['span']], base_font_size, symbol_map, func_names)
                        den_latex = _spans_to_latex([next_item['span']], base_font_size, symbol_map, func_names)

                        if num_latex and den_latex:
                            processed_parts.append(f"\\frac{{{num_latex}}}{{{den_latex}}}")
                            is_fraction_bar = True
                            # Mark the processed spans
                            all_spans_with_line_info[i-1]['processed'] = True
                            item['processed'] = True
                            all_spans_with_line_info[i+1]['processed'] = True
                            i += 2 # Skip the next two spans as they were part of the fraction
                            continue

        # --- Basic Sqrt Detection ---
        if text.strip() ==  '√' :
             is_sqrt_symbol = True
             # Look for the content under the sqrt symbol (next span)
             if i + 1 < len(all_spans_with_line_info):
                 next_item = all_spans_with_line_info[i+1]
                 next_bbox = next_item['span'].get("bbox")
                 span_bbox = span.get("bbox")
                 # Check if the next span is roughly vertically aligned with the sqrt symbol's body
                 if next_bbox and span_bbox and abs(((next_bbox[1]+next_bbox[3])/2) - ((span_bbox[1]+span_bbox[3])/2)) < (span_bbox[3]-span_bbox[1]) * 0.5 :
                     # Recursively process the content under the sqrt
                     # Again, this is basic and assumes a single span under the sqrt.
                     sqrt_content_latex = _spans_to_latex([next_item['span']], base_font_size, symbol_map, func_names)
                     processed_parts.append(f"\\sqrt{{{sqrt_content_latex}}}")
                     item['processed'] = True
                     next_item['processed'] = True
                     i += 2 # Skip the next span
                     continue
                 else:
                     # If no content found immediately after, just output the sqrt symbol
                     processed_parts.append("\\sqrt")
                     item['processed'] = True
                     i += 1
                     continue
             else:
                 # If sqrt is the last span, just output the symbol
                 processed_parts.append("\\sqrt")
                 item['processed'] = True
                 i += 1
                 continue

        # Normal processing if not a recognized fraction bar or sqrt symbol
        if not is_fraction_bar and not is_sqrt_symbol:
            span_latex = _spans_to_latex([span], base_font_size, symbol_map, func_names)
            processed_parts.append(span_latex)
            item['processed'] = True
            i += 1

    latex_content = " ".join(p for p in processed_parts if p)
    if not latex_content or latex_content.strip() == "$$": return None # Return None for empty results

    # Basic check to avoid converting simple text that was borderline math
    # You might refine this based on the original math score or other details
    if len(latex_content.replace('$', '').strip()) < 5 and segment.get("processing_flags", {}).get("math_score", 0) < (MATH_SCORE_THRESHOLD - 1):
         return None


    return f"${latex_content}$"


def reconstruct_code_text(code_block_segments):
    """Reconstructs code text, preserving line breaks between original segments and attempting indentation."""
    # (Same as v11/v14)
    combined_lines = []
    min_indent = float('inf')
    lines_data = []
    indent_levels = []

    segment_index = 0
    for segment in code_block_segments:
        block_x0 = segment.get("bbox", [0,0,0,0])[0]
        if segment.get("block_type") in [TEXT_BLOCK_TYPE, GAP_BLOCK_TYPE]: # Include GAP_BLOCK_TYPE for text content
            lines_in_segment = segment.get("lines", [])
            if not lines_in_segment and segment.get("block_type") == GAP_BLOCK_TYPE:
                 # Handle GAP_BLOCK_TYPE text directly if no 'lines' structure
                 gap_lines = segment.get("text", "").splitlines()
                 y0_start = segment.get("bbox", [0,0,0,0])[1]
                 line_height_est = (segment.get("bbox", [0,0,0,0])[3] - y0_start) / len(gap_lines) if gap_lines else 10
                 x0_est = block_x0
                 for line_idx, line_text in enumerate(gap_lines):
                      y0 = y0_start + line_idx * line_height_est
                      lines_data.append((segment_index, line_idx, y0, x0_est, line_text))
                      if line_text.strip():
                          min_indent = min(min_indent, x0_est)
                          indent_levels.append(x0_est)
            else:
                for line_idx, line in enumerate(lines_in_segment):
                    first_span = line.get("spans", [{}])[0]
                    x0 = first_span.get("origin", [block_x0, 0])[0] if first_span.get("origin") else line.get("bbox", [block_x0,0,0,0])[0]
                    y0 = line.get("bbox", [0,0,0,0])[1]
                    raw_line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                    lines_data.append((segment_index, line_idx, y0, x0, raw_line_text))
                    if raw_line_text.strip():
                        min_indent = min(min_indent, x0)
                        indent_levels.append(x0)
        segment_index += 1

    if min_indent == float('inf'): min_indent = 0
    lines_data.sort(key=lambda item: (item[2], item[3])) # Sort by y0 then x0

    indent_width = INDENT_WIDTH_ESTIMATE if INDENT_WIDTH_ESTIMATE > 0 else 18
    if len(indent_levels) > 1:
        # Attempt to find a common indentation difference
        filtered_indents = [il for il in indent_levels if il > min_indent + 1]
        if filtered_indents:
            unique_indents = sorted(list(set(filtered_indents)))
            diffs_from_min = [ui - min_indent for ui in unique_indents]
            diffs_between = [unique_indents[i] - unique_indents[i-1] for i in range(1, len(unique_indents))]
            all_diffs = diffs_from_min + diffs_between
            positive_diffs = [d for d in all_diffs if d > MIN_INDENT_DIFF]
            if positive_diffs:
                # Use median or most common difference for indent width
                detected_indent = statistics.median(positive_diffs)
                if len(positive_diffs) > 2:
                    counts = {}
                    for d in positive_diffs:
                        # Quantize differences to group similar values
                        quantized = round(d / 4.0) * 4.0 # Quantize to nearest 4 pts (adjust as needed)
                        if quantized > 0: counts[quantized] = counts.get(quantized, 0) + 1
                    if counts:
                         most_common_diff = max(counts, key=counts.get)
                         if MIN_INDENT_DIFF < most_common_diff < MAX_INDENT_WIDTH: detected_indent = most_common_diff

                if MIN_INDENT_DIFF < detected_indent < MAX_INDENT_WIDTH: indent_width = detected_indent
                # print(f"DEBUG: Detected indent width: {indent_width:.1f}")

    last_segment_index = -1
    for seg_idx, line_idx, _, x0, text in lines_data:
        # Add a blank line between segments if the previous line wasn't empty
        if seg_idx != last_segment_index and last_segment_index != -1:
            if combined_lines and combined_lines[-1].strip():
                 combined_lines.append("")

        # Calculate indentation based on min_indent and detected indent_width
        indent_pixels = max(0, x0 - min_indent)
        indent_level = round(indent_pixels / indent_width) if indent_width > 0 else 0
        leading_spaces = "    " * indent_level # Use 4 spaces per indent level
        combined_lines.append(leading_spaces + text)
        last_segment_index = seg_idx

    return "\n".join(combined_lines)


# Helper function to group consecutive code segments
def group_code_segments(all_segments, start_index):
    """Groups consecutive segments that are likely code."""
    code_block_segments = [all_segments[start_index]]
    combined_bbox = list(all_segments[start_index].get("bbox", [0,0,0,0]))
    j = start_index + 1
    num_segments = len(all_segments)

    while j < num_segments:
        next_segment = all_segments[j]
        # Check if next segment is on the same page and likely code (using detailed check but only the 'is_code' part for grouping)
        # Note: This grouping is done BEFORE the second refinement pass, so it groups based on initial 'is_code'
        next_is_code_result = is_likely_code_segment_detailed(next_segment)
        if next_segment.get("page_number") == all_segments[start_index].get("page_number") and \
           next_is_code_result['is_code']:
            # Check vertical proximity
            v_distance = calculate_vertical_distance(combined_bbox, next_segment.get("bbox"))
            proximity_threshold = GAP_CODE_MAX_V_DISTANCE if code_block_segments[0].get("block_type") == GAP_BLOCK_TYPE else CODE_MAX_V_DISTANCE
            if v_distance < proximity_threshold:
                code_block_segments.append(next_segment)
                # Update combined bbox
                next_bbox = next_segment.get("bbox", [0,0,0,0])
                if next_bbox != [0,0,0,0]:
                    combined_bbox[0] = min(combined_bbox[0], next_bbox[0])
                    combined_bbox[1] = min(combined_bbox[1], next_bbox[1])
                    combined_bbox[2] = max(combined_bbox[2], next_bbox[2])
                    combined_bbox[3] = max(combined_bbox[3], next_bbox[3])
                j += 1
            else: break # Too far vertically
        else: break # Not code or different page

    return code_block_segments, j


# =============================================
# --- NEW REFINEMENT FUNCTIONS (PLACEHOLDERS) ---
# =============================================

def contains_inline_math_pattern(segment_text):
    """
    Placeholder: Implement logic to detect patterns common in inline math
    that might have been missed by the main math heuristic.
    Examples: text containing single characters surrounded by operators, simple equations
    like 'y = 2x + 5' in a text block, or text with mixed alpha and math symbols
    without full math font/size variations.
    """
    # This is where you'll add specific regex or text analysis for inline math.
    # Example: Simple regex for variable = expression pattern
    if re.search(r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9+\-*/]+\b', segment_text):
        return True
    # Example: Text with a high density of math symbols but low varied size/math font
    # You would use the details from the first pass here
    # if segment.get("processing_flags", {}).get("details", {}).get("math_symbol_ratio", 0) > 0.03 and \
    #    segment.get("processing_flags", {}).get("details", {}).get("varied_size_ratio", 0) < 0.05:
    #     return True

    return False # Default: Does not look like inline math


def convert_inline_math(segment_text):
    """
    Placeholder: Implement logic to convert detected inline math patterns to LaTeX.
    This might be simpler than the full math conversion, focusing on basic symbols and structure.
    """
    # This is where you'll add conversion logic for inline math patterns.
    # You might use regex to find patterns and replace them with LaTeX equivalents.
    # Example: Replace simple operators and equals sign
    latex_text = segment_text.replace('=', '=').replace('+', '+').replace('-', '-').replace('*', '\\times').replace('/', '\\div')
    # You might need to handle superscripts/subscripts if they appear in inline text
    # Example: simple x^2 -> x^2
    latex_text = re.sub(r'([a-zA-Z0-9])\^([0-9]+)', r'\1^{\2}', latex_text)

    # Add math delimiters if they are not already present
    if not latex_text.strip().startswith('$') or not latex_text.strip().endswith('$'):
         latex_text = f"${latex_text}$"

    return latex_text # Return the converted LaTeX string


def looks_like_code_comment(segment):
    """
    Placeholder: Implement logic to detect patterns common in code comments
    or simple code-like text within what was classified as ambiguous text.
    Examples: lines starting with '#', lines with code keywords but prose-like structure,
    text blocks immediately following a code block with similar indentation.
    """
    segment_text = segment.get("text", "")
    lines = segment.get("lines", [])

    # Example: Check if a significant portion of lines start with '#'
    comment_lines = sum(1 for line in lines if line.get("text", "").strip().startswith('#'))
    if lines and comment_lines / len(lines) > 0.5:
        return True

    # Example: Check for code keywords combined with prose-like structure (low operator/mono ratio)
    # You would use the details from the first pass here
    # details = segment.get("processing_flags", {}).get("details", {})
    # if details.get("keyword_hits", 0) > 0 and details.get("operator_ratio", 0) < 0.01 and details.get("mono_ratio", 0) < 0.1:
    #      return True

    # Example: Check indentation relative to a preceding code block (requires context)
    # This would need access to the list of all segments and the current segment's index.
    # If the segment follows a code block and has similar indentation, it might be a comment.

    return False # Default: Does not look like a code comment

# Add other refinement helper functions as needed...
# def is_contextually_code(segment, all_segments, segment_index):
#     """
#     Placeholder: Check if a segment is likely code based on its context (surrounding segments).
#     """
#     # Implement logic to look at segments before and after the current one.
#     # If surrounded by code, it's more likely to be code.
#     pass


# =============================================
# --- Core Processing Logic (Multi-Pass) ---
# =============================================

def process_segments_multi_pass(segments):
    """
    Processes segments in multiple passes:
    1. Classify clear code/math and identify ambiguous segments.
    2. Refine ambiguous segments.
    """
    processed_segments = []
    ambiguous_segments = []
    # Keep original segments with their indices for contextual analysis if needed in refinement
    segments_with_indices = list(enumerate(segments))
    num_segments = len(segments)

    # --- Pass 1: Initial Classification and Ambiguity Identification ---
    print("Pass 1: Initial Classification and Ambiguity Identification...")
    i = 0
    while i < num_segments:
        segment_index, current_segment = segments_with_indices[i]
        segment_processed_in_pass1 = False

        # --- Math Segment Identification ---
        math_result = segment_is_math_detailed(current_segment)
        if math_result['is_math']:
            # Attempt LaTeX conversion immediately for clear math
            latex_text = convert_math_to_latex(current_segment)
            if latex_text:
                 current_segment["block_type"] = LATEX_MATH_TYPE
                 current_segment["latex_text"] = latex_text
            else:
                 # If conversion fails, keep as MATH_BLOCK_TYPE or even TEXT_BLOCK_TYPE depending on confidence
                 current_segment["block_type"] = MATH_BLOCK_TYPE # Or TEXT_BLOCK_TYPE if very low confidence
                 # Store the math score/details even if not converted, for potential later review
                 current_segment["processing_details"] = math_result['details']
                 current_segment["processing_score"] = math_result['score']

            processed_segments.append(current_segment)
            segment_processed_in_pass1 = True
            i += 1

        # --- Code Block Reconstruction ---
        if not segment_processed_in_pass1:
            code_result = is_likely_code_segment_detailed(current_segment)
            if code_result['is_code']:
                # Group consecutive code segments
                code_block_segments, j = group_code_segments(segments, i) # Pass original segments and current index
                combined_code_text = reconstruct_code_text(code_block_segments)

                # Create a new combined segment for the code block
                combined_bbox = list(code_block_segments[0].get("bbox", [0,0,0,0]))
                for seg in code_block_segments[1:]:
                     next_bbox = seg.get("bbox", [0,0,0,0])
                     if next_bbox != [0,0,0,0]:
                         combined_bbox[0] = min(combined_bbox[0], next_bbox[0])
                         combined_bbox[1] = min(combined_bbox[1], next_bbox[1])
                         combined_bbox[2] = max(combined_bbox[2], next_bbox[2])
                         combined_bbox[3] = max(combined_bbox[3], next_bbox[3])


                processed_segments.append({
                    "id": str(uuid.uuid4()), # New ID for the combined block
                    "source_file": current_segment.get("source_file"), "category": current_segment.get("category"),
                    "page_number": current_segment.get("page_number"), "page_width": current_segment.get("page_width"), "page_height": current_segment.get("page_height"),
                    "block_type": CODE_BLOCK_TYPE, "bbox": combined_bbox,
                    "text": combined_code_text,
                    "original_segment_ids": [seg.get("id") for seg in code_block_segments],
                    # Store classification details from the first segment of the group if helpful
                    "processing_details": code_result['details'],
                    "processing_score": code_result['score']
                })
                segment_processed_in_pass1 = True
                i = j # Move index past the processed group

        # --- Identify Ambiguous Segments ---
        if not segment_processed_in_pass1:
             # If not classified as clear math or code, check for ambiguity
             if math_result['is_ambiguous'] or code_result['is_ambiguous'] or \
                (math_result['score'] >= AMBIGUOUS_MATH_SCORE_MIN and code_result['score'] >= AMBIGUOUS_CODE_SCORE_MIN):
                 # Flag as ambiguous and store classification details for the second pass
                 current_segment["processing_flags"] = {"ambiguous": True}
                 current_segment["processing_details"] = {
                     "math": math_result['details'],
                     "code": code_result['details']
                 }
                 current_segment["processing_score"] = {
                     "math": math_result['score'],
                     "code": code_result['score']
                 }
                 ambiguous_segments.append(current_segment)
             else:
                 # Keep Other Segment Types (including non-ambiguous text, tables, images etc.)
                 # Clean up potential replacement characters in text blocks
                 if current_segment.get("block_type") == TEXT_BLOCK_TYPE and isinstance(current_segment.get("text"), str):
                      current_segment["text"] = current_segment["text"].replace("\uFFFD", "?")
                 processed_segments.append(current_segment) # Add non-ambiguous segments to the processed list
             i += 1 # Move to the next segment if not part of a grouped code block


    # --- Pass 2: Refine Ambiguous Segments ---
    print(f"Pass 2: Refining {len(ambiguous_segments)} ambiguous segments...")
    # Pass the original full list of segments for contextual analysis if needed
    refined_ambiguous_segments = refine_ambiguous_segments(ambiguous_segments, segments)
    processed_segments.extend(refined_ambiguous_segments) # Add results of refinement

    return processed_segments

# =============================================
# --- Refinement Logic Implementation ---
# =============================================

def refine_ambiguous_segments(ambiguous_segments, all_segments):
    """
    Applies refined heuristics to segments flagged as ambiguous.
    Takes the full list of segments for contextual analysis.
    """
    refined_segments = []
    # Create a dictionary mapping segment id to segment and index for quick lookup
    segment_lookup = {seg.get("id"): (idx, seg) for idx, seg in enumerate(all_segments)}

    for segment in ambiguous_segments:
        segment_text = segment.get("text", "")
        segment_id = segment.get("id")
        original_index = segment_lookup.get(segment_id, (None, None))[0] # Get original index

        reclassified = False

        # Access scores and details from the first pass
        processing_details = segment.get("processing_details", {})
        processing_score = segment.get("processing_score", {})
        math_details = processing_details.get("math", {})
        code_details = processing_details.get("code", {})
        math_score = processing_score.get("math", 0)
        code_score = processing_score.get("code", 0)


        # --- Add your refined heuristics here ---
        # Use the helper functions you implement.
        # You can use the stored details and scores in your refinement logic.

        # Example Refinement 1: Check if it's likely inline math
        if not reclassified and contains_inline_math_pattern(segment_text):
             latex_text = convert_inline_math(segment_text)
             if latex_text:
                 segment["block_type"] = LATEX_MATH_TYPE
                 segment["latex_text"] = latex_text
                 segment["refinement_reason"] = "Inline Math Pattern"
                 refined_segments.append(segment)
                 reclassified = True

        # Example Refinement 2: Check if it's likely a code comment or simple code line
        if not reclassified and looks_like_code_comment(segment): # This helper can use segment details
             segment["block_type"] = CODE_BLOCK_TYPE # Or a more specific type if defined
             segment["refinement_reason"] = "Code Comment/Simple Code Pattern"
             # You might want to reconstruct text here if it was a text block initially
             if segment.get("block_type") == TEXT_BLOCK_TYPE:
                  segment["text"] = reconstruct_code_text([segment]) # Reconstruct just this segment
             refined_segments.append(segment)
             reclassified = True

        # Example Refinement 3: Contextual analysis (requires original_index and all_segments)
        # if not reclassified and original_index is not None:
        #      if is_contextually_code(segment, all_segments, original_index):
        #           segment["block_type"] = CODE_BLOCK_TYPE
        #           segment["refinement_reason"] = "Contextual Code"
        #           if segment.get("block_type") == TEXT_BLOCK_TYPE:
        #                segment["text"] = reconstruct_code_text([segment])
        #           refined_segments.append(segment)
        #           reclassified = True
        #      elif is_contextually_math(segment, all_segments, original_index): # You'd need to implement this
        #           segment["block_type"] = LATEX_MATH_TYPE # Or MATH_BLOCK_TYPE
        #           # Attempt math conversion if not already done
        #           if "latex_text" not in segment:
        #                latex_text = convert_math_to_latex(segment)
        #                if latex_text: segment["latex_text"] = latex_text
        #           segment["refinement_reason"] = "Contextual Math"
        #           refined_segments.append(segment)
        #           reclassified = True


        # --- Default: If not reclassified, keep original type (likely TEXT_BLOCK_TYPE) ---
        if not reclassified:
            # You might assign a specific "ambiguous text" type here if you defined one
            # segment["block_type"] = AMBIGUOUS_TEXT_TYPE # Example
            segment["refinement_reason"] = "Remains Ambiguous"
            # Clean up potential replacement characters again if needed
            if segment.get("block_type") == TEXT_BLOCK_TYPE and isinstance(segment.get("text"), str):
                 segment["text"] = segment["text"].replace("\uFFFD", "?")
            refined_segments.append(segment)


    return refined_segments


# --- Main Processing Function ---
def process_jsonl_file(input_path, output_path):
    """ Reads input JSONL, processes segments page by page using multi-pass, writes output JSONL. """
    print(f"Processing {input_path} -> {output_path}")
    segments_by_page = {}
    total_lines = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                total_lines += 1
                try:
                    segment = json.loads(line)
                    page_num = segment.get("page_number")
                    if page_num is not None:
                        if page_num not in segments_by_page: segments_by_page[page_num] = []
                        segments_by_page[page_num].append(segment)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line {line_num + 1} in {input_path}")
    except FileNotFoundError: print(f"Error: Input file not found: {input_path}"); return
    except Exception as e: print(f"Error reading {input_path}: {e}"); traceback.print_exc(); return

    print(f"Read {total_lines} segments from {len(segments_by_page)} pages.")

    processed_count = 0
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"CRITICAL ERROR: Failed to create output directory '{output_dir}'. Error: {e}"); return

        with open(output_path, 'w', encoding='utf-8') as outfile:
            use_tqdm = len(segments_by_page) > 10
            page_iterator = tqdm(sorted(segments_by_page.keys()), desc="Processing Pages", leave=False) if use_tqdm else sorted(segments_by_page.keys())

            for page_num in page_iterator:
                page_segments = segments_by_page.get(page_num, [])
                # Sort segments vertically on the page before processing
                page_segments.sort(key=lambda s: s.get("bbox", [0,0,0,0])[1])
                # Use the new multi-pass processing function
                processed_page_segments = process_segments_multi_pass(page_segments)
                for segment in processed_page_segments:
                    try:
                        # Remove temporary processing flags/details before writing if desired
                        # segment.pop("processing_flags", None)
                        # segment.pop("processing_details", None)
                        # segment.pop("processing_score", None)
                        json_line = json.dumps(segment, ensure_ascii=False, default=str)
                        outfile.write(json_line + '\n')
                        processed_count += 1
                    except Exception as write_err: print(f"Error writing segment {segment.get('id')}: {write_err}")

    except Exception as e: print(f"Error processing pages or writing output file {output_path}: {e}"); traceback.print_exc()

    print(f"Finished processing. Wrote {processed_count} processed segments.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Phase 2: Post-Processing and Structuring (Multi-Pass Refinement)...")
    if CONFIG is None or not CONFIG:
        print("CRITICAL ERROR: Configuration could not be loaded or is empty. Check config.yaml. Exiting.")
    else:
        os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)
        print(f"DEBUG: Script configured to use input directory: {INPUT_JSONL_DIR}")
        print(f"DEBUG: Script configured to use output directory: {OUTPUT_PROCESSED_DIR}")

        if INPUT_FILENAME:
            input_file_path = os.path.join(INPUT_JSONL_DIR, INPUT_FILENAME)
            print(f"DEBUG: Attempting to process single file: {input_file_path}")
            if os.path.exists(input_file_path):
                output_file_path = os.path.join(OUTPUT_PROCESSED_DIR, f"structured_{INPUT_FILENAME}")
                process_jsonl_file(input_file_path, output_file_path)
            else:
                print(f"Error: Specified input file not found: {input_file_path}")
        else:
            try:
                print(f"DEBUG: Attempting to list files in: {INPUT_JSONL_DIR}")
                all_files = [f for f in os.listdir(INPUT_JSONL_DIR) if f.lower().endswith('.jsonl')]
                if not all_files:
                    print(f"No JSONL files found in {INPUT_JSONL_DIR}")
                else:
                    print(f"Found {len(all_files)} JSONL files to process.")
                    file_iterator = tqdm(all_files, desc="Processing Files", unit="file")
                    for filename in file_iterator:
                        input_file_path = os.path.join(INPUT_JSONL_DIR, filename)
                        output_file_path = os.path.join(OUTPUT_PROCESSED_DIR, f"structured_{filename}")
                        process_jsonl_file(input_file_path, output_file_path)
            except FileNotFoundError:
                 print(f"Error: Input directory not found: {INPUT_JSONL_DIR}")
            except Exception as e:
                 print(f"An error occurred listing files in {INPUT_JSONL_DIR}: {e}")

    print("\nPost-processing script finished.")

# --- Add these to your config.yaml ---
# ambiguity_rules:
#   ambiguous_code_score_min: 1 # Minimum score for a text block to be considered potentially ambiguous code
#   ambiguous_math_score_min: 1 # Minimum score for a text block to be considered potentially ambiguous math
# # Note: Segments with scores between MIN and THRESHOLD for either type can be ambiguous
