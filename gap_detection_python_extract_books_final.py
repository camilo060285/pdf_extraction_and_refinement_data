# Full updated script with GAP DETECTION and TARGETED RE-EXTRACTION
# Aims to keep detailed "dict" output while recovering content from gaps.
# Set up for processing all files in the specified directories.

import fitz  # PyMuPDF
import json
import os
from tqdm import tqdm
import uuid # To generate unique IDs
import traceback # To print detailed error info
import shutil # To remove temporary files/directories
import statistics # For potential median calculations later
# import io # Uncomment if using OCR with Pillow/PIL
# from PIL import Image # Uncomment if using OCR with Pillow/PIL
# import pytesseract # Uncomment if using OCR

# --- Updated Core Extraction Function (v6.1 - Gap Detection Integrated) ---
def extract_structured_data_from_pdf(pdf_path, category="unknown"):
    """
    Extracts detailed structured data using 'dict', then attempts to find
    vertical gaps between blocks and re-extract text from those gaps using 'text'.
    Handles potential None values during table extraction more gracefully.
    """
    doc = None
    structured_data = [] # List to hold all segments for the document
    file_name = os.path.basename(pdf_path)

    # Configuration for gap detection
    GAP_THRESHOLD = 15 # Vertical points threshold to trigger gap check (tune as needed)

    try:
        # Attempt to open the PDF document
        doc = fitz.open(pdf_path)
    except Exception as e:
        # Print error if PDF cannot be opened and return empty list
        print(f"\nERROR: Failed to open PDF '{file_name}'. Skipping file. Error: {e}")
        return structured_data # Return empty list

    # Iterate through each page of the document
    for page_num in range(doc.page_count):
        page_data = [] # Segments extracted by 'dict' for this page
        gap_data = [] # Segments extracted from gaps for this page
        try:
            # Load the current page
            page = doc.load_page(page_num)
            page_width = page.rect.width
            page_height = page.rect.height

            # --- Stage 1: Primary Extraction using get_text("dict") ---

            # Extract Links (Once per page)
            links = page.get_links()
            page_links = []
            for link in links:
                 if link.get('kind') == fitz.LINK_URI and 'uri' in link:
                     link_data = {"uri": link['uri']}
                     link_rect = link.get('from', None)
                     if link_rect:
                          try:
                              # Attempt to extract text associated with the link
                              link_text = page.get_text("text", clip=link_rect).strip()
                              if link_text: link_data["text"] = link_text
                          except Exception as link_err:
                              # Ignore errors if link text extraction fails
                              pass
                     page_links.append(link_data)

            # Extract Drawings (Once per page)
            drawings = page.get_drawings()
            drawing_bboxes = []
            for path in drawings:
                 # Store bounding boxes of valid, non-empty drawings
                 if 'rect' in path and path['rect'].is_valid and not path['rect'].is_empty:
                     drawing_bboxes.append(list(path['rect']))

            # Extract Text Blocks using 'dict' method for detailed structure
            # Flags preserve images and whitespace which might help with layout
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES | fitz.TEXT_PRESERVE_WHITESPACE)
            blocks = page_dict.get('blocks', []) # Get list of blocks, default to empty list
            processed_block_bboxes = [] # Store bboxes of successfully processed blocks from this stage

            # Process each block found by get_text("dict")
            for block_id, block in enumerate(blocks):
                segment_id = str(uuid.uuid4()) # Generate unique ID for the segment
                block_bbox = list(block.get('bbox', (0,0,0,0))) # Get bounding box
                block_type = block.get('type', -1) # 0=text, 1=image

                # Skip block if its bounding box is invalid or has zero area
                if not block_bbox or (block_bbox[2]-block_bbox[0] <= 0 or block_bbox[3]-block_bbox[1] <= 0): continue

                # Initialize segment data dictionary
                segment_data = {
                    "id": segment_id, "source_file": file_name, "category": category,
                    "page_number": page_num + 1, "page_width": page_width, "page_height": page_height,
                    "block_id": block_id, "block_type": block_type, "bbox": block_bbox,
                }

                is_valid_segment = False # Flag to track if the segment contains useful data

                # Process TEXT blocks (type 0)
                if block_type == 0 and 'lines' in block:
                    lines_data = []; block_text_parts = [] # Initialize lists for lines and text parts
                    # Iterate through lines in the block
                    for line_num, line in enumerate(block.get('lines', [])):
                        line_bbox = list(line.get('bbox', (0,0,0,0)))
                        # Skip line if invalid/empty bounding box
                        if not line_bbox or (line_bbox[2]-line_bbox[0] <= 0 or line_bbox[3]-line_bbox[1] <= 0): continue
                        line_dir = line.get('dir', (1.0, 0.0)); line_wmode = line.get('wmode', 0)
                        spans_data = []; line_text_parts = [] # Initialize lists for spans within the line
                        # Iterate through spans in the line
                        for span_num, span in enumerate(line.get('spans', [])):
                            span_bbox = list(span.get('bbox', (0,0,0,0)))
                            span_text = span.get('text', '').strip() # Get and strip span text
                            # Skip span if invalid/empty bbox or empty text content
                            if not span_bbox or (span_bbox[2]-span_bbox[0] <= 0 or span_bbox[3]-span_bbox[1] <= 0) or not span_text: continue
                            # Extract detailed font and style information
                            span_font_size = span.get('size', 0.0); span_font_flags = span.get('flags', 0); span_font_name = span.get('font', 'Unknown')
                            span_color = span.get('color', 0); span_origin = list(span.get('origin', (0.0, 0.0)))
                            # Decode font flags into boolean properties
                            is_superscript = (span_font_flags & 1) > 0; is_italic = (span_font_flags & 2) > 0; is_serifed = (span_font_flags & 4) > 0
                            is_monospaced = (span_font_flags & 8) > 0; is_bold = (span_font_flags & 16) > 0
                            # Append span data
                            spans_data.append({
                                "span_id": f"{block_id}-{line_num}-{span_num}", "text": span_text, "bbox": span_bbox, "origin": span_origin,
                                "font_size": span_font_size, "font_flags": span_font_flags, "font_name": span_font_name, "color": span_color,
                                "is_superscript": is_superscript, "is_italic": is_italic, "is_serifed": is_serifed, "is_monospaced": is_monospaced, "is_bold": is_bold,
                            })
                            line_text_parts.append(span_text) # Collect text parts for the line
                        # If the line contained valid spans, reconstruct line text and add line data
                        if spans_data:
                            line_text = " ".join(line_text_parts) # Reconstruct line text by joining spans with spaces
                            lines_data.append({"line_id": f"{block_id}-{line_num}", "bbox": line_bbox, "dir": line_dir, "wmode": line_wmode, "spans": spans_data, "text": line_text})
                            block_text_parts.append(line_text) # Collect line texts for the block
                    # If the block contained valid lines, add line data and reconstructed block text
                    if lines_data:
                        segment_data["lines"] = lines_data; segment_data["text"] = "\n".join(block_text_parts); is_valid_segment = True

                # Process IMAGE blocks (type 1)
                elif block_type == 1:
                     img_width = block.get("width", 0); img_height = block.get("height", 0)
                     # Check for valid image dimensions
                     if img_width > 0 and img_height > 0:
                         segment_data["image_width"] = img_width; segment_data["image_height"] = img_height
                         segment_data["image_ext"] = block.get("ext", "unknown"); segment_data["image_xref"] = block.get("xref", 0) # Xref can be used to extract image bytes later if needed
                         is_valid_segment = True

                # If the segment (text or image) was valid, add page-level metadata and store it
                if is_valid_segment:
                    segment_data["links_on_page"] = page_links
                    segment_data["drawings_on_page"] = drawing_bboxes
                    page_data.append(segment_data)
                    processed_block_bboxes.append(fitz.Rect(block_bbox)) # Store bbox as fitz.Rect for sorting/comparison

            # Extract Table Information using PyMuPDF's find_tables()
            try:
                tables = page.find_tables() # Attempt to find tables on the page
                page_tables_data = []
                for table_count, table in enumerate(tables):
                    header_content = []; rows_content = []; cells_content = []
                    # Safely process header, rows, and cells, converting None to ""
                    if table.header and table.header.cells: header_content = [[str(h_cell) if h_cell is not None else "" for h_cell in row] for row in table.header.cells if row is not None]
                    if table.rows: rows_content = [[str(cell) if cell is not None else "" for cell in row] for row in table.rows if row is not None]
                    if table.cells: cells_content = [[str(cell) if cell is not None else "" for cell in row] for row in table.cells if row is not None]
                    # Skip adding the table if it has no header and no rows
                    if not header_content and not rows_content: continue
                    # Store extracted table data
                    table_data = {"table_id": f"table-{table_count}", "bbox": list(table.bbox), "header": header_content, "rows": rows_content, "row_count": table.row_count, "col_count": table.col_count, "cells": cells_content}
                    page_tables_data.append(table_data)
                    processed_block_bboxes.append(table.bbox) # Also consider table bboxes for gap detection
                # If tables were found, create a single segment to hold all table data for the page
                if page_tables_data:
                    table_segment_id = str(uuid.uuid4())
                    page_data.append({
                        "id": table_segment_id, "source_file": file_name, "category": category,
                        "page_number": page_num + 1, "page_width": page_width, "page_height": page_height,
                        "block_id": -1, # Special ID for table container
                        "block_type": 2, # Type code for table container
                        "bbox": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1], # Use page bbox for container
                        "tables_on_page": page_tables_data, # List of actual tables found
                        "links_on_page": page_links, # Add page-level meta
                        "drawings_on_page": drawing_bboxes # Add page-level meta
                    })
            except Exception as e:
                 # Silently ignore table extraction errors during normal runs
                 pass

            # --- Stage 2: Gap Detection and Targeted Re-extraction ---
            if processed_block_bboxes: # Only proceed if Stage 1 found any blocks/tables
                # Sort blocks primarily by top coordinate (y0), then left (x0) for consistent gap checking
                processed_block_bboxes.sort(key=lambda r: (r.y0, r.x0))

                page_top = 0
                page_bottom = page_height

                # Check gap between page top and first block
                first_block = processed_block_bboxes[0]
                if first_block.y0 > page_top + GAP_THRESHOLD:
                    gap_rect = fitz.Rect(page.rect.x0, page_top, page.rect.x1, first_block.y0)
                    try:
                        # Use simpler 'text' extraction within the gap rectangle
                        gap_text = page.get_text("text", clip=gap_rect, sort=True).strip()
                        if gap_text: # If text is found in the gap
                            gap_id = str(uuid.uuid4())
                            # Create a new segment for the gap text
                            gap_data.append({
                                "id": gap_id, "source_file": file_name, "category": category,
                                "page_number": page_num + 1, "page_width": page_width, "page_height": page_height,
                                "block_id": -5, # Special ID for gap-fill block
                                "block_type": 5, # New type code for gap-fill text
                                "bbox": list(gap_rect), # Bbox of the gap area itself
                                "text": gap_text, # The extracted text
                                "reason": "Gap at page top" # Note why this block exists
                            })
                    except Exception as gap_err:
                         pass # Ignore errors during gap extraction

                # Check gaps between consecutive blocks
                for i in range(len(processed_block_bboxes) - 1):
                    box_above = processed_block_bboxes[i]
                    box_below = processed_block_bboxes[i+1]
                    vertical_gap = box_below.y0 - box_above.y1 # Calculate vertical distance

                    # If the gap is larger than the threshold
                    if vertical_gap > GAP_THRESHOLD:
                        # Define the gap area (full page width between the two blocks)
                        gap_rect = fitz.Rect(page.rect.x0, box_above.y1, page.rect.x1, box_below.y0)
                        try:
                            gap_text = page.get_text("text", clip=gap_rect, sort=True).strip()
                            if gap_text: # If text is found
                                gap_id = str(uuid.uuid4())
                                gap_data.append({
                                    "id": gap_id, "source_file": file_name, "category": category,
                                    "page_number": page_num + 1, "page_width": page_width, "page_height": page_height,
                                    "block_id": -5, "block_type": 5, "bbox": list(gap_rect),
                                    "text": gap_text,
                                    "reason": f"Gap between block ending at {box_above.y1:.1f} and block starting at {box_below.y0:.1f}"
                                })
                        except Exception as gap_err:
                             pass # Ignore errors during gap extraction

                # Check gap between last block and page bottom
                last_block = processed_block_bboxes[-1]
                if page_bottom - last_block.y1 > GAP_THRESHOLD:
                    gap_rect = fitz.Rect(page.rect.x0, last_block.y1, page.rect.x1, page_bottom)
                    try:
                        gap_text = page.get_text("text", clip=gap_rect, sort=True).strip()
                        if gap_text: # If text is found
                            gap_id = str(uuid.uuid4())
                            gap_data.append({
                                "id": gap_id, "source_file": file_name, "category": category,
                                "page_number": page_num + 1, "page_width": page_width, "page_height": page_height,
                                "block_id": -5, "block_type": 5, "bbox": list(gap_rect),
                                "text": gap_text,
                                "reason": "Gap at page bottom"
                            })
                    except Exception as gap_err:
                        pass # Ignore errors during gap extraction


            # Combine original data (Stage 1) and gap data (Stage 2) for the page
            # Appending gap data ensures it comes after the primary blocks for that page
            structured_data.extend(page_data)
            structured_data.extend(gap_data)

            # --- Optional: OCR Placeholder ---
            # Consider adding OCR logic here if needed, potentially checking if
            # both page_data and gap_data are empty after processing.

        # Catch errors occurring during the processing of a single page
        except Exception as e:
            print(f"\nERROR processing page {page_num + 1} of '{file_name}': {e}")
            traceback.print_exc() # Show full traceback for page errors

    # Safely close the PDF document after processing all pages
    if doc:
        try:
            doc.close()
        except Exception as e:
            print(f"\nWarning: Error closing PDF '{file_name}': {e}")

    # Return the list of all extracted segments for the document
    return structured_data


# --- Book Processing and Combination Logic (Restored for full processing) ---
def process_book_by_book_and_combine(base_feed_dir, temp_output_base_dir, final_output_base_dir, skip_processed=True):
    """
    Processes each PDF book individually using the enhanced extraction function,
    saves detailed structured data to a temp file, then combines temp files
    per category into final category files.
    """
    print(f"Starting detailed processing in feed directory: {base_feed_dir}")
    print(f"Temporary book files will be saved to: {temp_output_base_dir}")
    print(f"Final category files will be saved to: {final_output_base_dir}")
    print(f"Skip already processed temporary files: {skip_processed}")

    # Create output directories if they don't exist
    try:
        os.makedirs(temp_output_base_dir, exist_ok=True)
        os.makedirs(final_output_base_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create output directories. Check permissions. Error: {e}")
        return

    # Find category subdirectories
    try:
        categories = [d for d in os.listdir(base_feed_dir) if os.path.isdir(os.path.join(base_feed_dir, d))]
    except FileNotFoundError:
        print(f"ERROR: Base feed directory not found at {base_feed_dir}")
        return
    except OSError as e:
        print(f"ERROR: Could not read base feed directory {base_feed_dir}. Error: {e}")
        return

    if not categories:
        print(f"No subdirectories (categories) found in {base_feed_dir}. Make sure subject folders are directly inside.")
        return

    print(f"Found {len(categories)} subject categories: {', '.join(categories)}")

    # --- Phase 1: Process Book by Book ---
    print("\n--- Phase 1: Processing Book by Book (Extracting Detailed Data) ---")
    all_books_processed_count = 0
    all_books_failed_count = 0
    all_books_skipped_count = 0

    # Iterate through each category directory
    for category in categories:
        category_feed_path = os.path.join(base_feed_dir, category)
        category_temp_output_path = os.path.join(temp_output_base_dir, category)

        # Create temporary output directory for the category
        try:
            os.makedirs(category_temp_output_path, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Could not create temp directory for category '{category}'. Skipping category. Error: {e}")
            continue

        # List PDF files in the category directory
        try:
            pdf_files = [f for f in os.listdir(category_feed_path) if f.lower().endswith('.pdf')]
        except FileNotFoundError:
            print(f"\n--- Skipping category '{category}': Directory not found ---")
            continue
        except OSError as e:
            print(f"\n--- Skipping category '{category}': Could not read directory. Error: {e} ---")
            continue

        if not pdf_files:
            print(f"\n--- Skipping category '{category}': No PDF files found ---")
            continue

        print(f"\n--- Processing {len(pdf_files)} files in category '{category}' ---")
        category_processed_count = 0; category_failed_count = 0; category_skipped_count = 0

        # Process each PDF file in the category
        for pdf_file in tqdm(pdf_files, desc=f"Books in '{category}'", leave=False, unit="book"):
            pdf_path = os.path.join(category_feed_path, pdf_file)
            # Sanitize filename to create a safe path for the temporary JSONL file
            safe_filename_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.splitext(pdf_file)[0])
            book_temp_jsonl_path = os.path.join(category_temp_output_path, f"{safe_filename_base}.jsonl")

            # Option to skip files if a temporary output file already exists
            if skip_processed:
                try:
                    # Check if temp file exists and has some content (size > 10 bytes)
                    if os.path.exists(book_temp_jsonl_path) and os.path.getsize(book_temp_jsonl_path) > 10:
                         category_skipped_count += 1
                         continue # Skip to the next file
                except OSError as e:
                     # Warn if status check fails, but attempt processing anyway
                     print(f"\nWarning: Could not check status of temp file {book_temp_jsonl_path}. Error: {e}")

            # Extract data from the current PDF
            try:
                # Call the core extraction function (now includes gap detection)
                book_data_segments = extract_structured_data_from_pdf(pdf_path, category=category)

                # Write extracted segments to the temporary JSONL file
                if book_data_segments: # Check if any data was extracted
                    with open(book_temp_jsonl_path, 'w', encoding='utf-8') as outfile:
                        for segment in book_data_segments:
                            try:
                                # Dump each segment as a JSON string on a new line
                                json_line = json.dumps(segment, ensure_ascii=False, default=str) # default=str handles potential non-serializable types
                                outfile.write(json_line + '\n')
                            except TypeError as json_err:
                                # Log error if a specific segment fails serialization
                                print(f"\nERROR serializing segment in '{pdf_file}', page {segment.get('page_number', '?')}. Skipping segment. Error: {json_err}")
                    category_processed_count += 1
                else:
                    # Handle cases where no data was extracted (e.g., empty PDF)
                    print(f"\nWarning: No data extracted or saved for '{pdf_file}'. Creating empty temp file.")
                    # Create an empty temp file to mark it as processed (if skip_processed is True)
                    open(book_temp_jsonl_path, 'w').close()
                    category_processed_count += 1 # Count as processed (attempted)

            # Catch fatal errors during processing of a single file
            except Exception as e:
                print(f"\nFATAL ERROR during processing or writing for '{pdf_file}': {e}")
                traceback.print_exc() # Print full traceback for debugging
                category_failed_count += 1
                # Attempt to remove potentially incomplete/corrupt temp file
                if os.path.exists(book_temp_jsonl_path):
                    try:
                        os.remove(book_temp_jsonl_path)
                    except OSError:
                        # Ignore errors if removal fails
                        pass

        # Print summary for the processed category
        print(f"\n--- Category '{category}' processing complete: {category_processed_count} processed/attempted, {category_failed_count} failed, {category_skipped_count} skipped ---")
        # Update overall counts
        all_books_processed_count += category_processed_count
        all_books_failed_count += category_failed_count
        all_books_skipped_count += category_skipped_count

    # Print overall summary for Phase 1
    print("\n--- Phase 1 Complete ---")
    print(f"Total books processed/attempted: {all_books_processed_count}")
    print(f"Total books failed: {all_books_failed_count}")
    print(f"Total books skipped (already processed): {all_books_skipped_count}")
    print("Individual book detailed JSONL files created/updated in:", temp_output_base_dir)

    # --- Phase 2: Combine Temporary Book Files ---
    print("\n--- Phase 2: Combining Book Files into Category Files ---")
    all_final_files_created = 0
    # Iterate through categories again to combine temp files
    for category in categories:
        category_temp_output_path = os.path.join(temp_output_base_dir, category)
        # Sanitize category name for the final output filename
        safe_category_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in category)
        final_category_jsonl_path = os.path.join(final_output_base_dir, f"{safe_category_name}_detailed.jsonl")

        # Skip if no temporary directory exists for this category
        if not os.path.isdir(category_temp_output_path):
             continue

        # List temporary JSONL files for the category
        try:
            temp_book_jsonl_files = [f for f in os.listdir(category_temp_output_path) if f.lower().endswith('.jsonl')]
        except OSError as e:
            print(f"--- Error reading temp directory for category '{category}'. Skipping combination. Error: {e} ---")
            continue

        # Skip if no temporary files were found
        if not temp_book_jsonl_files:
            continue

        print(f"--- Combining {len(temp_book_jsonl_files)} files for category '{category}' into {final_category_jsonl_path} ---")
        # Combine all temporary files into the final category file
        try:
            with open(final_category_jsonl_path, 'w', encoding='utf-8') as outfile:
                # Use tqdm for progress bar during combination
                for temp_jsonl_file in tqdm(temp_book_jsonl_files, desc=f"Combining '{category}'", leave=False, unit="file"):
                    temp_file_path = os.path.join(category_temp_output_path, temp_jsonl_file)
                    try:
                        # Skip empty temporary files
                        if os.path.getsize(temp_file_path) == 0: continue
                        # Read each line from the temp file and write to the final file
                        with open(temp_file_path, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                # Basic JSON validation before writing to final file
                                try:
                                   json.loads(line) # Check if the line is valid JSON
                                   outfile.write(line) # Write the valid line
                                except json.JSONDecodeError:
                                   # Warn about invalid lines found in temp files
                                   print(f"\nWarning: Skipping invalid JSON line in {temp_jsonl_file}: {line.strip()}")
                    except FileNotFoundError: print(f"\nWarning: Temporary file {temp_jsonl_file} not found during combination.")
                    except OSError as e: print(f"\nWarning: Could not read or get size of temp file {temp_jsonl_file}. Skipping. Error: {e}")
                    except Exception as e: print(f"\nError combining temporary file {temp_jsonl_file}: {e}")

            print(f"Successfully created/updated final category file: {final_category_jsonl_path}")
            all_final_files_created += 1

            # Optional: Clean up temporary directory for the category after successful combination
            # print(f" Cleaning up temporary files for '{category}'...")
            # try:
            #     shutil.rmtree(category_temp_output_path)
            #     print(f" Cleaned up temp directory: {category_temp_output_path}")
            # except Exception as e:
            #     print(f"\nError cleaning up temp directory {category_temp_output_path}: {e}")

        except Exception as e:
            # Catch errors during opening/writing the final category file
            print(f"\nError opening or writing to final category file {final_category_jsonl_path}: {e}")
            traceback.print_exc()

    # Print summary for Phase 2
    print("\n--- Phase 2 Complete ---")
    print(f"Total final category files created/updated: {all_final_files_created}")
    print("Final category detailed JSONL files created/updated in:", final_output_base_dir)


# --- Configuration (Update these paths as needed) ---
# Use forward slashes or raw strings for paths to avoid issues across OS
base_feed_directory = r'C:\apex_project\feed'
temp_output_base_directory = r'C:\apex_project\temp_extracted_jsonl_detailed'
final_output_base_directory = r'C:\apex_project\final_processed_jsonl_by_category_detailed'
# Set to False if you want to force re-processing of all PDFs every time
SKIP_EXISTING_TEMP_FILES = True
# ---------------------

# --- Main Execution Block (Restored for Full Processing) ---
if __name__ == "__main__":
    print("Script started for full processing...")
    # Basic checks for directory existence before starting the main process
    if not os.path.isdir(base_feed_directory):
        print(f"Error: Base feed directory not found at '{base_feed_directory}'")
        print("Please ensure the path is correct and the directory exists.")
    elif not os.path.isdir(temp_output_base_directory):
         # Warn but proceed; the function will create the directory
         print(f"Warning: Temporary output directory not found at '{temp_output_base_directory}'. It will be created.")
         process_book_by_book_and_combine(
             base_feed_directory,
             temp_output_base_directory,
             final_output_base_directory,
             skip_processed=SKIP_EXISTING_TEMP_FILES
         )
    elif not os.path.isdir(final_output_base_directory):
        # Warn but proceed; the function will create the directory
        print(f"Warning: Final output directory not found at '{final_output_base_directory}'. It will be created.")
        process_book_by_book_and_combine(
             base_feed_directory,
             temp_output_base_directory,
             final_output_base_directory,
             skip_processed=SKIP_EXISTING_TEMP_FILES
         )
    else:
        # Call the main processing function if all directories seem okay (or will be created)
        process_book_by_book_and_combine(
             base_feed_directory,
             temp_output_base_directory,
             final_output_base_directory,
             skip_processed=SKIP_EXISTING_TEMP_FILES
         )
    print("\nScript finished.")
# --- End of Script ---
