import os
import math
import ijson # Using ijson for potentially better memory efficiency on very large files, though line-by-line reading is often sufficient. Install using: pip install ijson
import json # Fallback for standard json loading if needed, or for writing
import traceback # For printing detailed error stack traces

def count_lines(filepath):
    """Counts the total number of lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f)
        return count
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return 0
    except Exception as e:
        print(f"Error counting lines in {filepath}: {e}")
        return 0

def chunk_jsonl_file(input_filepath, output_dir, num_chunks=4):
    """
    Splits a large JSONL file into a specified number of smaller chunk files.

    Args:
        input_filepath (str): Path to the large input JSONL file.
        output_dir (str): Directory where the chunk files will be saved.
        num_chunks (int): The number of chunks to split the file into.
    """
    print(f"Starting chunking process for: {input_filepath}")
    print(f"Target output directory: {output_dir}")
    print(f"Number of chunks: {num_chunks}")

    # --- Validate Input ---
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found: {input_filepath}")
        return
    if not os.path.isdir(output_dir):
        try:
            print(f"Output directory not found. Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}'. Error: {e}")
            return

    # --- Calculate Chunk Size ---
    print("Counting total lines in the input file (this might take a while)...")
    total_lines = count_lines(input_filepath)
    if total_lines == 0:
        print("Error: Could not count lines or file is empty. Aborting.")
        return
    print(f"Total lines found: {total_lines}")

    lines_per_chunk = math.ceil(total_lines / num_chunks)
    print(f"Calculated lines per chunk: {lines_per_chunk}")

    # --- Process and Write Chunks ---
    current_chunk_num = 1
    lines_in_current_chunk = 0
    output_file = None
    input_filename_base = os.path.splitext(os.path.basename(input_filepath))[0]

    try:
        print(f"Reading input file and writing chunks...")
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                # --- Open new chunk file if needed ---
                if lines_in_current_chunk == 0:
                    if output_file: # Close previous chunk file if open
                        output_file.close()
                        print(f"    Closed chunk {current_chunk_num - 1}")

                    chunk_filename = f"{input_filename_base}_chunk_{current_chunk_num}.jsonl"
                    chunk_filepath = os.path.join(output_dir, chunk_filename)
                    print(f"    Opening chunk {current_chunk_num}: {chunk_filepath}")
                    output_file = open(chunk_filepath, 'w', encoding='utf-8')

                # --- Write line to current chunk ---
                output_file.write(line)
                lines_in_current_chunk += 1

                # --- Check if chunk is full ---
                if lines_in_current_chunk >= lines_per_chunk:
                    lines_in_current_chunk = 0 # Reset counter for next chunk
                    current_chunk_num += 1     # Move to next chunk number

                # Optional: Add progress indicator for very large files
                # if (line_num + 1) % 100000 == 0:
                #     print(f"  Processed {line_num + 1} lines...")

        # --- Close the last chunk file ---
        if output_file and not output_file.closed:
            output_file.close()
            print(f"    Closed chunk {current_chunk_num - 1 if lines_in_current_chunk == 0 else current_chunk_num}")

        print("\nChunking process completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during chunking: {e}")
        traceback.print_exc()
        if output_file and not output_file.closed:
            output_file.close() # Ensure file is closed on error

# --- Configuration ---
input_file = r"C:\apex_project\structured_jsonl_by_category\structured_math_detailed.jsonl"
output_directory = r"C:\apex_project\math_into_pieces"
number_of_chunks = 4
# --------------------

if __name__ == "__main__":
    chunk_jsonl_file(input_file, output_directory, number_of_chunks)
