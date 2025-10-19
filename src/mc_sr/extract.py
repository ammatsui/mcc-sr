import os

def aggregate_files_to_single_txt(directory='.'):
    """
    Copies the content of all files in the given directory (excluding the 
    output file itself) into a single text file.
    
    Args:
        directory (str): The path to the directory to process. Defaults to 
                         the current directory ('.').
    """
    # Define the name of the output file
    output_filename = "combined_output.txt"
    
    # Resolve the full path for the output file
    output_path = os.path.join(directory, output_filename)
    
    print(f"Starting aggregation process in directory: {os.path.abspath(directory)}")
    
    # Use a 'with open' block for the output file to ensure it's closed properly
    with open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Iterate over all items in the specified directory
        for item_name in os.listdir(directory):
            
            item_path = os.path.join(directory, item_name)
            
            # 1. Skip the output file to prevent an infinite loop/error
            if item_name == output_filename:
                continue
                
            # 2. Only process files (not subdirectories)
            if os.path.isfile(item_path):
                
                # Create a clear separator header
                separator = f"\n\n==================== START OF FILE: {item_name} ====================\n\n"
                outfile.write(separator)
                
                try:
                    # Read the content of the current file
                    with open(item_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        
                    print(f"  - Successfully added content from: {item_name}")

                except UnicodeDecodeError:
                    # Handle files that are not simple text (e.g., binary files)
                    error_msg = f"\n[!] WARNING: Could not read {item_name} (Possible binary file). Skipping content.\n"
                    outfile.write(error_msg)
                    print(f"  - Skipped (Unicode Error): {item_name}")
                except Exception as e:
                    # Handle other potential errors
                    error_msg = f"\n[!] ERROR reading {item_name}: {e}\n"
                    outfile.write(error_msg)
                    print(f"  - Skipped (Error): {item_name}")

    print(f"\nAggregation complete! All contents saved to: {output_filename}")

# --- Execution ---

# Note: This script will run in the directory where you save and execute it.
# If you want to specify a different directory, change the argument below:
# For example: aggregate_files_to_single_txt('/path/to/my/folder')
aggregate_files_to_single_txt()