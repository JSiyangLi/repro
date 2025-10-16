import sys  # Added for command-line argument handling


def merge_txt_files(file1_path, file2_path, output_path):
    """
    Merges two text files by adding lines from file2 to the end of file1.
    Creates a new output file with the combined content.

    Args:
        file1_path (str): Path to first input file
        file2_path (str): Path to second input file
        output_path (str): Path for output merged file
    """
    try:
        # Read contents of both files
        with open(file1_path, 'r') as f1:
            content1 = f1.readlines()

        with open(file2_path, 'r') as f2:
            content2 = f2.readlines()

        # Combine contents (file1 first, then file2)
        merged_content = content1 + content2

        # Write to output file
        with open(output_path, 'w') as out_file:
            out_file.writelines(merged_content)

        print(f"Successfully merged files. Output saved to {output_path}")
        print(f"Lines from {file1_path}: {len(content1)}")
        print(f"Lines from {file2_path}: {len(content2)}")
        print(f"Total lines in output: {len(merged_content)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


