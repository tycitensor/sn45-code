from difflib import unified_diff
from typing import Dict
from swebase import Patch, Edit

def create_patch(original_files: Dict[str, str], edited_files: Dict[str, str]) -> Patch:
    """
    Create a Patch object by comparing original and edited file contents
    
    Args:
        original_files (Dict[str, str]): Dictionary mapping filenames to original file contents
        edited_files (Dict[str, str]): Dictionary mapping filenames to edited file contents
        
    Returns:
        Patch: Patch object containing the edits
    """
    edits = []
    
    # Process each edited file
    for filename in edited_files:
        if filename not in original_files:
            continue
            
        # Split files into lines
        original_lines = original_files[filename].splitlines()
        edited_lines = edited_files[filename].splitlines()
        
        # Generate diff
        diff = list(unified_diff(
            original_lines,
            edited_lines,
            lineterm='',
        ))

        print(f"Diff for {filename}:")
        for d in diff:
            print(d)
        
        # Parse diff to create Edit objects
        line_num = 0
        j = 0
        while j < len(diff):
            line = diff[j]
            if line.startswith('@@'):
                # Parse the line numbers from the @@ line
                # Format is @@ -start,length +start,length @@
                parts = line.split(' ')
                if len(parts) >= 2:
                    old_range = parts[1]  # Get the -start,length part
                    line_num = int(old_range.split(',')[0][1:])  # Extract start number after '-'
            elif line.startswith('- '):
                old_content = line[2:]
                # Check if next line is an addition (modification)
                if j + 1 < len(diff) and diff[j + 1].startswith('+ '):
                    new_content = diff[j + 1][2:]
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=line_num,
                            line_content=old_content,
                            new_line_content=new_content
                        )
                    )
                    j += 1  # Skip the next line since we handled it
                line_num += 1
            elif line.startswith('+ '):
                # This is a new line being added
                if line_num == 0:  # Handle additions at start of file
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=0,
                            line_content="",
                            new_line_content=line[2:]
                        )
                    )
                else:  # Handle additions elsewhere
                    edits.append(
                        Edit(
                            file_name=filename,
                            line_number=line_num,
                            line_content="",
                            new_line_content=line[2:]
                        )
                    )
            elif not line.startswith('@@'):
                line_num += 1
            j += 1
    return Patch(edits=edits)