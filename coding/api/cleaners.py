from detect_secrets.core import scan
from detect_secrets.settings import default_settings

FIM_PREFIXES = ["<fim_prefix>", "[PREFIX]", "<PRE>", "<|fim_begin|>"]
FIM_SUFFIXES = ["<fim_suffix>", "[SUFFIX]", "<SUF>", "<|fim_end|>"]


def clean_fixes(text):
    for prefix in FIM_PREFIXES:
        text = text.replace(prefix, "")
    for suffix in FIM_SUFFIXES:
        text = text.replace(suffix, "")
    return text

def remove_secret_lines(multiline_string):
    # Split the input string into individual lines
    lines = multiline_string.split('\n')
    
    # Initialize a list to hold lines without secrets
    clean_lines = []

    # Scan each line for secrets
    with default_settings() as settings:
        settings.disable_plugins(
            'Base64HighEntropyString',
            'HexHighEntropyString'
        )
        for line in lines:
            is_secret = False
            for secret in scan.scan_line(line):
                is_secret = True
                break  # Exit the inner loop if a secret is found
            
            # If no secret is found, add the line to clean_lines
            if not is_secret:
                clean_lines.append(line)
    
    # Join the clean lines back into a single string
    return '\n'.join(clean_lines)