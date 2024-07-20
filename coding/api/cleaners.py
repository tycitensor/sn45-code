FIM_PREFIXES = ["<fim_prefix>", "[PREFIX]", "<PRE>", "<|fim_begin|>"]
FIM_SUFFIXES = ["<fim_suffix>", "[SUFFIX]", "<SUF>", "<|fim_end|>"]


def clean_fixes(text):
    for prefix in FIM_PREFIXES:
        text = text.replace(prefix, "")
    for suffix in FIM_SUFFIXES:
        text = text.replace(suffix, "")
    return text