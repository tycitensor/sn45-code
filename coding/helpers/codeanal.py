import ast

ALLOWED_MODULES = {"math", "json", "re"}

def is_code_safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_MODULES:
                        return False, f"Disallowed module: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module not in ALLOWED_MODULES:
                    return False, f"Disallowed module: {node.module}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.value.id not in ALLOWED_MODULES:
                        return False, f"Disallowed function call: {node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func, ast.Name):
                    if node.func.id not in ALLOWED_MODULES:
                        return False, f"Disallowed function call: {node.func.id}"
        return True, "Code is safe"
    except Exception as e:
        return False, f"Error during parsing: {e}"