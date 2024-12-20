import ast
from typing import List

def verify_code_usage(code: str, allowed_modules: List[str]) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in allowed_modules:
                        return False, f"Disallowed module: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module not in allowed_modules:
                    return False, f"Disallowed module: {node.module}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Skip checking function calls if they come from an allowed module
                    pass
                elif isinstance(node.func, ast.Name):
                    if node.func.id == 'eval' or node.func.id == 'exec':
                        return False, f"Dangerous built-in function call: {node.func.id}"
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id in ['eval', 'exec']:
                        return False, f"Dangerous built-in function call: {node.value.func.id}"
        return True, "Code is safe"
    except Exception as e:
        return False, f"Error during parsing: {e}"