import ast
from typing import List, Dict

def verify_code_usage(code: str, allowed_modules: List[str], allowed_imports: Dict[str, List[str]]) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        imported_modules = set()
        imported_names = {}  # Track what names were imported from each module
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Only block import if module is in allowed_imports but used without restrictions
                    if alias.name in allowed_imports and not allowed_imports[alias.name]:
                        return False, f"Disallowed unrestricted use of module: {alias.name}"
                    if alias.name not in allowed_modules and alias.name not in allowed_imports:
                        return False, f"Disallowed module: {alias.name}"
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module not in allowed_modules and node.module not in allowed_imports:
                    return False, f"Disallowed module: {node.module}"
                # Track imported names from restricted modules
                if node.module in allowed_imports:
                    imported_names[node.module] = set()
                    for alias in node.names:
                        if alias.name not in allowed_imports[node.module]:
                            return False, f"Disallowed import {alias.name} from module {node.module}"
                        imported_names[node.module].add(alias.name)
                imported_modules.add(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check if attribute access like os.getenv is allowed
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        # Only check restricted functions if module was imported and has restrictions
                        if module_name in imported_modules and module_name in allowed_imports:
                            if node.func.attr not in allowed_imports[module_name]:
                                return False, f"Disallowed function {module_name}.{node.func.attr}"
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