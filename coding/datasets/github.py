import re
import os
import random
import lib2to3
import itertools
import bittensor as bt
from io import StringIO
from datasets import load_dataset, Dataset, load_from_disk
from lib2to3.refactor import RefactoringTool, get_fixers_from_package

from .base import Dataset
from coding.schemas import Context
from coding.helpers.selector import Selector

LANGUAGES = {
    "C++": {
        "keywords": [
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "int",
            "long",
            "register",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
        ],
        "libraries": [
            "iostream",
            "fstream",
            "string",
            "vector",
            "map",
            "set",
            "algorithm",
            "cmath",
            "cstdio",
            "cstdlib",
            "ctime",
            "cstring",
            "cassert",
            "cctype",
            "cerrno",
            "cfloat",
            "ciso646",
            "climits",
            "clocale",
            "cmath",
            "csetjmp",
            "csignal",
            "cstdarg",
            "cstddef",
            "cstdio",
            "cstdlib",
            "cstring",
            "ctime",
            "cwchar",
            "cwctype",
            "complex",
            "deque",
            "exception",
            "fstream",
            "functional",
            "iomanip",
            "ios",
            "iosfwd",
            "iostream",
            "istream",
            "iterator",
            "limits",
            "list",
            "locale",
            "map",
            "memory",
            "new",
            "numeric",
            "ostream",
            "queue",
            "set",
            "sstream",
            "stack",
            "stdexcept",
            "streambuf",
            "string",
            "typerow",
            "utility",
            "valarray",
            "vector",
        ],
        "comments": ["//", "/*", "*/"],
        "multiline_comments": [("/*", "*/")]
    },
    "Dockerfile": {
        "keywords": [
            "from",
            "maintainer",
            "run",
            "cmd",
            "expose",
            "env",
            "add",
            "copy",
            "entrypoint",
            "volume",
            "user",
            "workdir",
            "onbuild",
        ],
        "libraries": [],
        "comments": ["#"],
        "multiline_comments": []
    },
    "HTML": {
        "keywords": [
            "div",
            "span",
            "input",
            "ul",
            "body",
            "tag",
            "html",
            "head",
            "title",
            "meta",
            "link",
            "script",
            "style",
            "a",
            "img",
            "table",
            "label",
        ],
        "libraries": [],
        "comments": ["<!--", "-->"],
        "multiline_comments": [("<!--", "-->")]
    },
    "Java": {
        "keywords": [
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
        ],
        "libraries": [
            "java.awt",
            "java.awt.event",
            "java.io",
            "java.lang",
            "java.math",
            "java.net",
            "java.text",
            "java.util",
            "javax.swing",
        ],
        "comments": ["//", "/*", "*/", "*"],
         "multiline_comments": [("/*", "*/")]
    },
    "JavaScript": {
        "keywords": [
            "abstract",
            "arguments",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "double",
            "else",
            "enum",
            "eval",
            "export",
            "extends",
            "false",
            "final",
            "finally",
            "float",
            "for",
            "function",
            "goto",
            "if",
            "implements",
            "import",
            "in",
            "instanceof",
            "int",
            "interface",
            "let",
            "long",
            "native",
            "module.exports" "new",
            "null",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "true",
            "try",
            "typeof",
            "var",
            "void",
            "volatile",
            "while",
            "with",
            "yield",
        ],
        "libraries": [
            "react",
            "express",
            "mongoose",
            "axios",
            "redux",
            "react-redux",
            "react-router-dom",
            "react-dom",
            "react-scripts",
            "material-ui",
        ],
        "comments": ["//", "/*", "*/"],
        "multiline_comments": [("/*", "*/")]
    },
    "Python": {
        "keywords": [
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        ],
        "libraries": [
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scipy",
            "sklearn",
            "tensorflow",
            "keras",
            "pytorch",
            "django",
            "flask",
            "requests",
            "bs4",
            "selenium",
            "pyautogui",
            "pyperclip",
            "pyinputplus",
            "pillow",
        ],
        "comments": ["#"],
        "multiline_comments": [('"""', '"""'), ("'''", "'''")]
    },
    "SQL": {
        "keywords": [
            "add",
            "all",
            "alter",
            "and",
            "any",
            "as",
            "asc",
            "backup",
            "between",
            "case",
            "check",
            "column",
            "constraint",
            "create",
            "database",
            "default",
            "delete",
            "desc",
            "distinct",
            "drop",
            "exec",
            "exists",
            "foreign",
            "from",
            "full",
            "group",
            "having",
            "in",
            "index",
            "inner",
            "insert",
            "into",
            "is",
            "join",
            "key",
            "left",
            "like",
            "limit",
            "not",
            "null",
            "on",
            "or",
            "order",
            "outer",
            "primary",
            "procedure",
            "right",
            "rownum",
            "select",
            "set",
            "table",
            "top",
            "truncate",
            "union",
            "unique",
            "update",
            "values",
            "view",
            "where",
        ],
        "comments": ["--", "/*", "*/"],
    },
    "Shell": {
        "keywords": [
            "alias",
            "bg",
            "bind",
            "break",
            "builtin",
            "caller",
            "cd",
            "command",
            "compgen",
            "complete",
            "continue",
            "declare",
            "dirs",
            "disown",
            "echo",
            "enable",
            "eval",
            "exec",
            "exit",
            "export",
            "false",
            "fc",
            "fg",
            "getopts",
            "hash",
            "help",
            "history",
            "jobs",
            "kill",
            "let",
            "local",
            "logout",
            "popd",
            "printf",
            "pushd",
            "pwd",
            "read",
            "readonly",
            "return",
            "set",
            "shift",
            "shopt",
            "source",
            "suspend",
            "test",
            "times",
            "trap",
            "true",
            "type",
            "typeset",
            "ulimit",
            "umask",
            "unalias",
            "unset",
            "wait",
        ],
        "comments": ["#"],
        "multiline_comments": [(':\'', '\'')]
    },
}

def convert_to_python3(code: str) -> str:
    """
    Convert Python 2/3 code to Python 3 code.

    Args:
    - code (str): A string containing Python 2/3 code.

    Returns:
    - str: A string containing Python 3 code.
    """
    def replace_print_statement(match):
        return f'print({match.group(1)})'
    
    code = re.sub(r'print (.*)', replace_print_statement, code)
    
    # Replace xrange with range
    code = code.replace('xrange', 'range')
    
    return code

 
def cache_dataset(
    dataset_id: str,
    seed=None,
    cache_dir="~/.cache/huggingface/datasets/github",
    cache_file="github_dataset.arrow"
):
    # Expand user path
    cache_dir = os.path.expanduser(cache_dir)
    cache_path = os.path.join(cache_dir, cache_file)
    shard = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
    # Check if cached dataset exists
    if os.path.exists(f"{cache_path}.{shard}.{seed}"):
        # Load cached dataset
        bt.logging.info(f"Loading cached dataset from {cache_path}")
        dataset = load_from_disk(f"{cache_path}.{shard}.{seed}")
    else:
        # Load, shuffle, and shard the dataset
        bt.logging.info(f"Downloading and processing dataset {dataset_id}")
        
        dataset = load_dataset(
            dataset_id,
            split="train",
            # languages=languages, # TODO: Uncomment if using large git repo
        ).sort("path").shard(8, shard).shuffle(seed=seed) # buffer_size=buffer_size)
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save processed dataset to disk
        dataset.save_to_disk(f"{cache_path}.{shard}.{seed}")
        bt.logging.info(f"Dataset cached at {cache_path}")

    return dataset

def filter_comments(code, language):
    # Filter out multiline comments
    if "multiline_comments" in LANGUAGES[language]:
        for start_tag, end_tag in LANGUAGES[language]['multiline_comments']:
            code = re.sub(rf'{re.escape(start_tag)}.*?{re.escape(end_tag)}', '', code, flags=re.DOTALL)

    # Filter out single-line comments
    lines = []
    for line in code.splitlines():
        if any(
            line.strip().startswith(symbol)
            for symbol in LANGUAGES[language]["comments"]
        ):
            continue
        lines.append(line.lower())

    return "\n".join(lines)

# TODO python_to_python3 function should only be called when python code is used
class GithubDataset(Dataset):
    name = "github"

    def __init__(
        self,
        # dataset_id="codeparrot/github-code",
        dataset_id="angie-chen55/python-github-code",
        seed=None,
        languages=None,
    ):
        
        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed

        if languages is None:
            languages = list(LANGUAGES.keys())
        self.languages = languages

        self.dataset_id = dataset_id

        # self.dataset = cache_dataset(dataset_id=dataset_id, seed=seed)
        self.dataset = load_dataset(
                dataset_id,
                split="train",
                # languages=languages, # TODO: Uncomment if using large git repo
            ).sort("path").shard(8, random.choice([0, 1, 2, 3, 4, 5, 6, 7])).shuffle(seed=seed)
        self.iterset = iter(self.dataset)

    def random(self, min_lines=10, max_lines=3000, selector: Selector = None, include_sibling_docs=False, min_sibling_docs=1, **kwargs):
        return self.get(min_lines, max_lines, selector, include_sibling_docs, min_sibling_docs, **kwargs)
    
    def get(self, min_lines=10, max_lines=3000, selector: Selector = None, include_sibling_docs=False, min_sibling_docs=1, **kwargs):
        row = next(self.iterset)
        if not (min_lines <= len(row["code"].splitlines()) <= max_lines):
            return None

        present_keywords, present_libraries = self.get_special_contents(
            row["code"], row["language"]
        )
        keywords = list(present_keywords) + list(present_libraries)
        code_words = [
            "code",
            "programming",
            "coding",
            "code reference",
            "programming technique",
        ]
        external_links = []
        for bigram in itertools.combinations(keywords, 2):
            words = list(bigram) + [selector(code_words) + row["language"]]
            # shuffle the words e.g. ['react', 'promise', 'code reference'] -> 'code reference promise react'
            external_links.append(" ".join(random.sample(words, len(words))))
        sibling_docs = []
        if include_sibling_docs:
            sibling_docs = [Context(**row) for row in self.search(row["path"])]
            if len(sibling_docs) < min_sibling_docs:
                raise Exception(
                    f"Could not find some code with atleast {min_sibling_docs} sibling documents"
                )
        return {
            "title": row["repo_name"],  # name of the repo
            "topic": row["language"],  # language of the code
            "subtopic": row["path"],
            "content": convert_to_python3(filter_comments(row["code"], row["language"])),
            "internal_links": [row["repo_name"], row["path"], row["language"]],
            "external_links": external_links,
            "source": "GitHub",
            "tags": [row["language"], row["repo_name"], row["path"]],
            "extras": {
                "size": row["size"],
                "license": row["license"],
                "sibling_docs": sibling_docs,
            },
        }

    def search(
        self,
        query,
        column="path",
        min_lines=5,
        max_lines=100,
        selector: Selector = None,
        **kwargs,
    ):
        filtered_dataset = iter(self.dataset.filter(lambda row: row[column] == query))
        return [
            {
                "title": row["repo_name"],  # name of the repo
                "topic": row["language"],  # language of the code
                "subtopic": row["path"],
                "content": convert_to_python3(filter_comments(row["code"], row["language"])),
                "internal_links": [row["repo_name"], row["path"], row["language"]],
                "external_links": [],  # TODO complete
                "source": "GitHub",
                "tags": [row["language"], row["repo_name"], row["path"]],
                "extras": {"size": row["size"], "license": row["license"]},
            }
            for row in filtered_dataset
        ]

    def random(self, min_lines=5, max_lines=100, selector: Selector = None, **kwargs):
        return self.get(min_lines, max_lines, selector)

    def extract_keywords(self, code, language, field):
        matches = set()

        # check which keywords and libraries are present in the code
        for keyword in LANGUAGES[language].get(field, []):
            if re.search(r"\b" + keyword + r"\b", code):
                matches.add(keyword)

        return matches

    def get_special_contents(self, code, language, remove_comments=True):
        if remove_comments:
            code = filter_comments(code, language)

        present_libraries = self.extract_keywords(code, language, "libraries")
        present_keywords = self.extract_keywords(code, language, "keywords")

        return present_keywords, present_libraries
    
    