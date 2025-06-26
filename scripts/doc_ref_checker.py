"""
Documentation Reference Checker (Refactored - v3)

A command-line utility to find "dangling" references in documentation. This
version includes intelligent AST parsing for variables/constants and allows for
ignoring known external symbols to prevent false positives.
"""
import argparse
import ast
import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Set

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def extract_references_from_doc(text: str) -> tuple[set[str], set[str]]:
    """Extract file and symbol references from documentation."""
    # Remove code blocks to avoid false positives
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    
    # Ignore common URL patterns and image sources
    url_patterns = [
        r'https?://[^\s]+',  # URLs
        r'//[^\s]+\.(?:png|jpg|jpeg|gif|svg|ico)',  # Image paths
        r'www\.[^\s]+\.(?:com|org|net|io|dev)'  # Website domains
    ]
    for pattern in url_patterns:
        text = re.sub(pattern, '', text)
    
    # Match file references in backticks or code blocks
    file_refs = set()
    # Match Python files in backticks, ignoring URLs and image sources
    for match in re.finditer(r'`([^`\s]+\.py)`', text):
        file_ref = match.group(1)
        # Skip common false positives
        if not any(part in file_ref.lower() for part in ['http', 'www', '//']):
            file_refs.add(file_ref)
    
    # Also find file references in code blocks with paths
    code_blocks = re.findall(r'```(?:python|bash|sh|text|console)[\s\S]*?```', text)
    for block in code_blocks:
        # Find file paths that look like Python files
        for match in re.finditer(r'[\'"]([^\'"\s]+\.py)[\'"]', block):
            file_ref = match.group(1)
            if not any(part in file_ref.lower() for part in ['http', 'www', '//']):
                file_refs.add(file_ref)
    
    # Match symbol references in backticks (excluding file references)
    symbol_refs = set()
    for match in re.finditer(r'`([A-Za-z0-9_]+)`', text):
        symbol = match.group(1)
        # Skip if it looks like a file reference
        if not symbol.endswith('.py') and not any(c in symbol for c in './\\'):
            symbol_refs.add(symbol)
    
    return file_refs, symbol_refs


class SymbolVisitor(ast.NodeVisitor):
    """An AST visitor to find all class, function, and global constant definitions."""
    def __init__(self):
        self.symbols = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.symbols.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.symbols.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.symbols.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Visits an assignment statement (e.g., VAR = 1)."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbols.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visits an annotated assignment (e.g., VAR: int = 1)."""
        if isinstance(node.target, ast.Name):
            self.symbols.add(node.target.id)
        self.generic_visit(node)


def find_symbols_in_codebase(code_dir: Path) -> tuple[set[str], set[str]]:
    """Finds all .py files and their defined symbols using AST parsing."""
    code_files = set()
    code_symbols = set()
    
    # Directories to exclude
    EXCLUDE_DIRS = {
        '.git', '__pycache__', '.pytest_cache', 'venv', 'env',
        'build', 'dist', '*.egg-info', 'node_modules', 'docs'
    }
    
    # File patterns to exclude
    EXCLUDE_FILES = {
        '*_test.py', 'test_*.py', 'conftest.py', 'setup.py',
        '*example*.py', '*demo*.py'
    }
    
    logger.info(f"Scanning codebase at: '{code_dir}'")
    
    for root, dirs, files in os.walk(code_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            # Skip non-Python files and excluded file patterns
            if not file.endswith('.py'):
                continue
            if any(fnmatch.fnmatch(file, pattern) for pattern in EXCLUDE_FILES):
                continue
                
            filepath = Path(root) / file
            try:
                relative_path = filepath.relative_to(code_dir).as_posix()
                code_files.add(relative_path)

                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=str(filepath))
                    visitor = SymbolVisitor()
                    visitor.visit(tree)
                    code_symbols.update(visitor.symbols)
            except Exception as e:
                logger.warning(f"Could not parse {filepath}: {e}")

    logger.info(f"Found {len(code_files)} source files and {len(code_symbols)} unique symbols.")
    return code_files, code_symbols


def validate_references(doc_path: Path, code_dir: Path, ignored_symbols: Set[str]):
    """Main function to validate documentation references against the codebase."""
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_content = f.read()
    except FileNotFoundError:
        logger.critical(f"Documentation file not found: {doc_path}")
        return

    doc_files, doc_symbols = extract_references_from_doc(doc_content)
    code_files, code_symbols = find_symbols_in_codebase(code_dir)

    code_basenames = {Path(p).name for p in code_files}
    missing_files = {f for f in doc_files if Path(f).name not in code_basenames}
    
    # Check symbol references, ignoring common false positives
    missing_symbols = []
    for s in sorted(doc_symbols):
        # Skip if it's an ignored symbol or a built-in
        if (s.upper() == s and len(s) > 2) or s.lower() in ignored_symbols:
            continue
            
        # Check if symbol exists in codebase (exact match or as part of a qualified name)
        if s not in code_symbols and not any(s in symbol for symbol in code_symbols):
            # Additional check for common patterns that might be false positives
            if not any(pat in s.lower() for pat in ['example', 'test', 'mock', 'dummy']):
                missing_symbols.append(s)

    # --- Reporting ---
    print("\n--- Documentation Reference Check Report ---")
    print(f"📄 Document: {doc_path}")
    print(f"💻 Codebase: {code_dir.resolve()}")
    if ignored_symbols:
        print(f"🙈 Ignoring Symbols: {sorted(list(ignored_symbols))}")
    print("-" * 40)

    has_errors = False
    if missing_files:
        has_errors = True
        print(f"🚫 Found {len(missing_files)} Potentially Missing File References:")
        for f in sorted(list(missing_files)):
            print(f"  - {f}")
    else:
        print("✅ All file references appear to be valid.")

    if missing_symbols:
        print("\n🚫 Potentially missing class/function references:")
        for s in missing_symbols:
            # Try to find similar symbols that might be a match
            similar = [sym for sym in code_symbols if s.lower() in sym.lower()]
            if similar:
                print(f"  - {s} (similar to: {', '.join(similar[:3])}{'...' if len(similar) > 3 else ''})")
            else:
                print(f"  - {s}")
    else:
        print("\n✅ All class/function references appear to be valid.")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"  - Files referenced: {len(doc_files)}")
    print(f"  - Symbols checked: {len(doc_symbols)}")
    print(f"  - Missing symbols found: {len(missing_symbols)}")
    
    print("-" * 40)
    if not has_errors:
        print("🎉 Success! No broken references found.")
    else:
        print("⚠️ Issues found. Please review the missing references above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check documentation for broken code references using AST parsing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--doc", type=Path, required=True,
        help="Path to the documentation file to check (e.g., ARCHITECTURE.md)."
    )
    parser.add_argument(
        "--code", type=Path, default=Path("."),
        help="Root directory of the codebase to scan. It will automatically skip 'venv' and '.git'."
    )
    # REFACTORED: Added argument to ignore known external symbols
    parser.add_argument(
        "--ignore-symbols", type=str, default="",
        help="A comma-separated list of symbols to ignore (e.g., 'black,isort')."
    )

    args = parser.parse_args()
    ignored_symbols_set = set(args.ignore_symbols.split(',')) if args.ignore_symbols else set()
    
    validate_references(args.doc, args.code, ignored_symbols_set)
