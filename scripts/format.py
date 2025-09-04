#!/usr/bin/env python3
"""
Code formatting and quality check script.
Run all code quality tools: black, isort, flake8, and mypy.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Run all code quality checks."""
    project_root = Path(__file__).parent.parent
    backend_path = project_root / "backend"
    main_py_path = project_root / "main.py"
    
    # Paths to check
    paths = [str(backend_path), str(main_py_path)]
    
    print("🔧 Running code quality tools...\n")
    
    all_passed = True
    
    # Format with black
    all_passed &= run_command(
        ["uv", "run", "black"] + paths,
        "Black formatting"
    )
    
    # Sort imports with isort
    all_passed &= run_command(
        ["uv", "run", "isort"] + paths,
        "Import sorting"
    )
    
    # Lint with flake8
    all_passed &= run_command(
        ["uv", "run", "flake8"] + paths,
        "Flake8 linting"
    )
    
    # Type check with mypy
    all_passed &= run_command(
        ["uv", "run", "mypy"] + paths,
        "MyPy type checking"
    )
    
    if all_passed:
        print("\n🎉 All code quality checks passed!")
        return 0
    else:
        print("\n💥 Some code quality checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())