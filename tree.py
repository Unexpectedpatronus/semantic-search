import os
from pathlib import Path


def print_tree(start_path, prefix="", ignore=None):
    if ignore is None:
        ignore = {
            "__pycache__",
            ".git",
            ".venv",
            "*.pyc",
            ".pytest_cache",
            "tree.py",
        }

    contents = [item for item in os.listdir(start_path) if item not in ignore]
    for i, item in enumerate(sorted(contents)):
        path = Path(start_path) / item
        is_last = i == len(contents) - 1

        print(f"{prefix}{'└── ' if is_last else '├── '}{item}")

        if os.path.isdir(path):
            ext = "    " if is_last else "│   "
            print_tree(path, prefix + ext, ignore)


if __name__ == "__main__":
    print(".")
    print_tree(
        ".",
        ignore={
            "__pycache__",
            ".git",
            ".venv",
            "*.pyc",
            ".pytest_cache",
            "tree.py",
        },
    )
