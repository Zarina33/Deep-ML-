#!/usr/bin/env python3
"""
Auto-generates README.md by scanning solution folders and parsing docstring metadata.

Each .py solution file should have a docstring with the first 3 lines:
    Problem Name
    Difficulty (Easy/Medium/Hard)
    Category (e.g. Linear Algebra, Machine Learning)
"""

import os
import re
import urllib.parse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".git", ".github", "__pycache__", ".claude"}
SKIP_FILES = {"generate_readme.py"}
DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}
DIFFICULTY_BADGES = {
    "Easy": "![Easy](https://img.shields.io/badge/-Easy-brightgreen)",
    "Medium": "![Medium](https://img.shields.io/badge/-Medium-orange)",
    "Hard": "![Hard](https://img.shields.io/badge/-Hard-red)",
}
CATEGORY_ORDER = [
    "Linear Algebra",
    "Machine Learning",
    "Deep Learning",
    "Probability",
    "Statistics",
]


def extract_metadata(filepath):
    """Extract problem name, difficulty, and category from a .py file's docstring."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the first docstring (triple quotes)
    match = re.search(r'(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')', content, re.DOTALL)
    if not match:
        return None

    docstring = match.group(1).strip()
    lines = [line.strip() for line in docstring.split("\n") if line.strip()]

    if len(lines) < 3:
        return None

    name = lines[0]
    difficulty = lines[1]
    category = lines[2]

    if difficulty not in DIFFICULTY_ORDER:
        return None

    return {"name": name, "difficulty": difficulty, "category": category}


def find_solutions():
    """Scan all subdirectories for .py solution files and extract metadata."""
    solutions = []

    for entry in os.listdir(REPO_ROOT):
        entry_path = os.path.join(REPO_ROOT, entry)

        if not os.path.isdir(entry_path):
            continue
        if entry in SKIP_DIRS or entry.startswith("."):
            continue

        # Find .py files in the directory
        py_files = [f for f in os.listdir(entry_path) if f.endswith(".py")]
        if not py_files:
            continue

        py_file = py_files[0]
        filepath = os.path.join(entry_path, py_file)
        metadata = extract_metadata(filepath)

        if metadata is None:
            print(f"  WARNING: Could not parse metadata from {entry}/{py_file}")
            continue

        # Build relative link with URL encoding
        rel_path = f"./{urllib.parse.quote(entry)}/{urllib.parse.quote(py_file)}"
        solutions.append({**metadata, "folder": entry, "file": py_file, "link": rel_path})

    return solutions


def generate_readme(solutions):
    """Generate the full README.md content."""
    # Group by category
    by_category = {}
    for s in solutions:
        by_category.setdefault(s["category"], []).append(s)

    # Sort within each category by difficulty then name
    for cat in by_category:
        by_category[cat].sort(key=lambda s: (DIFFICULTY_ORDER.get(s["difficulty"], 99), s["name"]))

    # Count stats
    total = len(solutions)
    easy = sum(1 for s in solutions if s["difficulty"] == "Easy")
    medium = sum(1 for s in solutions if s["difficulty"] == "Medium")
    hard = sum(1 for s in solutions if s["difficulty"] == "Hard")

    lines = []
    lines.append("# Deep-ML Solutions\n")
    lines.append(
        "My solutions to problems from [deep-ml.com](https://www.deep-ml.com)"
        " — a platform for practicing ML, deep learning, and math fundamentals"
        " through coding challenges.\n"
    )
    lines.append(
        f"![Problems Solved](https://img.shields.io/badge/Solved-{total}-blue) "
        f"![Easy](https://img.shields.io/badge/Easy-{easy}-brightgreen) "
        f"![Medium](https://img.shields.io/badge/Medium-{medium}-orange) "
        f"![Hard](https://img.shields.io/badge/Hard-{hard}-red)\n"
    )
    lines.append("## Problems by Category\n")

    # Order categories
    ordered_cats = [c for c in CATEGORY_ORDER if c in by_category]
    # Add any categories not in the predefined order
    for c in sorted(by_category.keys()):
        if c not in ordered_cats:
            ordered_cats.append(c)

    for cat in ordered_cats:
        problems = by_category[cat]
        lines.append(f"### {cat} ({len(problems)})\n")
        lines.append("| # | Problem | Difficulty | Solution |")
        lines.append("|---|---------|:----------:|----------|")

        for i, s in enumerate(problems, 1):
            badge = DIFFICULTY_BADGES.get(s["difficulty"], s["difficulty"])
            lines.append(f"| {i} | {s['name']} | {badge} | [Python]({s['link']}) |")

        lines.append("")

    return "\n".join(lines)


def main():
    solutions = find_solutions()
    if not solutions:
        print("No solutions found!")
        return

    readme_content = generate_readme(solutions)
    readme_path = os.path.join(REPO_ROOT, "README.md")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"README.md updated: {len(solutions)} problems across {len(set(s['category'] for s in solutions))} categories")


if __name__ == "__main__":
    main()
