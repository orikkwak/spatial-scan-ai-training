from pathlib import Path


ROOT = Path("datasets/raw/structured3d")


def print_tree(path: Path, max_depth: int = 4, current_depth: int = 0):
    if current_depth > max_depth:
        return

    if not path.exists():
        print(f"[ERROR] Not found: {path}")
        return

    indent = "  " * current_depth

    for item in sorted(path.iterdir()):
        print(f"{indent}- {item.name}")

        if item.is_dir():
            print_tree(item, max_depth, current_depth + 1)


if __name__ == "__main__":
    print_tree(ROOT, max_depth=5)
