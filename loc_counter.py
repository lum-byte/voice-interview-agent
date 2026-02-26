import os

EXTENSIONS = (
    ".py",
    ".json",
    ".yml",
    ".yaml",
    ".md",
    ".toml",
    ".ini"
)

EXCLUDE_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    ".idea",
    "node_modules",
    "dist",
    "build"
}

def count_loc(root_dir="."):
    total_lines = 0
    total_chars = 0
    file_stats = {}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            if file.endswith(EXTENSIONS):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        lines = content.count("\n") + 1 if content else 0
                        chars = len(content)

                        total_lines += lines
                        total_chars += chars

                        file_stats[path] = (lines, chars)

                except Exception:  # noqa
                    pass

    # per-file breakdown
    for file, (lines, chars) in sorted(file_stats.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{lines:6}:{chars:<10} {file}")

    print("\nTOTAL PROJECT STATS")
    print(f"LOC       : {total_lines}")
    print(f"CHARACTERS: {total_chars}")


if __name__ == "__main__":
    count_loc(".")