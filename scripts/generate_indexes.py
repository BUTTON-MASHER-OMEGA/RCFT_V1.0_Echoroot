#!/usr/bin/env python3
import yaml
from pathlib import Path

# Paths
BOOK_YAML = Path("the_book_v1.0.yaml")
OUT_DIR   = Path("docs")
SUM_FILE  = OUT_DIR / "chapter_summaries.md"
CODE_FILE = OUT_DIR / "code_snippets_index.md"
FT_FILE   = OUT_DIR / "field_tests_index.md"

# Load master scroll
book = yaml.safe_load(BOOK_YAML)
chapters = book.get("chapters", [])

# Helpers
def md_header(level, text): return f"{'#'*level} {text}\n\n"

# 2.1 Chapter Summaries
with open(SUM_FILE, "w") as out:
    out.write(md_header(1, "The Book v1.0 — Chapter Summaries"))
    for chap in chapters:
        num   = chap["number"]
        title = chap.get("title","(no title)")
        desc  = chap.get("description","")
        out.write(md_header(2, f"Chapter {num}: {title}"))
        out.write(desc.strip() + "\n\n")

# 2.2 Code Snippets Index
with open(CODE_FILE, "w") as out:
    out.write(md_header(1, "Code Snippets Index"))
    for chap in chapters:
        cs_list = chap.get("code_snippets", [])
        if not cs_list: continue
        out.write(md_header(2, f"Chapter {chap['number']}"))
        for cs in cs_list:
            name = cs.get("name")
            desc = cs.get("description","")
            file = cs.get("file","")
            out.write(f"- **{name}** (`{file}`): {desc}\n")
        out.write("\n")

# 2.3 Field Tests Index
with open(FT_FILE, "w") as out:
    out.write(md_header(1, "Field Tests Index"))
    for chap in chapters:
        ft_list = chap.get("field_tests", [])
        if not ft_list: continue
        out.write(md_header(2, f"Chapter {chap['number']}"))
        for ft in ft_list:
            name = ft.get("name")
            desc = ft.get("description","")
            out.write(f"- **{name}**: {desc}\n")
        out.write("\n")

print("✅ Indexes generated in docs/")

chmod +x scripts/generate_indexes.py
