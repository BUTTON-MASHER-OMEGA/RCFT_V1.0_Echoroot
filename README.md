# RCFT_V1.0_Echoroot
Steve's Findings
RCFT-Book-v1/
├── the_book_v1.0.yaml         # Master scroll
├── # .yamllint
│   ├──extends: default
│   ├──rules:
│   ├──  line-length:
│   ├── max: 120
│   ├── level: warning
│   ├── indentation:
│   ├── spaces: 2
│   ├── indent-sequences: consistent
├── pip install pyyaml
│   ├── chmod +x scripts/generate_indexes.py
│   ├── ./scripts/generate_indexes.py
│   ├── python3 scripts/generate_indexes.py
├── rcft_lib/
│   ├── __init__.py
│   ├── chapter1.py
│   ├── chapter2.py
│   └── visuals.py
├── docs/                      # GitHub Pages source
│   ├── introduction.md
│   ├── glyph_foundations.md
│   ├── calabi_yau_glyphs.md
│   └── ...
├── assets/                    # Images and plots
│   ├── figures/
│   └── diagrams/
├── _config.yml                # Jupyter Book or Docs config
├── _toc.yml                   # Chapter ordering
└── .github/
    └── workflows/
        └── validate_yaml.yml  # CI to lint schema and build docs

git add docs/chapter_summaries.md docs/code_snippets_index.md docs/field_tests_index.md
git commit -m "Regenerate chapter indexes"
git push

