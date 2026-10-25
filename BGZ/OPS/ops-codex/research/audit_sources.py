"""Read-only source inventory and syntax audit; outputs stay in ops-codex."""
from pathlib import Path
import ast, hashlib, json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = ['rcf', 'rcmt', 'projects/Operational_Phase_Space', 'projects/lisi-smg-test']
def manifest():
    result = {}
    for source in SOURCES:
        for p in sorted((ROOT/source).rglob('*')):
            if p.is_file() and '.git' not in p.parts:
                result[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result

if __name__ == '__main__':
    current = manifest()
    baseline = OUT/'source-manifest.json'
    if baseline.exists():
        old = json.loads(baseline.read_text())
        changes = {k: ('added' if k not in old else 'deleted' if k not in current else 'changed')
                   for k in old.keys() | current.keys() if old.get(k) != current.get(k)}
        (OUT/'source-integrity-check.json').write_text(json.dumps(changes, indent=2))
        print('Source changes since baseline:', changes)
    else:
        baseline.write_text(json.dumps(current, indent=2))
        print('Hashed source files:', len(current))
    audit = []
    for p in sorted((ROOT/'projects/Operational_Phase_Space').glob('SIM*.py')):
        text = p.read_text(encoding='utf-8-sig')
        row = {'path': p.relative_to(ROOT).as_posix(), 'lines':len(text.splitlines())}
        try:
            tree = ast.parse(text)
            row['syntax'] = 'valid'
            row['imports'] = sorted({n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)})
        except SyntaxError as exc:
            row.update(syntax='invalid', error_line=exc.lineno, error=exc.msg)
        audit.append(row)
    (OUT/'ops-syntax-audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
