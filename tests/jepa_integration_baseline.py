"""Strict source guards against the pinned multiview integration bases.

The static manifest records only the reviewed STEP 5/6 JEPA changes. Rebuild
the expected file from immutable Git objects and compare the ENTIRE source
before returning the baseline for historical source/AST comparisons. No methods
or branches are ignored, including the JEPA additions themselves.
"""
import json
from pathlib import Path
import subprocess


def integration_source(repo, path, current=None):
    repo = Path(repo)
    manifest = json.loads((Path(__file__).parent / 'fixtures/jepa_integration_changes.json').read_text())
    key = f'{repo.name}/{path}'
    if current is None:
        current = (repo / path).read_text()
    if key not in manifest:
        return current
    entry = manifest[key]
    baseline = subprocess.check_output(
        ['git', 'show', f"{entry['revision']}:{path}"], cwd=repo, text=True)
    lines = baseline.splitlines(keepends=True)
    for edit in reversed(entry['edits']):
        lines[edit['start']:edit['end']] = [edit['replacement']]
    assert current == ''.join(lines), f'Unreviewed integration source change: {key}'
    return baseline
