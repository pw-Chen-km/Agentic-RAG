"""Read-only local verification of the NEW32MiB data/code series, not old350parts."""
import hashlib
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
directory = Path(sys.argv[2]).resolve()
assert manifest['format'] == 'data-code-only-rootfs-v1'
assert manifest['archive_verified'] and manifest['all_member_hashes_modes_and_json_verified']
assert manifest['chunk_bytes'] == 32 * 1024**2 and not manifest['model_runtime_payload_included']
assert manifest['parts'] and len({p['name'] for p in manifest['parts']}) == len(manifest['parts'])
full = hashlib.sha256(); size = 0
for number, item in enumerate(manifest['parts']):
    assert item['name'] == f'data-code-v1.tar.zst.part{number:04d}'
    assert 0 < item['bytes'] <= manifest['chunk_bytes']
    path = directory / item['name']
    assert path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            full.update(block); h.update(block); size += len(block)
    assert h.hexdigest() == item['sha256'], item['name']
assert size == manifest['archive_bytes'] and full.hexdigest() == manifest['archive_sha256']
print(json.dumps({'series': 'data-code-v1', 'parts_verified': len(manifest['parts']),
    'archive_bytes': size, 'archive_sha256': full.hexdigest(), 'files_modified': False}))
