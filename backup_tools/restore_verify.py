"""Offline read-only verification of an extracted backup rootfs."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
rootfs = Path(sys.argv[2]).resolve()
assert rootfs.is_dir()
checked = 0
for item in manifest['entries']:
    relative = Path(item['original'].lstrip('/'))
    assert '..' not in relative.parts
    path = rootfs / relative
    if item['kind'] == 'symlink':
        assert path.is_symlink() and os.readlink(path) == item['target'], item['original']
        target = rootfs / item['resolved'].lstrip('/')
        assert target.exists() or target.is_symlink(), item['resolved']
    else:
        assert path.is_file() and path.stat().st_size == item['bytes'], item['original']
        assert stat.S_IMODE(path.stat().st_mode) == item['mode'], item['original']
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == item['sha256'], item['original']
        if item.get('json_validation_not_applicable'):
            assert '.no_exist' in path.parts and item['bytes'] == 0
        elif path.suffix == '.json':
            json.loads(path.read_text())
        elif path.suffix == '.jsonl':
            with path.open() as stream:
                for line in stream:
                    if line.strip():
                        json.loads(line)
    checked += 1
print(json.dumps({'entries_verified': checked, 'file_hashes_and_symlinks_verified': True,
                  'json_documents_verified': True, 'inference_tested': False,
                  'no_files_modified': True}, indent=2))
