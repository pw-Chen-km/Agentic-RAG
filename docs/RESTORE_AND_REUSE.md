# Private data restore and future reuse

The public source branch is not the data backup. The owner's separately delivered
Drive receipts identify the private 48-part `data-code-v1.tar.zst.partNNNN` series,
part hashes, archive hash, original path mapping, contracts and restore manifest.
Do not use an older incomplete full-environment series or infer upload completion
from this source publication. Follow the final private delivery receipt.

On a future disk with adequate free space, fetch all matching parts and controls.
Verify every part and their ordered concatenation before extracting into a new,
empty directory, never over an existing system or repository:

```bash
python3 backup_tools/verify_data_code_parts.py /ABS/parts.json /ABS/parts
```

The archive contains `rootfs/` and `backup-control/`. Stream the verified archive
through zstd/tar, then verify every restored file and JSON record:

```bash
python3 backup_tools/restore_verify.py /ABS/extracted/backup-control/manifest.json /ABS/extracted/rootfs
```

Retain original relative directory relationships or make a deliberately derived
config with a documented new path mapping. Do not mutate archived launches,
contracts, original contract owners, exclusions or cache provenance. Old queue
status/PID files do not authorize restarting any job. Completed Novel data must
not be rerun or regraded automatically; Hotpot remains explicitly deferred.

This data-only backup excludes weights, virtualenvs, runtime binaries and caches.
It keeps environment inventory, model manifest/template, layer hashes/digests and
embedding revision references. A future runtime requires separate trusted-provider
downloads and exact checks. Recorded Qwen manifest digest:
`25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e`.
Third-party artifact availability is not guaranteed. No GPU restoration/inference
was tested by this source-export process. Nothing here downloads models to a Mac.
