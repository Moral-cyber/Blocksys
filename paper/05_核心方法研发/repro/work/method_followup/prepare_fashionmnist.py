"""Download only public official FashionMNIST gzip files; verify official MD5.

No training and no external code execution. Mirror URLs, hashes, gzip header
metadata, and failed retrieval attempts are persisted for review.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import struct
import subprocess
import os

from torchvision.datasets import FashionMNIST
import torchvision

ROOT = Path(__file__).resolve().parent / 'datasets' / 'FashionMNIST' / 'raw'
ROOT.mkdir(parents=True, exist_ok=True)
MIRRORS = [
    'https://fashion-mnist.s3.eu-central-1.amazonaws.com/',
    'https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/',
    'https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/',
    'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
]
MANIFEST_PATH = ROOT.parent / 'download_manifest.json'
PREVIOUS = {}
if MANIFEST_PATH.exists():
    PREVIOUS = {item['file']: item for item in json.loads(MANIFEST_PATH.read_text()).get('files', [])}


def verify(path, name, official_md5):
    raw = path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != official_md5:
        raise ValueError(f'MD5 mismatch: {md5} != {official_md5}')
    with gzip.open(path, 'rb') as f:
        content = f.read()
    magic = struct.unpack('>I', content[:4])[0]
    ndim = magic & 255
    shape = struct.unpack('>' + 'I' * ndim, content[4:4 + ndim * 4])
    if magic >> 8 != 8:
        raise ValueError(f'Unexpected IDX type: {magic}')
    expected_size = ndim * 4 + 4
    for value in shape:
        if value <= 0:
            raise ValueError('Nonpositive shape')
    n_elements = 1
    for value in shape:
        n_elements *= value
    if len(content) != expected_size + n_elements:
        raise ValueError('IDX payload size mismatch')
    return {
        'file': name, 'official_md5': official_md5, 'actual_md5': md5,
        'sha256': hashlib.sha256(raw).hexdigest(), 'compressed_bytes': len(raw),
        'uncompressed_bytes': len(content), 'idx_magic': magic,
        'idx_shape': shape, 'gzip_integrity_verified': True,
        'official_md5_verified': True,
    }


def one(resource):
    name, official_md5 = resource
    destination = ROOT / name
    previous = PREVIOUS.get(name, {})
    attempts = list(previous.get('attempts', []))
    if destination.exists():
        try:
            return {**verify(destination, name, official_md5), 'source': 'previously_existing_verified_file', 'url':previous.get('url'), 'attempts': attempts}
        except Exception as e:
            attempts.append({'source': 'existing_file', 'error': str(e)})
    for mirror in MIRRORS:
        url = mirror + name
        temporary = ROOT / (name + '.partial')
        try:
            call = subprocess.run([
                'curl', '--fail', '--location', '--silent', '--show-error',
                '--connect-timeout', '8', '--max-time', '180', '--retry', '0',
                '--output', str(temporary), url,
            ], capture_output=True, text=True, timeout=185)
            attempt = {'url': url, 'curl_exit_code': call.returncode, 'stderr': call.stderr[-1000:]}
            attempts.append(attempt)
            if call.returncode != 0:
                continue
            record = verify(temporary, name, official_md5)
            os.replace(temporary, destination)
            return {**record, 'url': url, 'attempts': attempts}
        except Exception as e:
            attempts.append({'url': url, 'error': repr(e)})
        finally:
            if temporary.exists():
                temporary.unlink()
    return {'file': name, 'official_md5': official_md5, 'error': 'No mirror yielded an MD5-matching gzip file', 'attempts': attempts}


with ThreadPoolExecutor(max_workers=4) as executor:
    files = list(executor.map(one, FashionMNIST.resources))
manifest = {
    'dataset': 'FashionMNIST', 'checked_utc': datetime.now(timezone.utc).isoformat(),
    'resource_md5_source': 'torchvision.datasets.FashionMNIST.resources',
    'torchvision_version': torchvision.__version__,
    'official_repository': 'https://github.com/zalandoresearch/fashion-mnist',
    'download_only_no_training': True, 'files': files,
    'all_official_md5_verified': all(f.get('official_md5_verified', False) for f in files),
}
MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
raise SystemExit(0 if manifest['all_official_md5_verified'] else 1)
