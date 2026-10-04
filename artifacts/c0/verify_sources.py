"""Verify the downloaded, pinned C0 source contract without executing a model."""
import ast
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
from zoneinfo import ZoneInfo

from inspect_checkpoint import MetadataReader, inspect


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


manifest = json.loads(Path('docs/c0_manifest.json').read_text())
assert manifest == json.loads(Path('artifacts/c0/manifest.json').read_text())
repo = manifest['code']['local_path']
head = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
assert head == manifest['code']['commit']
assert not subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'], text=True).strip()
for record in manifest['sources'] + manifest['api_snapshots'] + [manifest['inspection_script']]:
    assert digest(record['path']) == record['sha256'], record['path']
api = json.loads(Path('artifacts/c0/api/clm_model.json').read_text())
assert api['sha'] == manifest['checkpoint']['revision']
file = next(s for s in api['siblings'] if s['rfilename'] == manifest['checkpoint']['file'])
checkpoint = inspect(manifest['checkpoint']['path'])
assert checkpoint['sha256'] == file['lfs']['sha256'] == manifest['checkpoint']['sha256']
assert checkpoint['size_bytes'] == file['lfs']['size'] == manifest['checkpoint']['size_bytes']
assert checkpoint['cfg'] == manifest['checkpoint']['cfg']
for key in ('logit_scale', 'exp_logit_scale', 'inference_scale', 'total_head_parameters'):
    assert checkpoint[key] == manifest['checkpoint'][key], key
assert checkpoint == json.loads(Path('artifacts/c0/checkpoint_inspection.json').read_text())
encoder_config = json.loads(Path('artifacts/c0/encoder/config.json').read_text())
model_config = json.loads(Path('artifacts/c0/model/config.json').read_text())
assert checkpoint['Dh'] == encoder_config['hidden_size'] == model_config['embedding_dim'] == 4096
assert checkpoint['d'] == 512 and model_config['encoder_pooling'] == 'last-token'
assert model_config['base_model'] == checkpoint['cfg']['model'] == manifest['encoder']['repo']
assert not manifest['encoder']['weights_downloaded']
assert len(manifest['encoder']['weight_files']) == 5
for name in ('state_head', 'action_head'):
    assert {k: v['shape'] for k, v in checkpoint['heads'][name]['tensors'].items()} == manifest['checkpoint']['shapes'][name]

# Check the scale rule against the parsed official source, without importing it.
tree = ast.parse(Path(repo, 'src/clm/heads.py').read_text())
clamps = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
          and isinstance(n.func, ast.Attribute) and n.func.attr == 'clamp']
assert any(any(k.arg == 'max' and isinstance(k.value, ast.Constant) and k.value.value == 100.0
               for k in call.keywords) for call in clamps)
assert checkpoint['exp_logit_scale'] > 100.0 and checkpoint['inference_scale'] == 100.0

# Prove the static reader rejects a global outside the three-name allowlist.
try:
    MetadataReader(io.BytesIO(pickle.dumps(eval))).load()
except pickle.UnpicklingError:
    pass
else:
    raise AssertionError('static checkpoint reader accepted an unexpected global')

test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                      text=True, capture_output=True)
Path('artifacts/c0/unit_tests.log').write_text(test.stdout + test.stderr)
assert test.returncode == 0, test.stdout + test.stderr
print(test.stderr.strip())
report = dict(checked_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
    code_commit=head, source_file_hashes='verified', code_worktree='clean',
    checkpoint_sha256=checkpoint['sha256'], official_lfs_hash_and_size='matched',
    checkpoint_cfg_shapes_scale='verified', encoder_config_hidden_size='verified',
    inspected_file_hashes=len(manifest['sources']) + len(manifest['api_snapshots']) + 1,
    disallowed_pickle_global='rejected',
    unit_tests=dict(command='python -m unittest discover -s tests -v', exit_code=test.returncode,
                    log='artifacts/c0/unit_tests.log'),
    model_execution='not performed; source/static checkpoint verification only',
    manifest_sha256=digest('docs/c0_manifest.json'))
Path('artifacts/c0/verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
