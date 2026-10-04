"""Inspect this pinned CLM torch.save archive without importing torch or running a model.

Only three pickle globals are accepted; tensor reconstruction returns metadata.
No checkpoint-provided callable is executed and no tensor storage is allocated.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import struct
import zipfile


def tensor_metadata(storage, offset, size, stride, requires_grad, hooks):
    return dict(storage=storage, storage_offset=offset, shape=list(size),
                stride=list(stride), requires_grad=requires_grad)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('collections', 'OrderedDict'): OrderedDict,
                   ('torch._utils', '_rebuild_tensor_v2'): tensor_metadata,
                   ('torch', 'FloatStorage'): 'float32'}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'unsupported global {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, identifier):
        if len(identifier) != 5 or identifier[0] != 'storage' or identifier[1] != 'float32':
            raise pickle.UnpicklingError('unsupported storage descriptor')
        _, dtype, key, location, count = identifier
        return dict(dtype=dtype, key=key, recorded_location=location, count=count)


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        pickle_path = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = pickle_path.rsplit('/', 1)[0]
        checkpoint = MetadataReader(io.BytesIO(archive.read(pickle_path))).load()
        byteorder = archive.read(f'{prefix}/byteorder').decode()
        scale = checkpoint['logit_scale']
        if byteorder != 'little' or scale['shape'] != [] or scale['storage_offset'] != 0:
            raise ValueError('expected little-endian scalar logit_scale')
        raw = archive.read(f"{prefix}/data/{scale['storage']['key']}")
        logit_scale, = struct.unpack('<f', raw)
        if not math.isfinite(logit_scale):
            raise ValueError('logit_scale must be finite')
        cfg = checkpoint['cfg']
        heads = {}
        for key in ('state_head', 'action_head'):
            tensors = checkpoint[key]
            expected = {
                'inp.weight': [cfg['width'], cfg['hidden_size']], 'inp.bias': [cfg['width']],
                'hidden.0.weight': [cfg['width'], cfg['width']], 'hidden.0.bias': [cfg['width']],
                'norms.0.weight': [cfg['width']], 'norms.0.bias': [cfg['width']],
                'out.weight': [cfg['projection_dim'], cfg['width']], 'out.bias': [cfg['projection_dim']]}
            if {name: value['shape'] for name, value in tensors.items()} != expected:
                raise ValueError(f'{key} shapes do not match cfg and official depth-3 head')
            for tensor in tensors.values():
                storage = tensor['storage']
                if archive.getinfo(f"{prefix}/data/{storage['key']}").file_size != storage['count'] * 4:
                    raise ValueError('storage size mismatch')
            heads[key] = dict(tensors=tensors,
                             parameter_count=sum(math.prod(v['shape']) for v in tensors.values()))
        if checkpoint['hidden_size'] != cfg['hidden_size'] or checkpoint['projection_dim'] != cfg['projection_dim']:
            raise ValueError('top-level dimensions differ from cfg')
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(checkpoint_path=str(path), size_bytes=Path(path).stat().st_size, sha256=digest,
        checkpoint_keys=list(checkpoint), cfg=cfg, Dh=cfg['hidden_size'], d=cfg['projection_dim'],
        heads=heads, total_head_parameters=sum(v['parameter_count'] for v in heads.values()),
        logit_scale=logit_scale, exp_logit_scale=math.exp(logit_scale),
        inference_scale=min(math.exp(logit_scale), 100.0),
        inspection='static ZIP and allowlisted pickle metadata only; no torch/model execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/c0/model/CLM_v0.1-8B.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/c0/checkpoint_inspection.json'))
    args = parser.parse_args()
    result = inspect(args.checkpoint)
    api = json.loads(Path('artifacts/c0/api/clm_model.json').read_text())
    expected = next(s for s in api['siblings'] if s['rfilename'] == args.checkpoint.name)
    if result['sha256'] != expected['lfs']['sha256'] or result['size_bytes'] != expected['lfs']['size']:
        raise ValueError('checkpoint does not match pinned official LFS hash/size')
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('sha256', 'cfg', 'Dh', 'd', 'total_head_parameters',
                                           'logit_scale', 'exp_logit_scale', 'inference_scale')}))
