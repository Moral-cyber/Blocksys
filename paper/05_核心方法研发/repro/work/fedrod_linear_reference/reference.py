"""Plain FedRoD-linear reference components; importing never loads CIFAR data."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = 1
PARAMETERS = 136874
HEAD_PARAMETERS = 650
CAP = 54808
CLIENTS = 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_frozen_sources():
    identities = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
    for relative, expected in identities.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('frozen source changed: ' + relative)
    return identities


def source_identities():
    identities = verify_frozen_sources()
    for name in ('reference.py', 'train.py', 'PLAN.md', 'SOURCE_LOCK.json'):
        path = HERE / name
        identities[str(path.relative_to(ROOT))] = sha(path)
    return identities


def verify_identities(identities):
    for relative, expected in identities.items():
        if sha(ROOT / relative) != expected:
            raise ValueError('source changed during execution: ' + relative)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def model_class():
    """Extract only the unchanged class, bypassing the source's data imports."""
    verify_frozen_sources()
    source = ROOT / 'work/tinyproto_diag/tools/fedavg_update_diagnostic.py'
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SmallCNN']
    if len(nodes) != 1:
        raise ValueError('SmallCNN class missing or ambiguous')
    namespace = {'torch': torch, 'nn': nn, '__name__': __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    # Make deepcopy/pickle metadata deterministic without importing the original file.
    return namespace['SmallCNN']


SmallCNN = model_class()


def new_model(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SmallCNN()
    if vectorize(model).numel() != PARAMETERS:
        raise ValueError('unexpected shared parameter count')
    return model


def new_head():
    # nn.Linear initialization must not disturb the global model RNG stream.
    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(64, 10, bias=True)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def vectorize(model):
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def assign_vector(model, value):
    if value.shape != (PARAMETERS,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('invalid shared vector')
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(value[offset:offset + size].reshape_as(parameter).to(parameter.device))
            offset += size
    if offset != PARAMETERS:
        raise ValueError('shared layout mismatch')


def features(model, x):
    h = model.features(x)
    for layer in list(model.classifier)[:-1]:
        h = layer(h)
    return h


def branch_logits(model, head, x):
    h = features(model, x)
    generic = model.classifier[-1](h)
    residual = head(h.detach())
    return generic, generic.detach() + residual


def balanced_softmax(generic, y, counts):
    """FedRoD Eq.6, gamma=1, raw counts; no pseudocount smoothing."""
    counts = torch.as_tensor(counts, dtype=generic.dtype, device=generic.device)
    if (counts.shape != (10,) or not torch.isfinite(counts).all()
            or (counts < 0).any() or counts.sum() <= 0):
        raise ValueError('BSM requires finite nonnegative nonempty ten-class counts')
    if (y.ndim != 1 or y.numel() != generic.shape[0] or y.numel() == 0
            or (y < 0).any() or (y >= 10).any() or (counts[y] <= 0).any()):
        raise ValueError('BSM targets must belong to observed TRAIN classes')
    return nn.functional.cross_entropy(generic + counts.log(), y)


def losses(model, head, x, y, counts):
    generic, personal = branch_logits(model, head, x)
    return balanced_softmax(generic, y, counts), nn.functional.cross_entropy(personal, y)


def synchronize(backend):
    if backend == 'mps':
        torch.mps.synchronize()


def clock(backend):
    synchronize(backend)
    return time.perf_counter()


def train_local(model, head, dataset, counts, seed, backend):
    """One fixed epoch; one shared forward and combined backward per minibatch."""
    if backend not in ('cpu', 'mps'):
        raise ValueError('unsupported backend')
    if backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    torch.manual_seed(seed)
    if backend == 'mps':
        torch.mps.manual_seed(seed)
    anchor = vectorize(model).clone()
    began = clock(backend)
    local = copy.deepcopy(model).to(backend).train()
    personal = copy.deepcopy(head).to(backend).train()
    counts_device = torch.as_tensor(counts, dtype=torch.float32, device=backend)
    preparation_seconds = clock(backend) - began
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.SGD(list(local.parameters()) + list(personal.parameters()), lr=.03)
    counters = {'steps': 0, 'train_images': 0, 'feature_forward_batches': 0,
        'generic_forward_batches': 0, 'residual_forward_batches': 0,
        'brm_loss_calls': 0, 'erm_loss_calls': 0, 'combined_backward_calls': 0,
        'sgd_steps': 0, 'body_gradient_updates': 0, 'generic_gradient_updates': 0,
        'residual_gradient_updates': 0, 'residual_gradient_scalars': 0,
        'feature_brm_seconds': 0., 'residual_erm_seconds': 0.,
        'joint_backward_step_seconds': 0., 'data_transfer_seconds': 0.,
        'model_head_preparation_seconds': preparation_seconds,
        'independent_residual_backward_seconds': None,
        'backward_time_note': 'combined backward cannot attribute an exclusive head-backward time',
        'train_counts': list(map(int, counts)),
        'bsm': 'CE(generic+log(raw_TRAIN_counts),y), gamma1, no smoothing',
        'local_head_parameters': HEAD_PARAMETERS}
    for x, y in loader:
        began = clock(backend)
        x, y = x.to(backend), y.to(backend)
        counters['data_transfer_seconds'] += clock(backend) - began
        optimizer.zero_grad(set_to_none=True)
        began = clock(backend)
        h = features(local, x)
        generic = local.classifier[-1](h)
        brm = balanced_softmax(generic, y, counts_device)
        counters['feature_brm_seconds'] += clock(backend) - began
        began = clock(backend)
        personal_logits = generic.detach() + personal(h.detach())
        erm = nn.functional.cross_entropy(personal_logits, y)
        objective = brm + erm
        counters['residual_erm_seconds'] += clock(backend) - began
        began = clock(backend)
        objective.backward()
        optimizer.step()
        counters['joint_backward_step_seconds'] += clock(backend) - began
        for name in ('steps', 'feature_forward_batches', 'generic_forward_batches',
                     'residual_forward_batches', 'brm_loss_calls', 'erm_loss_calls',
                     'combined_backward_calls', 'sgd_steps', 'body_gradient_updates',
                     'generic_gradient_updates', 'residual_gradient_updates'):
            counters[name] += 1
        counters['train_images'] += len(y)
        counters['residual_gradient_scalars'] += HEAD_PARAMETERS
    began = clock(backend)
    delta = vectorize(local) - anchor
    head_state = {name: value.detach().cpu().clone() for name, value in personal.state_dict().items()}
    counters['cpu_export_seconds'] = clock(backend) - began
    counters['feature_forward_images'] = counters['train_images']
    counters['generic_forward_images'] = counters['train_images']
    counters['residual_forward_images'] = counters['train_images']
    counters['body_gradient_scalars'] = counters['steps'] * (PARAMETERS - HEAD_PARAMETERS)
    counters['generic_gradient_scalars'] = counters['steps'] * HEAD_PARAMETERS
    if not torch.isfinite(delta).all() or not torch.equal(vectorize(model), anchor):
        raise ValueError('invalid update or changed global anchor')
    if not all(torch.isfinite(v).all() for v in head_state.values()):
        raise ValueError('nonfinite personal head')
    return delta, head_state, counters


def load_codec():
    verify_frozen_sources()
    name = '_fedrod_linear_frozen_codec'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'work/mixed_precision/codec.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def encode_shared(delta, residual):
    for value in (delta, residual):
        if (value.shape != (PARAMETERS,) or value.dtype != torch.float32
                or value.device.type != 'cpu' or not torch.isfinite(value).all()):
            raise ValueError('only the full shared FP32 vector may enter codec/EF')
    candidate = delta + residual
    codec = load_codec()
    began = time.perf_counter()
    encoded = codec.encode(candidate, CAP, mode='uniform3')
    encode_seconds = time.perf_counter() - began
    began = time.perf_counter()
    decoded = codec.decode(encoded.wire)
    decode_seconds = time.perf_counter() - began
    if not torch.equal(decoded, encoded.sent) or len(encoded.wire) > CAP:
        raise ValueError('wire decoder or upload cap failure')
    next_residual = candidate - decoded
    return decoded, next_residual, encoded.wire, {
        'upload_bytes': len(encoded.wire), 'shared_coordinates': decoded.numel(),
        'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
        'ef_max_error': float((decoded + next_residual - candidate).abs().max()),
        'residual_norm': float(next_residual.norm()), 'personal_parameters_uploaded': 0}


def state_layout(model):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in model.state_dict().items()}


def validate_state(state, template, label):
    if not isinstance(state, dict) or set(state) != set(template.state_dict()):
        raise ValueError('incompatible ' + label + ' state keys')
    for name, expected in template.state_dict().items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('incompatible ' + label + ' state tensor')


def validate_config(config, base):
    if not isinstance(config, dict):
        raise ValueError('incompatible saved config')
    unsigned = {k: v for k, v in config.items() if k != 'signature'}
    if config.get('signature') != signature(unsigned) or config.get('base') != base:
        raise ValueError('incompatible saved config signature/source/layout')
    weights = config.get('weights')
    if (not isinstance(weights, list) or len(weights) != CLIENTS
            or any(not isinstance(v, (int, float)) or not 0 < v <= 1 for v in weights)
            or abs(sum(weights) - 1) > 1e-12):
        raise ValueError('incompatible client weights')


def validate_checkpoint(checkpoint, config, model=None):
    if (not isinstance(checkpoint, dict) or checkpoint.get('version') != VERSION
            or checkpoint.get('config') != config or checkpoint.get('signature') != config['signature']):
        raise ValueError('incompatible checkpoint signature/source/config')
    rows = checkpoint.get('rows')
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or len(rows) > config['base']['rounds']
            or [row.get('round') for row in rows] != list(range(1, len(rows) + 1))):
        raise ValueError('incompatible checkpoint round history')
    model = new_model(config['base']['seed']) if model is None else model
    if config['base']['shared_layout'] != state_layout(model):
        raise ValueError('incompatible checkpoint shared layout')
    validate_state(checkpoint.get('model'), model, 'shared')
    heads, residuals = checkpoint.get('heads'), checkpoint.get('residuals')
    if not isinstance(heads, list) or len(heads) != CLIENTS:
        raise ValueError('incompatible checkpoint personal head count')
    template = new_head()
    if config['base']['personal_layout'] != state_layout(template):
        raise ValueError('incompatible checkpoint personal layout')
    for state in heads:
        validate_state(state, template, 'personal head')
    if not isinstance(residuals, list) or len(residuals) != CLIENTS:
        raise ValueError('incompatible checkpoint EF count')
    for value in residuals:
        if (not isinstance(value, torch.Tensor) or value.shape != (PARAMETERS,)
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('incompatible checkpoint EF tensor')
    model.load_state_dict(checkpoint['model'])
    return model


def save_checkpoint(path, config, model, heads, residuals, rows):
    began = time.perf_counter()
    checkpoint = {'version': VERSION, 'signature': config['signature'], 'config': config,
        'model': model.state_dict(), 'heads': [h.state_dict() for h in heads],
        'residuals': residuals, 'rows': rows}
    validate_checkpoint(checkpoint, config, copy.deepcopy(model))
    temporary = Path(path).with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return {'checkpoint_save_seconds': time.perf_counter() - began,
            'checkpoint_file_bytes': Path(path).stat().st_size}


@torch.no_grad()
def evaluate(model, head, dataset, backend):
    """One shared feature pass per batch, report raw G and exact G+P logits."""
    began = clock(backend)
    shared = copy.deepcopy(model).to(backend).eval()
    personal = copy.deepcopy(head).to(backend).eval() if head is not None else None
    sums = {'generic_raw': [0, 0.], 'generic_plus_local_residual': [0, 0.]}
    batches = 0
    for x, y in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
        x, y = x.to(backend), y.to(backend)
        h = features(shared, x)
        generic = shared.classifier[-1](h)
        outputs = {'generic_raw': generic}
        if personal is not None:
            outputs['generic_plus_local_residual'] = generic + personal(h)
        for name, logits in outputs.items():
            sums[name][0] += int((logits.argmax(1) == y).sum())
            sums[name][1] += float(nn.functional.cross_entropy(logits, y, reduction='sum'))
        batches += 1
    names = ('generic_raw',) if personal is None else tuple(sums)
    return {name: {'samples': len(dataset), 'accuracy': sums[name][0] / len(dataset),
                   'ce': sums[name][1] / len(dataset)} for name in names}, {
        'feature_forward_batches': batches, 'generic_forward_batches': batches,
        'residual_forward_batches': batches if personal is not None else 0,
        'feature_forward_images': len(dataset), 'generic_forward_images': len(dataset),
        'residual_forward_images': len(dataset) if personal is not None else 0,
        'seconds': clock(backend) - began}
