"""Content-derived identities and private immutable response policy; no I/O."""
import hashlib
import math
from copy import deepcopy

from .common import canonical, digest, fields, identity, integer, label, require


def binding(value):
    """Task15 SOURCE-BINDING-V2, including canonical vector/matrix slices."""
    fields(value, ('version', 'model_identity', 'source_identity', 'tensor', 'name',
                   'dtype', 'shape', 'rows', 'cols', 'slice'))
    integer(value['version'], 2, 2)
    digest(value['model_identity'])
    digest(value['source_identity'])
    integer(value['tensor'], 0, 9999)
    label(value['name'], 512)
    require(value['dtype'] in ('BF16', 'F16', 'F32'), 'Numeric artifact dtype unavailable')
    shape = value['shape']
    require(type(shape) is list and 1 <= len(shape) <= 32, 'Invalid native shape')
    count = 1
    for dimension in shape:
        count *= integer(dimension, 1, 2**63-1)
        require(count <= 2**63-1, 'Native shape product overflow')
    integer(value['rows'], 1, 200000)
    integer(value['cols'], 1, 200000)
    require(value['rows'] == (shape[-2] if len(shape) >= 2 else 1)
            and value['cols'] == shape[-1], 'Display dimensions differ from native trailing axes')
    fields(value['slice'], ('leading_indices', 'display_axes'))
    selected = value['slice']['leading_indices']
    axes = value['slice']['display_axes']
    require(type(selected) is list and len(selected) == max(0, len(shape)-2),
            'Every leading slice index is required')
    require(type(axes) is list and all(type(axis) is int for axis in axes)
            and axes == ([0] if len(shape) == 1 else [len(shape)-2, len(shape)-1]),
            'Display axes must be native trailing axes')
    for selected_index, dimension in zip(selected, shape):
        integer(selected_index, 0, dimension-1)
    return deepcopy(value)


def derivation_identity(value):
    fields(value, ('content_digest', 'binding', 'algorithm', 'calibration_digest',
                   'rule', 'parameters', 'level', 'x', 'y', 'control', 'encoding'))
    digest(value['content_digest'])
    binding(value['binding'])
    digest(value['calibration_digest'])
    for key in ('algorithm', 'rule', 'encoding'):
        label(value[key], 128)
    for key in ('level', 'x', 'y'):
        integer(value[key], 0, 2**31-1)
    parameters = value['parameters']
    require(type(parameters) is dict and len(parameters) <= 16, 'Invalid rule parameters')
    for key, parameter in parameters.items():
        label(key, 64)
        require(type(parameter) in (int, float, str, bool), 'Invalid rule parameter type')
        if type(parameter) is float:
            require(math.isfinite(parameter), 'Nonfinite rule parameter')
        elif type(parameter) is int:
            require(abs(parameter) <= 2**63-1, 'Rule parameter outside range')
        elif type(parameter) is str:
            label(parameter, 128)
    if value['control'] is not None:
        fields(value['control'], ('algorithm', 'seed', 'permutation_digest'))
        label(value['control']['algorithm'], 128)
        integer(value['control']['seed'], 0, 2**64-1)
        digest(value['control']['permutation_digest'])
    require(len(canonical(value)) <= 8192, 'Derivation descriptor too large')
    portable = deepcopy(value)
    # Local fingerprint identities are validated before derivation but are not
    # portable content identity. No root/inode-dependent bits enter this key.
    del portable['binding']['source_identity']
    del portable['binding']['model_identity']
    return identity('weight-atlas-derived-v1', portable)


def no_store_headers():
    return {'Cache-Control': 'no-store'}


def immutable_headers(requested_digest, payload, mime, artifact_kind):
    """Call only for a completed artifact after route ownership/access checks.

    No header helper authenticates a request or validates renderer correctness.
    Digest URLs cannot be used for live/partial/private inference output.
    """
    digest(requested_digest)
    allowed = {'tile': 'image/png', 'overview': 'image/png',
               'static_script': 'text/javascript', 'static_style': 'text/css'}
    require(artifact_kind in allowed and allowed[artifact_kind] == mime,
            'Artifact class cannot be immutable')
    require(type(payload) is bytes and len(payload) <= 2*1024**2,
            'Immutable payload exceeds existing response bound')
    require(hashlib.sha256(payload).hexdigest() == requested_digest,
            'Artifact digest does not match requested identity')
    return {'Cache-Control': 'private, max-age=31536000, immutable',
            'ETag': '"'+requested_digest+'"', 'Content-Type': mime,
            'Content-Length': str(len(payload)), 'X-Content-Type-Options': 'nosniff'}
