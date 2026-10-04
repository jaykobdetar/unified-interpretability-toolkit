"""Owner machine configuration. Validation never applies limits or starts work."""
from copy import deepcopy
from pathlib import Path

from .common import fields, integer, label, read_json, require

# Byte values are explicit. These reproduce, rather than replace, runtime guards.
GIB = 1024**3
MIB = 1024**2
LOCAL_LIMITS = {
    'cpu_count': 1, 'numeric_workers': 1, 'heavy_jobs': 1,
    'rust_address_space_bytes': 768*MIB, 'rust_available_floor_bytes': 3*GIB,
    'build_address_space_bytes': 2*GIB, 'build_jobs': 1,
    'build_start_available_bytes': 5*GIB,
    'browser_tree_rss_bytes': 768*MIB, 'browser_start_available_bytes': 5*GIB,
    'stop_available_bytes': 13*GIB//4,
    'inference_rss_bytes': 3*GIB//2, 'inference_address_space_bytes': 3*GIB,
    'inference_start_available_bytes': 19*GIB//4,
    'inference_wall_ms': 120000, 'inference_cpu_ms': 90000,
    'prompt_tokens': 128, 'new_tokens': 32, 'client_lease_ms': 15000,
    'inference_queue': 0, 'analytics_address_space_bytes': 768*MIB,
    'analytics_rss_bytes': 768*MIB, 'analytics_start_available_bytes': 15*GIB//4,
    'analytics_wall_ms': 5000, 'analytics_cpu_ms': 4000,
    'disk_reserve_bytes': 25*GIB, 'tile_disk_bytes': 2*GIB, 'tile_files': 1000,
    'pending_headers': 4, 'header_bytes': 8192, 'header_deadline_ms': 500,
    'dispatch_queue': 4, 'numeric_queue': 8, 'write_deadline_ms': 3000,
    'coordinator_body_bytes': 8192, 'upstream_response_bytes': 2*MIB,
}


def validate_config(value, base):
    fields(value, ('version', 'profile', 'bind', 'ports', 'paths'), ('limits',))
    integer(value['version'], 1, 1)
    require(value['profile'] == 'local-v1', 'Only unchanged local-v1 is enabled')
    require(value['bind'] == '127.0.0.1', 'Only existing loopback binding is enabled')
    fields(value['ports'], ('coordinator', 'renderer'))
    ports = [integer(p, 1, 65535) for p in value['ports'].values()]
    require(len(set(ports)) == len(ports), 'Ports must be distinct')
    # Match the current coordinator's reserved viewer ports.
    require(not set(ports) & {8774, 8775, 8785}, 'Port reserved by current viewer')
    fields(value['paths'], ('registry', 'cache'))
    base = Path(base).resolve()
    paths = {}
    for name, raw in value['paths'].items():
        label(raw, 4096)
        path = Path(raw).expanduser()
        paths[name] = str((base / path).resolve())
    require(paths['registry'] != paths['cache'], 'Registry and cache paths must differ')
    supplied = value.get('limits', {})
    fields(supplied, (), LOCAL_LIMITS)
    for key, number in supplied.items():
        integer(number)
        require(number == LOCAL_LIMITS[key], 'Resource changes require staged review')
    return {**deepcopy(value), 'paths': paths, 'limits': deepcopy(LOCAL_LIMITS)}


def load_config(path):
    path = Path(path)
    return validate_config(read_json(path, 16384), path.parent)


def capabilities(config):
    """Sanitized informational projection, not enforcement or authorization."""
    return {'version': 1, 'profile': config['profile'],
            'limits': deepcopy(config['limits']), 'downloads': False,
            'gpu': False, 'resident_activation': False, 'http_routes_active': False}
