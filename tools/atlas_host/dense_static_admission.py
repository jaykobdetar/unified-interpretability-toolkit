"""Fixed pinned Smol static policy. Trusted owner bootstrap only; no model execution."""
from dataclasses import dataclass
import hashlib
import fcntl
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys

from .common import canonical, digest, fields, integer, require
from .profile_os import strict_json
from .registry import fingerprint, validate_manifest
from .static_models import _current
from .validation_policy import package_root, source_names

REPOSITORY='HuggingFaceTB/SmolLM2-135M'
REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
CONFIG_SHA='1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843'
WEIGHTS_SHA='80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
PRIOR_TINY_SHA='645285c4c06a3525ed02e85484e1ef08f566b614bd40ac7bb9af1b7767d74ccd'
QUALIFIED_READER_SHA=None  # Combined reader qualification is pending.
# Exact combined native inputs; matching bytes do not qualify a reader.
NATIVE_SOURCE_PINS={'Cargo.lock': '36e31907d27773a3781dd6f012428204abe7da611625b20344e0572e67e06daf',
 'Cargo.toml': '179b96b488b8bda8dce3632708de2acdf1150641a0316979c87afaf6fe7c0004',
 'src/comparison.rs': 'd666ebb1441690429d6a81e6add0885dc8e8d86cec01b815939af45c95f3223c',
 'src/comparison_http.rs': '27866e03b02ff26aa4019ec5c68e1f3775386256a668ae2a7c280d93f7f2b098',
 'src/hosted_renderer.rs': '49cd8c6144c46c7286758b0947b2d8a41ca447f12152585c41fefb28780ef6dd',
 'src/lib.rs': '27f0029c1568d7d7f4add2f3b9191f21a0bc8c9661eb676bc02f38f221f80b54',
 'src/main.rs': '43dd07ce6abe2ad5da970d69697f03a38a020f911bb11dcca375d0dfb6fcea45',
 'src/profile_worker.rs': 'd9c9b07928e8b2b070568d71af79285973d4aa1380963e8ee6c19b1a5fdd38c6',
 'src/render.rs': '6552ecde66c90b88cf9e340f4ab89fed51bdfa94fbb30041929c891f131d2544',
 'src/resources.rs': '83f8f46daa2d45361e8b6bc7e7d916fd587ea16e6018b3eac51f057afb074aa8',
 'src/server.rs': 'fb0d9f433b65ab2eb74a9f7b7037523b01ab684d7e1e49179a02bbe6f17d2456',
 'src/server/reuse.rs': '999877634e9a8b119991ed49bbbb3c2bfd50cf4b3d14421f3c6f5296473e5cfa',
 'src/server/reuse_transport.rs': '336750cf1c4de3ca4d714c93e4597fc4fc42ebc7b8496b4278bba6220708f73f',
 'src/slice.rs': 'bd4e64f2161f9af1a696d179770a200fed940cc1bd356fda3bb5b68bb0b9d815',
 'src/source.rs': '13ed5932eeb2a54387f358fbf4f50ca31b9ba7a9b361237005eb6c059ee524a7',
 'src/state.rs': '948d840cf9f2b71a3293e63e2ec6d4000386e83b9a7e98e4a1586a4d162b55ff',
 'src/strength.rs': 'ac76cb5c4018fd088b30e61f46179a1a10673a2e7b9fd04db6593f16b0f2fc8e',
 'src/strength/snapshot.rs': '4f9387541756d5e9a3ba48aa47b6332e383814dc1aa202df663b5157ccb2475c'}
_SEAL=object()

SCHEMAS={'cacheProbe': {'additionalProperties': False,
                'properties': {'atomic_publish': {'additionalProperties': False,
                                                  'properties': {'after': {'$ref': '#/$defs/fingerprint'},
                                                                 'before': {'$ref': '#/$defs/fingerprint'},
                                                                 'expected_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                                                     'type': 'string'},
                                                                 'readback_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                                                     'type': 'string'},
                                                                 'temporary': {'$ref': '#/$defs/fingerprint'},
                                                                 'temporary_remaining': {'const': False}},
                                                  'required': ['before',
                                                               'temporary',
                                                               'after',
                                                               'expected_sha256',
                                                               'readback_sha256',
                                                               'temporary_remaining'],
                                                  'type': 'object'},
                               'exclusive_lock_release': {'additionalProperties': False,
                                                          'properties': {'acquired': {'const': True},
                                                                         'api': {'const': 'flock_LOCK_EX'},
                                                                         'reacquired': {'const': True},
                                                                         'released': {'const': True},
                                                                         'released_again': {'const': True}},
                                                          'required': ['api',
                                                                       'acquired',
                                                                       'released',
                                                                       'reacquired',
                                                                       'released_again'],
                                                          'type': 'object'},
                               'open_identity': {'$ref': '#/$defs/openIdentity'},
                               'ordinary_replace': {'$ref': '#/$defs/ordinaryReplace'},
                               'probe_root': {'$ref': '#/$defs/root'},
                               'same_length_write': {'$ref': '#/$defs/sameLengthWrite'}},
                'required': ['probe_root',
                             'open_identity',
                             'same_length_write',
                             'ordinary_replace',
                             'exclusive_lock_release',
                             'atomic_publish'],
                'type': 'object'},
 'file': {'additionalProperties': False,
          'properties': {'bytes': {'maximum': 524288, 'minimum': 0, 'type': 'integer'},
                         'path': {'maxLength': 180, 'minLength': 1, 'type': 'string'},
                         'sha256': {'pattern': '^[0-9a-f]{64}$', 'type': 'string'}},
          'required': ['path', 'bytes', 'sha256'],
          'type': 'object'},
 'filesystemReceipt': {'additionalProperties': False,
                       'properties': {'cache': {'$ref': '#/$defs/root'},
                                      'contract': {'const': 'local_linux_regular_files_v1'},
                                      'max_inert_file_bytes': {'const': 65536},
                                      'model_files_modified': {'const': False},
                                      'model_payload_read': {'const': False},
                                      'numeric_model_work': {'const': False},
                                      'observations': {'additionalProperties': False,
                                                       'properties': {'cache': {'$ref': '#/$defs/cacheProbe'},
                                                                      'source': {'$ref': '#/$defs/sourceProbe'}},
                                                       'required': ['source', 'cache'],
                                                       'type': 'object'},
                                      'platform': {'additionalProperties': False,
                                                   'properties': {'kernel_release': {'maxLength': 128,
                                                                                     'minLength': 1,
                                                                                     'type': 'string'},
                                                                  'native_arch': {'maxLength': 64,
                                                                                  'minLength': 1,
                                                                                  'type': 'string'},
                                                                  'python_implementation': {'const': 'cpython'},
                                                                  'python_version': {'maxLength': 32,
                                                                                     'pattern': '^3\\.[0-9]+\\.[0-9]+$',
                                                                                     'type': 'string'},
                                                                  'required_apis': {'additionalProperties': False,
                                                                                    'properties': {'flock_lock_ex': {'const': True},
                                                                                                   'fstat': {'const': True},
                                                                                                   'o_nofollow': {'const': True},
                                                                                                   'pread': {'const': True},
                                                                                                   'stat_ns': {'const': True}},
                                                                                    'required': ['stat_ns',
                                                                                                 'fstat',
                                                                                                 'pread',
                                                                                                 'o_nofollow',
                                                                                                 'flock_lock_ex'],
                                                                                    'type': 'object'},
                                                                  'system': {'const': 'linux'}},
                                                   'required': ['system',
                                                                'kernel_release',
                                                                'native_arch',
                                                                'python_implementation',
                                                                'python_version',
                                                                'required_apis'],
                                                   'type': 'object'},
                                      'scope': {'const': 'ordinary_inert_files_only'},
                                      'source': {'$ref': '#/$defs/root'},
                                      'version': {'const': 1}},
                       'required': ['version',
                                    'contract',
                                    'scope',
                                    'source',
                                    'cache',
                                    'platform',
                                    'observations',
                                    'max_inert_file_bytes',
                                    'model_files_modified',
                                    'model_payload_read',
                                    'numeric_model_work'],
                       'type': 'object'},
 'fingerprint': {'additionalProperties': False,
                 'properties': {'bytes': {'maximum': 65536, 'minimum': 0, 'type': 'integer'},
                                'ctime_ns': {'maximum': 9223372036854775807,
                                             'minimum': -9223372036854775808,
                                             'type': 'integer'},
                                'device': {'maximum': 9223372036854775807,
                                           'minimum': 0,
                                           'type': 'integer'},
                                'inode': {'maximum': 9223372036854775807,
                                          'minimum': 1,
                                          'type': 'integer'},
                                'mtime_ns': {'maximum': 9223372036854775807,
                                             'minimum': -9223372036854775808,
                                             'type': 'integer'}},
                 'required': ['device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'],
                 'type': 'object'},
 'openIdentity': {'additionalProperties': False,
                  'properties': {'after_fd': {'$ref': '#/$defs/fingerprint'},
                                 'after_path': {'$ref': '#/$defs/fingerprint'},
                                 'before': {'$ref': '#/$defs/fingerprint'},
                                 'bytes_read': {'maximum': 65536, 'minimum': 1, 'type': 'integer'},
                                 'expected_sha256': {'pattern': '^[0-9a-f]{64}$', 'type': 'string'},
                                 'opened': {'$ref': '#/$defs/fingerprint'},
                                 'readback_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                     'type': 'string'}},
                  'required': ['before',
                               'opened',
                               'after_fd',
                               'after_path',
                               'bytes_read',
                               'expected_sha256',
                               'readback_sha256'],
                  'type': 'object'},
 'ordinaryReplace': {'additionalProperties': False,
                     'properties': {'after': {'$ref': '#/$defs/fingerprint'},
                                    'before': {'$ref': '#/$defs/fingerprint'},
                                    'expected_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                        'type': 'string'},
                                    'readback_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                        'type': 'string'},
                                    'replacement': {'$ref': '#/$defs/fingerprint'}},
                     'required': ['before',
                                  'replacement',
                                  'after',
                                  'expected_sha256',
                                  'readback_sha256'],
                     'type': 'object'},
 'pureEvidence': {'additionalProperties': False,
                  'properties': {'failure_count': {'const': 0},
                                 'native_network_model_work': {'const': False},
                                 'report_sha256': {'pattern': '^[0-9a-f]{64}$', 'type': 'string'},
                                 'scope': {'const': 'source_only_pure_mocks'},
                                 'source_inventory_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                             'type': 'string'},
                                 'test_count': {'maximum': 4096, 'minimum': 1, 'type': 'integer'}},
                  'required': ['scope',
                               'source_inventory_sha256',
                               'test_count',
                               'failure_count',
                               'report_sha256',
                               'native_network_model_work'],
                  'type': 'object'},
 'root': {'additionalProperties': False,
          'properties': {'canonical_root': {'maxLength': 4096, 'pattern': '^/', 'type': 'string'},
                         'device': {'maximum': 9223372036854775807,
                                    'minimum': 0,
                                    'type': 'integer'},
                         'directory_inode': {'maximum': 9223372036854775807,
                                             'minimum': 1,
                                             'type': 'integer'},
                         'filesystem_type': {'const': 'ext4'}},
          'required': ['canonical_root', 'device', 'directory_inode', 'filesystem_type'],
          'type': 'object'},
 'sameLengthWrite': {'additionalProperties': False,
                     'properties': {'after': {'$ref': '#/$defs/fingerprint'},
                                    'after_bytes_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                           'type': 'string'},
                                    'before': {'$ref': '#/$defs/fingerprint'},
                                    'before_bytes_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                            'type': 'string'}},
                     'required': ['before', 'after', 'before_bytes_sha256', 'after_bytes_sha256'],
                     'type': 'object'},
 'sourceProbe': {'additionalProperties': False,
                 'properties': {'open_identity': {'$ref': '#/$defs/openIdentity'},
                                'ordinary_replace': {'$ref': '#/$defs/ordinaryReplace'},
                                'probe_root': {'$ref': '#/$defs/root'},
                                'same_length_write': {'$ref': '#/$defs/sameLengthWrite'}},
                 'required': ['probe_root',
                              'open_identity',
                              'same_length_write',
                              'ordinary_replace'],
                 'type': 'object'},
 'sourceRecipe': {'additionalProperties': False,
                  'properties': {'native_binary': {'additionalProperties': False,
                                                   'properties': {'bytes': {'maximum': 33554432,
                                                                            'minimum': 1,
                                                                            'type': 'integer'},
                                                                  'mode': {'const': 'hosted_renderer_private_channel_v1'},
                                                                  'sha256': {'pattern': '^[0-9a-f]{64}$',
                                                                             'type': 'string'}},
                                                   'required': ['sha256', 'bytes', 'mode'],
                                                   'type': 'object'},
                                 'prior_tiny_native_evidence_sha256': {'pattern': '^[0-9a-f]{64}$',
                                                                       'type': 'string'},
                                 'prior_tiny_native_scope': {'const': 'two_tiny_source_full_state_projection_cleanup_cases_only'},
                                 'pure_evidence': {'$ref': '#/$defs/pureEvidence'},
                                 'recipe': {'const': 'pinned_smol_static_source_v1'},
                                 'runtime_head': {'pattern': '^[0-9a-f]{40}$', 'type': 'string'},
                                 'runtime_inventory': {'items': {'$ref': '#/$defs/file'},
                                                       'maxItems': 128,
                                                       'minItems': 1,
                                                       'type': 'array'},
                                 'runtime_tree': {'pattern': '^[0-9a-f]{40}$', 'type': 'string'},
                                 'version': {'const': 1}},
                  'required': ['version',
                               'recipe',
                               'runtime_head',
                               'runtime_tree',
                               'runtime_inventory',
                               'native_binary',
                               'prior_tiny_native_evidence_sha256',
                               'prior_tiny_native_scope',
                               'pure_evidence'],
                  'type': 'object'}}


def validate_support(value, kind):
    """The reviewed finite schema subset, with exact JSON numeric/boolean types."""
    def visit(v,s):
        if '$ref' in s:return visit(v,SCHEMAS[s['$ref'].split('/')[-1]])
        if 'const' in s:require(type(v) is type(s['const']) and v==s['const'],'Supporting constant/type differs')
        t=s.get('type')
        if t=='object':
            fields(v,s['required'])
            for k,child in s['properties'].items():visit(v[k],child)
        elif t=='integer':integer(v,s.get('minimum',0),s.get('maximum',2**63-1))
        elif t=='string':
            require(type(v) is str and s.get('minLength',0)<=len(v)<=s.get('maxLength',2**31),'Supporting string bound')
            if 'pattern' in s:require(re.search(s['pattern'],v) is not None,'Supporting string encoding')
        elif t=='array':
            require(type(v) is list and s['minItems']<=len(v)<=s['maxItems'],'Supporting inventory bound')
            for x in v:visit(x,s['items'])
    require(kind in ('filesystemReceipt','sourceRecipe'),'Supporting scope unavailable')
    visit(value,SCHEMAS[kind])
    return value


def _verified(path, expected, limit, *, json_artifact=False):
    digest(expected);path=Path(path)
    require(not path.is_symlink() and str(path.resolve(strict=True))==str(path),'Canonical regular evidence path required')
    before=fingerprint(path.lstat());require(before['bytes']<=limit,'Owner evidence exceeds bound')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        require(fingerprint(os.fstat(stream.fileno()))==before,'Evidence identity changed')
        raw=stream.read(limit+1)
        require(fingerprint(os.fstat(stream.fileno()))==before,'Evidence changed while reading')
    require(fingerprint(path.lstat())==before and len(raw)<=limit and hashlib.sha256(raw).hexdigest()==expected,'Owner evidence digest/identity differs')
    value=strict_json(raw) if json_artifact else None
    if json_artifact:require(raw==canonical(value),'Owner artifact must use exact canonical bytes')
    return value,(str(path),tuple(sorted(before.items())))


def runtime_names(root):
    from host_atlas import ASSETS,BUNDLE
    names=set(source_names(root))
    names.update(str(p.relative_to(root)) for p in (root/'tools').glob('inference_*.py'))
    names.update({'tools/host_atlas.py','tools/live_inference.py','tools/profile_atlas.py','tools/static_atlas.py',
                  'docs/models/smollm2-135m.json','docs/models/smollm2-135m-config.json','web/index.html','web/profile-client.js'})
    names.update('web/'+v[0] for v in ASSETS.values());names.update('web/'+p for p in BUNDLE)
    require(1<=len(names)<=128,'Runtime inventory exceeds bound')
    return sorted(names)


def _git_version(root):
    # Trusted bootstrap, bounded Git metadata command; never an HTTP/model job.
    result=subprocess.run(['git','rev-parse','HEAD','HEAD^{tree}'],cwd=root,check=True,
                          stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=1)
    require(len(result.stdout)<=128,'Git metadata exceeds bound')
    values=result.stdout.decode('ascii').splitlines();require(len(values)==2,'Git source identity unavailable')
    return values


def _root_identity(path):
    path=Path(path);require(path.is_dir() and not path.is_symlink() and str(path.resolve(strict=True))==str(path),'Canonical filesystem root required')
    info=path.stat()
    # Read-only mount identity, bounded and local; no probe or mount mutation.
    with Path('/proc/self/mountinfo').open('rb') as stream:raw=stream.read(1024**2+1)
    require(len(raw)<=1024**2,'Mount metadata exceeds bound')
    chosen=None
    for line in raw.decode().splitlines():
        left,right=line.split(' - ',1);parts=left.split();mount=parts[4]
        mount=re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),mount)
        if path.is_relative_to(Path(mount)) and (chosen is None or len(mount)>len(chosen[0])):chosen=(mount,right.split()[0])
    require(chosen is not None,'Filesystem type unavailable')
    return {'canonical_root':str(path),'device':info.st_dev,'directory_inode':info.st_ino,'filesystem_type':chosen[1]}


def _platform():
    import fcntl
    return {'system':sys.platform,'kernel_release':platform.release(),'native_arch':platform.machine(),
            'python_implementation':sys.implementation.name,'python_version':platform.python_version(),
            'required_apis':{'stat_ns':hasattr(os.stat('.'),'st_mtime_ns'),'fstat':hasattr(os,'fstat'),
                             'pread':hasattr(os,'pread'),'o_nofollow':hasattr(os,'O_NOFOLLOW'),
                             'flock_lock_ex':hasattr(fcntl,'flock') and hasattr(fcntl,'LOCK_EX')}}


def filesystem_checks(v, source, cache):
    validate_support(v,'filesystemReceipt')
    require(v['source']==_root_identity(source) and v['cache']==_root_identity(cache)
            and v['platform']==_platform(),'Filesystem/platform binding differs')
    require(source!=cache and not Path(cache).is_relative_to(Path(source)),'Cache/source roots overlap')
    for role in ('source','cache'):
        probe=v['observations'][role];root=probe['probe_root'];target=v[role]
        require(root==_root_identity(root['canonical_root']) and root['device']==target['device']
                and root['filesystem_type']==target['filesystem_type'] and not Path(root['canonical_root']).is_relative_to(Path(source)),
                'Inert probe root/device differs')
        opened=probe['open_identity'];same=probe['same_length_write'];replace=probe['ordinary_replace']
        require(opened['before']==opened['opened']==opened['after_fd']==opened['after_path']
                and opened['bytes_read']==opened['before']['bytes'] and opened['expected_sha256']==opened['readback_sha256'],'Open identity evidence differs')
        keys=('device','inode','bytes')
        require(all(same['before'][k]==same['after'][k] for k in keys)
                and same['before_bytes_sha256']!=same['after_bytes_sha256']
                and any(same['before'][k]!=same['after'][k] for k in ('mtime_ns','ctime_ns')),'Ordinary write evidence differs')
        def publication(item,temp):
            require(item['before']['device']==item['after']['device'] and item['before']['inode']!=item['after']['inode']
                    and all(item[temp][k]==item['after'][k] for k in ('device','inode','bytes','mtime_ns'))
                    and item['after']['ctime_ns']>=item[temp]['ctime_ns']
                    and item['expected_sha256']==item['readback_sha256'],'Ordinary publication evidence differs')
        publication(replace,'replacement')
        if role=='cache':publication(probe['atomic_publish'],'temporary')
        def fingerprints(item):
            if type(item) is dict:
                if set(item)=={'device','inode','bytes','mtime_ns','ctime_ns'}:require(item['device']==target['device'],'Probe fingerprint device differs')
                else:
                    for child in item.values():fingerprints(child)
        fingerprints(probe)



def check_cache_available(path):
    """Cooperative owner preflight of the existing native atlas.lock, no writes.

    Native State::open repeats and retains the exclusive lock. This preflight
    cannot prove exclusion against an uncoordinated external cache writer.
    """
    path=Path(path)
    require(path.resolve()==path and not path.is_symlink(),'Canonical cache path required')
    if not path.exists():return
    require(path.is_dir(),'Cache directory required')
    lock=path/'atlas.lock'
    if not lock.exists():return
    require(not lock.is_symlink(),'Cache lock symlink refused')
    before=fingerprint(lock.lstat());fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        require(fingerprint(os.fstat(fd))==before,'Cache lock identity differs')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('Cache owner busy') from error
        try:require(fingerprint(os.fstat(fd))==before==fingerprint(lock.lstat()),'Cache lock changed')
        finally:fcntl.flock(fd,fcntl.LOCK_UN)
    finally:os.close(fd)

def target_entry(entry):
    manifest=validate_manifest(entry['manifest'])
    require(entry['enabled'] is True and manifest['provenance']=='owner_expected'
            and manifest['repository']==REPOSITORY and manifest['revision']==REVISION
            and manifest['license']['id']=='Apache-2.0' and manifest['license']['accepted'] is True,'Fixed accepted owner target required')
    files={f['name']:f for f in manifest['files']}
    require(files.get('config.json',{}).get('sha256')==CONFIG_SHA
            and files.get('model.safetensors',{}).get('sha256')==WEIGHTS_SHA
            and files['model.safetensors']['bytes']==269060552
            and {n for n in files if n.endswith('.safetensors')}=={'model.safetensors'}
            and 'model.safetensors.index.json' not in files,'Fixed target storage/pins differ')


@dataclass(frozen=True)
class BoundDenseStaticAdmission:
    registry:object
    model_id:str
    owner_sha:str
    binary_sha:str
    binary_path:str
    files:tuple
    inventory:tuple
    source_root:str
    cache_root:str
    filesystem:object
    _seal:object

    def check(self):
        require(self._seal is _SEAL,'Sealed owner policy required')
        for name,saved in self.files:
            path=Path(name);require(not path.is_symlink() and tuple(sorted(fingerprint(path.lstat()).items()))==saved,'Bound evidence changed')
        require(tuple(runtime_names(package_root()))==self.inventory,'Bound runtime inventory changed')
        # Root mount/platform identity remains current; no new filesystem experiment.
        require(self.filesystem['source']==_root_identity(self.source_root)
                and self.filesystem['cache']==_root_identity(self.cache_root)
                and self.filesystem['platform']==_platform(),'Bound filesystem moved/changed')
        entry=self.registry.owner_receipt(self.model_id);target_entry(entry);_current(self.registry,entry)
        require(entry['root']==self.source_root and hashlib.sha256(canonical(entry)).hexdigest()==self.owner_sha,'Full owner receipt changed')
        return entry

    def allows(self,identifier):
        if identifier!=self.model_id:return False
        try:
            require(self._seal is _SEAL,'Sealed policy required')
            for name,saved in self.files:
                path=Path(name);require(not path.is_symlink() and tuple(sorted(fingerprint(path.lstat()).items()))==saved,'Cached seal changed')
            entry=self.registry.owner_receipt(identifier);_current(self.registry,entry)
            return hashlib.sha256(canonical(entry)).hexdigest()==self.owner_sha
        except (ValueError,OSError,KeyError):return False

    def admit(self,prepared):
        require(prepared.entry==self.check(),'Prepared owner receipt differs')
        require(prepared.descriptor['model_type']=='llama' and len(prepared.tensors)==272
                and prepared.descriptor['parameter_count']==134515008 and all(t['dtype']=='BF16' for t in prepared.tensors.values()),'Fixed dense Smol topology required')
        prepared.check();return True

    def check_binary(self,binary):
        self.check();binary.check()
        require(str(binary.path)==self.binary_path,'Owner binary path differs')


def bind_dense_policy(path,approved_sha,registry,cache,binary_path):
    """Trusted owner pair approves an exact bundle for its reviewed test scope.

    Absence is closed. No registration, license insertion, filesystem experiment,
    model payload hashing or real-model qualification is performed here.
    """
    require(QUALIFIED_READER_SHA is not None,'Combined native reader qualification pending')
    root=package_root();path=Path(path)
    policy,identity=_verified(path,approved_sha,32768,json_artifact=True)
    fields(policy,('version','policy','enabled','model_id','owner_receipt_sha256','filesystem_receipt_sha256','source_recipe_sha256','native_binary_sha256'))
    require(type(policy['version']) is int and policy['version']==1 and policy['policy']=='pinned_smol_dense_static_v1'
            and policy['enabled'] is True and type(policy['model_id']) is str and re.fullmatch(r'm_[0-9a-f]{64}',policy['model_id']) is not None,'Closed fixed policy required')
    for key in ('owner_receipt_sha256','filesystem_receipt_sha256','source_recipe_sha256','native_binary_sha256'):digest(policy[key])
    receipts=path.parent/'receipts';require(not receipts.is_symlink() and receipts.resolve()==receipts,'Canonical receipt containment required')
    bound=[identity];support={}
    for key,kind in [('filesystem_receipt_sha256','filesystemReceipt'),('source_recipe_sha256','sourceRecipe')]:
        value,proof=_verified(receipts/(policy[key]+'.json'),policy[key],32768,json_artifact=True)
        support[kind]=validate_support(value,kind);bound.append(proof)
    entry=registry.owner_receipt(policy['model_id']);target_entry(entry);_current(registry,entry)
    require(hashlib.sha256(canonical(entry)).hexdigest()==policy['owner_receipt_sha256'],'Complete owner receipt differs')
    cache=str(Path(cache));filesystem_checks(support['filesystemReceipt'],entry['root'],cache)
    recipe=support['sourceRecipe'];names=runtime_names(root)
    require([f['path'] for f in recipe['runtime_inventory']]==names and _git_version(root)==[recipe['runtime_head'],recipe['runtime_tree']],'Reviewed runtime identity differs')
    for file in recipe['runtime_inventory']:
        name=file['path'];parts=Path(name).parts
        require(not Path(name).is_absolute() and all(p not in ('.','..') for p in parts)
                and (root/name).resolve().is_relative_to(root),'Source containment differs')
        _,proof=_verified(root/name,file['sha256'],524288)
        require(dict(proof[1])['bytes']==file['bytes'],'Reviewed source size differs');bound.append(proof)
    require(recipe['pure_evidence']['source_inventory_sha256']==hashlib.sha256(canonical(recipe['runtime_inventory'])).hexdigest()
            and recipe['prior_tiny_native_evidence_sha256']==PRIOR_TINY_SHA,'Source/pure/prior tiny evidence scope differs')
    native=recipe['native_binary'];require(native['sha256']==policy['native_binary_sha256']==QUALIFIED_READER_SHA,'Qualified native recipe digest differs')
    declared={f['path']:f['sha256'] for f in recipe['runtime_inventory']}
    require(all(declared.get(name)==sha for name,sha in NATIVE_SOURCE_PINS.items()),'Qualified native source behavior differs')
    _,proof=_verified(Path(binary_path),native['sha256'],32*1024**2)
    require(dict(proof[1])['bytes']==native['bytes'],'Native recipe size differs');bound.append(proof)
    result=BoundDenseStaticAdmission(registry,policy['model_id'],policy['owner_receipt_sha256'],native['sha256'],
            str(binary_path),tuple(bound),tuple(names),entry['root'],cache,support['filesystemReceipt'],_SEAL)
    result.check();return result


def check_native_model(prepared,model):
    """Structural binder is separate; this checks owner name and saved-stat claims."""
    require(type(model) is dict and model.get('name')==prepared.entry['name'],'Native owner name differs')
    c=model.get('coverage');require(type(c) is dict and 'fresh_source_hashes' in model and model['fresh_source_hashes'] is None,'Fresh hash claims unavailable')
    for key in ('sha_hashed_shards','sha_expected_matched_shards','sha_missing_expected_shards','sha_verified_shards'):integer(c.get(key),0,0)
    require(c.get('source_complete') is True and c.get('active_tensor') is None and c.get('all_requested') is False,'Native calibration is not idle')
    require('calibration_error' in c and c['calibration_error'] is None,'Native calibration error pending')
    integer(c.get('materialized_bytes'),0,2*1024**3);integer(c.get('materialized_tiles'),0,1000)
    require(c.get('cache_budget_bytes')==2*1024**3 and c.get('fine_tile_file_cap')==1000,'Native cache cap differs')
    tensors=model.get('catalog');require(type(tensors) is list,'Native catalog required')
    completed=0;values=0;maxima=[]
    numeric=('max_abs','median_nonzero_abs','q99')
    nullable=(*numeric,'quantile_order_statistics','quantile_interpolation','robust_clipped_count','exact_zero_count','unique_bit_patterns','calibration_method')
    for t in tensors:
        require(type(t) is dict and type(t.get('calibration_complete')) is bool and all(k in t for k in nullable),'Calibration fields/types differ')
        expected_rule='supported dtype; requires a valid exact histogram' if t['calibration_complete'] else 'calibration pending'
        require(t.get('rule_status')=={'tensor_signed_percentile':expected_rule},'Native BF16 rule status differs')
        if not t['calibration_complete']:
            require(all(t.get(k) is None for k in nullable),'Uncalibrated statistics present');continue
        integer(t.get('count'),1);completed+=1;values+=t['count']
        for key in numeric:require(type(t.get(key)) in (int,float) and math.isfinite(t[key]) and 0<=t[key]<=t['max_abs'],'Saved statistic range differs')
        for key in ('robust_clipped_count','exact_zero_count'):integer(t.get(key),0,t['count'])
        integer(t.get('unique_bit_patterns'),1,65536)
        require(t.get('calibration_method')=='exact-16-bit-histogram' and t.get('quantile_order_statistics')=='exact'
                and t.get('quantile_interpolation')=='linear in F64; final floating-point rounding possible','Saved BF16 encoding differs')
        maxima.append(t['max_abs'])
    integer(c.get('calibrated_tensors'),completed,completed);integer(c.get('values_streamed'),values,values)
    complete=completed==len(tensors) and bool(tensors)
    require(type(model.get('calibration_complete')) is bool and model['calibration_complete']==complete
            and type(c.get('statistics_complete')) is bool and c['statistics_complete']==complete,'Global coverage differs')
    require('global_max' in model and (type(model['global_max']) in (int,float) and math.isfinite(model['global_max']) if complete else model['global_max'] is None)
            and model['global_max']==(max(maxima) if complete else None),'Global scale differs')
