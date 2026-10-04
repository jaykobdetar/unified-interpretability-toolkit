"""Bounded progress projection, separate from job admission and numerical work."""
import math
import secrets
from copy import deepcopy

from .cache import binding
from .common import canonical, fields, integer, require

STATES = {'queued', 'running', 'partial', 'complete', 'cancelled', 'error'}
ERRORS = {None, 'budget_exhausted', 'source_changed', 'cancelled',
          'lease_expired', 'resource_limit', 'worker_error'}


def model_id(value):
    from .common import digest
    require(type(value) is str and value.startswith('m_'), 'Invalid model ID')
    digest(value[2:])
    return value


def validate_progress(value):
    fields(value, ('model_id', 'kind', 'binding', 'state', 'visited_values',
                   'total_values', 'elapsed_active_ms', 'remaining_authorized_work',
                   'complete', 'error'))
    model_id(value['model_id'])
    require(value['kind'] in ('calibration', 'profile', 'overview'), 'Invalid progress kind')
    selected = binding(value['binding'])
    expected = math.prod(selected['shape'] if value['kind'] == 'calibration'
                         else selected['shape'][-2:])
    integer(value['total_values'], expected, expected)
    integer(value['visited_values'], 0, expected)
    integer(value['elapsed_active_ms'])
    fields(value['remaining_authorized_work'], ('values', 'wall_ms', 'cpu_ms'))
    for number in value['remaining_authorized_work'].values():
        integer(number)
    require(value['remaining_authorized_work']['values'] <= expected-value['visited_values'],
            'Remaining values exceed unvisited source')
    require(type(value['state']) is str and value['state'] in STATES, 'Invalid progress state')
    require(type(value['complete']) is bool and value['complete'] == (value['state'] == 'complete'),
            'Inconsistent completion')
    require(not value['complete'] or value['visited_values'] == expected, 'Incomplete coverage')
    require((value['error'] is None or type(value['error']) is str)
            and value['error'] in ERRORS, 'Use a public error code, not exception text')
    return deepcopy(value)


class ProgressStore:
    MAX_RECORDS = 64
    MAX_PAGE = 16
    MAX_BYTES = 16384

    def __init__(self, selected_model):
        self.model_id = model_id(selected_model)
        self.epoch = secrets.token_hex(16)
        self.revision = self.floor = 0
        self._records = {}

    def publish(self, key, value, *, owner=None):
        """Trusted coordinator API. Owner is a separately generated job capability.

        Returns revision; HTTP adapters must never let clients publish progress.
        All profile records are private; calibration/overview may be public.
        """
        require(type(key) is str and len(key) == 32
                and all(c in '0123456789abcdef' for c in key), 'Invalid progress record ID')
        record = validate_progress(value)
        require(record['model_id'] == self.model_id, 'Model binding mismatch')
        require(owner is None or (type(owner) is str and len(owner) == 64
                and all(c in '0123456789abcdef' for c in owner)), 'Invalid owner capability')
        require(record['kind'] != 'profile' or owner is not None, 'Profile progress is owner-private')
        previous = self._records.get(key)
        if previous is not None:
            require(previous['owner'] == owner, 'Progress owner cannot change')
            old = previous['record']
            require(all(old[k] == record[k] for k in ('binding', 'kind', 'total_values')),
                    'Progress binding cannot change')
            require(record['visited_values'] >= old['visited_values']
                    and record['elapsed_active_ms'] >= old['elapsed_active_ms'],
                    'Progress cannot move backwards')
            if old == record:
                return previous['revision']
            require(old['state'] not in ('complete', 'cancelled', 'error'), 'Terminal record is frozen')
        elif len(self._records) == self.MAX_RECORDS:
            evictable = [key for key, entry in self._records.items()
                         if entry['owner'] is None
                         or entry['record']['state'] in ('complete', 'cancelled', 'error')]
            require(bool(evictable), 'Progress capacity reserved by active owner jobs')
            oldest = min(evictable, key=lambda k: self._records[k]['revision'])
            removed = self._records.pop(oldest)
            self.floor = max(self.floor, removed['revision'])
        self.revision += 1
        self._records[key] = {'record': record, 'owner': owner, 'revision': self.revision}
        return self.revision

    def snapshot(self, cursor=None, *, job=None, capability=None):
        scope = 'public'
        if job is not None:
            entry = self._records.get(job)
            require(entry is not None and entry['owner'] is not None
                    and type(capability) is str
                    and secrets.compare_digest(entry['owner'].encode(), capability.encode()),
                    'Job unavailable or not owned')
            scope = job
        else:
            require(capability is None, 'Job selector required for owner capability')
        valid = (type(cursor) is dict and set(cursor) == {'epoch', 'revision', 'scope'}
                 and cursor['epoch'] == self.epoch and cursor['scope'] == scope
                 and type(cursor['revision']) is int
                 and self.floor <= cursor['revision'] <= self.revision)
        since = cursor['revision'] if valid else 0
        entries = sorted(((key, entry) for key, entry in self._records.items()
                          if entry['revision'] > since
                          and (key == job if job is not None else entry['owner'] is None)),
                         key=lambda pair: pair[1]['revision'])
        response = {'version': 1, 'model_id': self.model_id, 'reset': not valid,
                    'records': [], 'has_more': False,
                    'cursor': {'epoch': self.epoch, 'revision': since, 'scope': scope}}
        for key, entry in entries:
            candidate = {'id': key, 'revision': entry['revision'], **entry['record']}
            trial = deepcopy(response)
            trial['records'].append(candidate)
            trial['cursor']['revision'] = entry['revision']
            if len(trial['records']) > self.MAX_PAGE or len(canonical(trial)) > self.MAX_BYTES-128:
                require(bool(response['records']), 'Progress record exceeds response limit')
                response['has_more'] = True
                break
            response = trial
        if not response['has_more']:
            response['cursor']['revision'] = self.revision
        require(len(canonical(response)) <= self.MAX_BYTES, 'Progress response exceeds byte limit')
        return deepcopy(response)
