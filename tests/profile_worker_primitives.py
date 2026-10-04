#!/usr/bin/env python3
"""Pure stdlib reference/mocks. No OS memfd, child, sockets or model reads."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from atlas_host import profile_snapshot as snapshot
from atlas_host.profile_worker import ProfileJob


def selected(rows=3, cols=4):
    return {'version':2,'model_identity':'a'*64,'source_identity':'b'*64,'tensor':0,
            'name':'weights','dtype':'F32','shape':[2,rows,cols],'rows':rows,'cols':cols,
            'slice':{'leading_indices':[1],'display_axes':[1,2]}}


def reference_permutation(n, seed):
    # Build explicit pair-swap maps in a tiny reference domain, independently
    # of production's per-index destination implementation.
    def mix(x):
        mask = (1 << 64)-1
        x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    mapping = list(range(n))
    for round_index in range(8):
        key = mix(seed ^ ((round_index*0x9e3779b97f4a7c15) % (1 << 64)))
        pairs = list(range(n))
        for a in range(n):
            b = (key % n-a) % n
            if a <= b and mix(key ^ a) % 2:
                pairs[a], pairs[b] = b, a
        mapping = [pairs[i] for i in mapping]
    return mapping


def frame(binding, seed=17, visited=None, initial=None, begin=0):
    rows, cols = binding['rows'], binding['cols']; n = rows*cols
    visited = n if visited is None else visited
    values = [(-1 if i % 3 else 1)*(i+1)/8 for i in range(n)]
    perm = reference_permutation(n, seed)
    sums = initial if initial is not None else [[0.,0.,0] for _ in range(2*(rows+cols))]
    for i in range(begin, visited):
        j = perm[i]
        for slot in (i//cols, rows+i % cols, rows+cols+j//cols, 2*rows+cols+j % cols):
            entry = sums[slot]; y = abs(values[i])-entry[1]; new = entry[0]+y
            entry[1] = (new-entry[0])-y; entry[0] = new; entry[2] += 1
    raw_binding = json.dumps(binding,sort_keys=True,separators=(',',':')).encode()
    identity = hashlib.sha256(json.dumps(['weight-atlas-strength-v1',binding,seed,'swap-or-not-8-v1'],sort_keys=True,separators=(',',':')).encode()).digest()
    raw = (b'WAPROF01'+struct.pack('<II7Q',1,len(raw_binding),rows,cols,n,visited,seed,len(sums),24*len(sums))
           +hashlib.sha256(raw_binding).digest()+hashlib.sha256(snapshot.ALGORITHM).digest()+identity
           +raw_binding+b''.join(struct.pack('<ddQ',*entry) for entry in sums))
    return raw, sums, values, perm


class FakeOS:
    MFD_CLOEXEC = 1
    MFD_ALLOW_SEALING = 2
    def __init__(self): self.files = {}; self.next = 20; self.closed = []
    def memfd_create(self, *_):
        self.next += 1; self.files[self.next] = [b'',0]; return self.next
    def fstat(self, fd): return types.SimpleNamespace(st_mode=0o100600,st_nlink=0,st_size=len(self.files[fd][0]))
    def pread(self, fd, count, offset): return self.files[fd][0][offset:offset+count]
    def close(self, fd): self.closed.append(fd); del self.files[fd]
    def fcntl(self, fd, op, value=None):
        if op == 1033: self.files[fd][1] |= value
        return self.files[fd][1]
    def put(self, raw, sealed=True):
        fd = self.memfd_create(); self.files[fd] = [raw, snapshot.SEALS if sealed else 0]; return fd


class Base(unittest.TestCase):
    def setUp(self):
        self.os = FakeOS()
        self.patches = [patch.object(snapshot,'os',self.os),patch.object(snapshot,'fcntl',types.SimpleNamespace(F_ADD_SEALS=1033,F_GET_SEALS=1034,fcntl=self.os.fcntl))]
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
    def validate(self, raw, binding=None, seed=17):
        return snapshot.validate(self.os.put(raw), binding or selected(), seed,
            hashlib.sha256(raw).hexdigest(),deadline=5,clock=lambda:0)
    def candidate(self, store, visited):
        raw,*_ = frame(store.selected,store.seed,visited)
        fd = store.begin(); self.os.files[fd] = [raw,snapshot.SEALS]
        return hashlib.sha256(raw).hexdigest()
    def publish(self, store, revision):
        return store.publish(revision,minimum_visited=0,maximum_visited=12,deadline=5,
                             source_check=lambda:None,final_check=lambda:None,clock=lambda:0)


class SnapshotTests(Base):
    def test_every_prefix_counts_means_and_split_equivalence(self):
        for rows,cols in [(1,1),(2,3),(3,4),(5,7)]:
            binding = selected(rows,cols)
            for seed in [0,17,2**32-1]:
                for k in range(rows*cols+1):
                    raw,sums,values,perm = frame(binding,seed,k)
                    info = self.validate(raw,binding,seed)
                    self.assertEqual(info.visited,k)
                    self.assertEqual(sorted(perm),list(range(rows*cols)))
                    # Direct independent means from selected source subsets.
                    groups = [[abs(values[i]) for i in range(k) if i//cols == r] for r in range(rows)]
                    groups += [[abs(values[i]) for i in range(k) if i % cols == c] for c in range(cols)]
                    groups += [[abs(values[i]) for i in range(k) if perm[i]//cols == r] for r in range(rows)]
                    groups += [[abs(values[i]) for i in range(k) if perm[i] % cols == c] for c in range(cols)]
                    for record, group in zip(sums,groups):
                        self.assertEqual(record[2],len(group)); self.assertEqual(record[0],math.fsum(group))
                    split,*_ = frame(binding,seed,rows*cols,sums,k)
                    full,*_ = frame(binding,seed)
                    self.assertEqual(split,full)
    def test_refuses_header_binding_and_semantic_corruption_even_rehashed(self):
        raw,*_ = frame(selected(),17,5)
        offsets = [0,8,12,16,24,32,40,48,56,64,72,104,136,168]
        first = 168+len(snapshot.canonical(selected()))
        offsets += [first+16]
        for offset in offsets:
            bad = bytearray(raw); bad[offset] ^= 1
            with self.subTest(offset=offset),self.assertRaises(ValueError): self.validate(bytes(bad))
        for value in [float('nan'),float('inf'),-1.]:
            bad = bytearray(raw); bad[first:first+8] = struct.pack('<d',value)
            with self.assertRaises(ValueError): self.validate(bytes(bad))
        for bad in (raw[:-1],raw+b'0'):
            with self.assertRaises(ValueError): self.validate(bad)
        wrong = selected(); wrong['slice']['leading_indices'] = [0]
        with self.assertRaises(ValueError): self.validate(raw,wrong)
        with self.assertRaises(ValueError): self.validate(raw,seed=18)
    def test_seals_checksum_deadline_and_unsafe_axes(self):
        raw,*_ = frame(selected())
        for seal_bits in (0,1,7,14):
            fd = self.os.put(raw); self.os.files[fd][1] = seal_bits
            with self.assertRaises(ValueError): snapshot.validate(fd,selected(),17,hashlib.sha256(raw).hexdigest(),deadline=5,clock=lambda:0)
        with self.assertRaises(ValueError): snapshot.validate(self.os.put(raw),selected(),17,'0'*64,deadline=5,clock=lambda:0)
        with self.assertRaises(ValueError): snapshot.validate(self.os.put(raw),selected(),17,hashlib.sha256(raw).hexdigest(),deadline=0,clock=lambda:0)
        with self.assertRaises(ValueError): snapshot.layout(selected(200000,200000))
    def test_atomic_replacement_failure_and_reader_overlap(self):
        store = snapshot.SnapshotStore(selected(),17)
        first = self.candidate(store,5); self.publish(store,first); old = store.latest
        with store.page_handle(first):
            with self.assertRaises(ValueError): store.begin()
            with self.assertRaises(ValueError): store.close()
        second = self.candidate(store,12)
        with store.page_handle(first),self.assertRaises(ValueError): self.publish(store,second)
        self.assertEqual(store.latest,old); self.assertIn(old,self.os.files)
        second = self.candidate(store,12)
        with self.assertRaises(ValueError): self.publish(store,'0'*64)
        self.assertEqual(store.latest,old)
        second = self.candidate(store,12); self.publish(store,second)
        self.assertNotIn(old,self.os.files); self.assertEqual(store.info.visited,12)
        store.close(); self.assertIsNone(store.latest)
    def test_chunked_validation_cancellation_cleans_candidate_keeps_old(self):
        store = snapshot.SnapshotStore(selected(),17)
        self.publish(store,self.candidate(store,5)); old = store.latest
        steps = store.publication_steps(self.candidate(store,10),minimum_visited=0,maximum_visited=12,
            deadline=5,source_check=lambda:None,final_check=lambda:None,clock=lambda:0)
        next(steps); steps.close()
        self.assertIsNone(store.pending); self.assertEqual(store.latest,old)
    def test_pages_pin_revision_and_preserve_paired_partial_counts(self):
        store=snapshot.SnapshotStore(selected(),17)
        revision=self.candidate(store,5); self.publish(store,revision)
        page=store.page(revision,'rows',0,3,source_check=lambda:None)
        self.assertEqual(sum(x['visited_count'] for x in page['original']),5)
        self.assertEqual(sum(x['visited_count'] for x in page['control']),5)
        self.assertEqual(page['revision'],revision)
        page['binding']['name']='changed'; self.assertEqual(store.selected['name'],'weights')
        with self.assertRaises(ValueError): store.page('0'*64,'rows',0,1,source_check=lambda:None)
        with self.assertRaises(ValueError): store.page(revision,'rows',0,1025,source_check=lambda:None)
    def test_no_memfd_fallback(self):
        with patch.object(snapshot,'os',types.SimpleNamespace()):
            with self.assertRaisesRegex(ValueError,'no disk fallback'): snapshot.new_memfd()


class Token:
    def __init__(self): self.held=True; self.poisoned=False; self.releases=0; self.release_hook=lambda:None
    def current(self): return self.held
    def poison(self): self.poisoned=True
    def release(self): self.release_hook(); self.held=False; self.releases+=1


class Child:
    def __init__(self, receipt): self.receipt=receipt; self.reaped=False; self.cpu=0.; self.stopped=False; self.init_fail=False
    def initialize(self):
        if self.init_fail: raise ValueError('injected post-spawn failure')
    def sample(self): return dict(cpu_seconds=self.cpu,reaped=self.reaped,exit_code=0 if self.reaped else None,receipt=self.receipt if self.reaped else None)
    def stop(self): self.stopped=True


class Platform:
    def __init__(self, fake_os):
        self.os=fake_os; self.now=0.; self.cpu=0.; self.token=Token(); self.calls=0
        self.fail_init=False; self.valid_source=True; self.valid_resources=True; self.watch_closed=False
    def clock(self): return self.now
    def owner_cpu(self): return self.cpu
    def acquire(self): self.calls+=1; return self.token
    def arm_watchdog(self, job): return types.SimpleNamespace(close=lambda:setattr(self,'watch_closed',True))
    def source_check(self):
        if not self.valid_source: raise ValueError('stale source')
    def start_gate(self): pass
    def resources_ok(self, storage): return self.valid_resources and storage < 32*1024**2
    def spawn(self, request, output, input_fd):
        assert input_fd is None
        raw,*_ = frame(request['binding'],request['seed'],request['values'])
        self.os.files[output]=[raw,snapshot.SEALS]
        self.child=Child({'schema':'weight-atlas.profile-candidate.v1','revision':hashlib.sha256(raw).hexdigest(),
            'visited_values':request['values'],'new_values':request['values'],
            'frame_bytes':len(raw),'live_bytes':snapshot.layout(request['binding'])[1]})
        self.child.init_fail=self.fail_init
        return self.child


class LifecycleTests(Base):
    def job(self, values=5):
        p=Platform(self.os); j=ProfileJob(selected(),17,'t'*32,'context',p)
        auth=('t'*32,j.job_capability,'context')
        j.start(*auth,values)
        return p,j,auth
    def finish(self,p,j):
        p.child.reaped=True
        for _ in range(20):
            j.tick()
            if j.state in ('complete','partial','error','cancelled'): break
    def test_terminal_only_after_reap_validation_no_autochain(self):
        p,j,auth=self.job()
        self.assertIsNone(j.status(*auth)['accepted'])
        j.tick(); self.assertEqual(j.state,'running')
        self.finish(p,j)
        self.assertEqual(j.state,'partial'); self.assertEqual(p.token.releases,1)
        self.assertEqual(j.accepted['visited_values'],5)
        for _ in range(5): j.tick(); j.status(*auth)
        self.assertEqual(p.calls,1)
        with self.assertRaises(ValueError): j.start(*auth,5,resume=True)
    def test_ownership_heartbeat_and_cancel_hold_token_until_reap(self):
        p,j,auth=self.job()
        for wrong in [('x'*32,auth[1],auth[2]),(auth[0],'x'*64,auth[2]),(auth[0],auth[1],'other')]:
            with self.assertRaises(ValueError): j.cancel(*wrong)
        original=j.deadline; j.heartbeat(*auth); self.assertEqual(j.deadline,original)
        j.cancel(*auth); self.assertTrue(p.child.stopped); j.tick(); j.tick()
        self.assertEqual(j.state,'stopping'); self.assertTrue(p.token.held)
        self.assertTrue(p.token.poisoned)
        p.child.reaped=True; j.tick()
        self.assertEqual(j.state,'cancelled'); self.assertFalse(p.token.held); self.assertIsNone(j.store.latest)
    def test_independent_watchdog_expiry_without_status(self):
        p,j,auth=self.job(); p.now=5.01
        j.guard(); self.assertTrue(p.child.stopped)
        j.tick(); p.child.reaped=True; j.tick()
        self.assertIsNone(j.accepted); self.assertEqual(j.state,'error')
    def test_aggregate_cpu_and_resource_refusal(self):
        for reason in ['cpu','memory','lease']:
            p,j,auth=self.job()
            if reason=='cpu': p.cpu=2.1; p.child.cpu=2.0
            if reason=='memory': p.valid_resources=False
            if reason=='lease': j.lease_end=-1
            j.guard(); self.assertTrue(p.child.stopped)
            j.tick(); p.child.reaped=True; j.tick()
            self.assertIsNone(j.accepted)
    def test_postspawn_initialization_failure_retains_owned_child(self):
        p=Platform(self.os); p.fail_init=True
        j=ProfileJob(selected(),17,'t'*32,'context',p); auth=('t'*32,j.job_capability,'context')
        with self.assertRaises(ValueError): j.start(*auth,5)
        self.assertIsNotNone(j.child); self.assertTrue(p.token.held)
        j.tick(); self.assertTrue(p.token.held)
        p.child.reaped=True; j.tick(); self.assertFalse(p.token.held)
    def test_failed_source_or_checksum_preserves_prior_snapshot(self):
        for reason in ['source','digest']:
            p,j,auth=self.job(); self.finish(p,j); old=j.store.latest
            p.token=Token(); j.start(*auth,12); p.child.reaped=True
            if reason=='source': p.valid_source=False
            else: p.child.receipt['revision']='0'*64
            for _ in range(20): j.tick()
            self.assertEqual(j.state,'error'); self.assertEqual(j.store.latest,old)
            self.assertIsNone(j.accepted)
    def test_cancel_after_acceptance_closes_all_session_handles(self):
        p,j,auth=self.job(); self.finish(p,j)
        fd=j.store.latest; self.assertIn(fd,self.os.files)
        j.cancel(*auth); j.tick()
        self.assertNotIn(fd,self.os.files); self.assertIsNone(j.store.latest)
        self.assertEqual(j.state,'cancelled')
    def test_cancel_during_validation_discards_candidate(self):
        p,j,auth=self.job(); p.child.reaped=True
        j.tick(); self.assertEqual(j.state,'validating')
        j.cancel(*auth); j.tick(); j.tick()
        self.assertIsNone(j.store.pending); self.assertIsNone(j.store.latest)
        self.assertIsNone(j.accepted)
    def test_reap_cpu_and_late_validation_fail_closed(self):
        for phase in ['reap_cpu','validation']:
            p,j,auth=self.job(); p.child.reaped=True
            if phase=='reap_cpu': p.child.cpu=4.1
            else:
                j.tick(); p.now=5.01
            for _ in range(4): j.tick()
            self.assertIsNone(j.accepted); self.assertIsNone(j.store.pending)
            self.assertEqual(j.state,'error')
    def test_final_cleanup_overrun_never_reports_success(self):
        p,j,auth=self.job(); p.token.release_hook=lambda:setattr(p,'now',6.)
        self.finish(p,j)
        self.assertIsNone(j.accepted)
        self.assertNotIn(j.state,('partial','complete'))


if __name__=='__main__': unittest.main(verbosity=2)
