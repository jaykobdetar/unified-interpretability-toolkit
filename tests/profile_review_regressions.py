#!/usr/bin/env python3
"""Review regressions: fake handles/clocks only; never live fault injection."""
import hashlib
import struct
import threading
import unittest

from profile_worker_primitives import Base, Platform, Token, frame, selected, snapshot
from atlas_host.profile_worker import ProfileJob


class ReviewRegressions(Base):
    def replacement(self):
        p=Platform(self.os); j=ProfileJob(selected(),17,'t'*32,'context',p)
        auth=('t'*32,j.job_capability,'context')
        j.start(*auth,5); p.child.reaped=True
        for _ in range(20):
            j.tick()
            if j.state=='partial': break
        self.assertEqual(j.state,'partial')
        old=j.store.latest; prior=j.store.info.revision
        p.token=Token(); j.start(*auth,12); p.child.reaped=True
        new=p.child.receipt['revision']
        return p,j,auth,old,prior,new

    def finish(self,j):
        for _ in range(20):
            j.tick()
            if j.state in ('error','cancelled','partial','complete'): break

    def test_profile1_failed_old_close_deadline_never_exposes_replacement(self):
        p,j,auth,old,prior,new=self.replacement()
        real_close=self.os.close
        def close(fd):
            real_close(fd)
            if fd==old: p.now=j.deadline+0.1
        self.os.close=close
        self.finish(j)
        self.assertIsNone(j.accepted)
        with self.assertRaises(ValueError): j.page(*auth,new,'rows',0,1)
        self.assertIsNone(j.store.latest)

    def test_profile1_old_close_exception_never_exposes_candidate(self):
        p,j,auth,old,prior,new=self.replacement()
        real_close=self.os.close
        def close(fd):
            if fd==old: raise OSError('fake uncertain close')
            real_close(fd)
        self.os.close=close
        self.finish(j)
        self.assertIsNone(j.accepted)
        with self.assertRaises(ValueError): j.page(*auth,new,'rows',0,1)
        self.assertNotIn(j.state,('partial','complete'))

    def test_profile1_page_denied_during_finalization_then_allowed_on_success(self):
        for phase in ('watchdog','slot'):
            with self.subTest(phase=phase):
                p,j,auth,old,prior,new=self.replacement()
                observed=[]
                def during():
                    try: j.page(*auth,new,'rows',0,1)
                    except ValueError: observed.append('denied')
                    else: observed.append('readable')
                if phase=='watchdog': j.watchdog.close=during
                else: p.token.release_hook=during
                self.finish(j)
                self.assertEqual(observed,['denied'])
                self.assertEqual(j.state,'complete')
                self.assertEqual(j.page(*auth,new,'rows',0,1)['revision'],new)

    def test_profile1_watchdog_and_slot_overruns_cannot_publish(self):
        for phase in ('watchdog','slot'):
            with self.subTest(phase=phase):
                p,j,auth,old,prior,new=self.replacement()
                def overrun(): p.now=j.deadline+0.1
                if phase=='watchdog': j.watchdog.close=overrun
                else: p.token.release_hook=overrun
                self.finish(j)
                self.assertIsNone(j.accepted)
                with self.assertRaises(ValueError): j.page(*auth,new,'rows',0,1)

    def test_profile1_preserves_prior_accepted_frame_on_preswap_failure(self):
        p,j,auth,old,prior,new=self.replacement()
        p.child.receipt['revision']='0'*64
        self.finish(j)
        self.assertEqual(j.store.latest,old)
        self.assertEqual(j.page(*auth,prior,'rows',0,1)['revision'],prior)
        with self.assertRaises(ValueError): j.page(*auth,new,'rows',0,1)

    def test_profile2_watchdog_counts_old_storage_until_confirmed_close(self):
        p,j,auth,old,prior,new=self.replacement()
        reported=[]; actual=[]
        p.resources_ok=lambda count: reported.append(count) or True
        real_close=self.os.close
        def close(fd):
            if fd==old:
                actual.append(len(self.os.files)*j.store.frame_bytes)
                watchdog=threading.Thread(target=j.guard)
                watchdog.start(); watchdog.join(timeout=1)
                self.assertFalse(watchdog.is_alive(), 'Watchdog blocked behind old-close lock')
            real_close(fd)
        self.os.close=close
        self.finish(j)
        self.assertEqual(j.state,'complete')
        self.assertIn(2*j.store.frame_bytes,actual)
        # Last guard during close must count BOTH allocations, irrespective of roles.
        self.assertEqual(reported[-1],actual[-1])

    def test_profile2_uncertain_close_stays_charged_and_slot_poisoned_no_retry(self):
        p,j,auth,old,prior,new=self.replacement()
        attempts=[]; reported=[]
        p.resources_ok=lambda count: reported.append(count) or True
        real_close=self.os.close
        def close(fd):
            attempts.append(fd)
            if fd==old: raise OSError('unknown descriptor disposition')
            real_close(fd)
        self.os.close=close
        self.finish(j)
        for _ in range(4): j.guard(); j.tick()
        self.assertTrue(p.token.held)
        self.assertTrue(p.token.poisoned)
        self.assertEqual(attempts.count(old),1)  # Never retry a possibly recycled fd.
        self.assertGreaterEqual(reported[-1],j.store.frame_bytes)
        self.assertIsNone(j.store.latest)

    def test_profile3_exact_dtype_maxima_and_consistent_above_max_refusal(self):
        for dtype,maximum in [('F16',65504.),('BF16',float.fromhex('0x1.fep+127')),
                              ('F32',float.fromhex('0x1.fffffep+127'))]:
            binding=selected(1,1); binding['dtype']=dtype
            raw,*_=frame(binding,17,1)
            start=168+len(snapshot.canonical(binding))
            for value,valid in [(maximum,True),(maximum*1.000001,False)]:
                with self.subTest(dtype=dtype,valid=valid):
                    modified=bytearray(raw)
                    for axis in range(4):
                        modified[start+axis*24:start+axis*24+8]=struct.pack('<d',value)
                    if valid: self.assertEqual(self.validate(bytes(modified),binding).visited,1)
                    else:
                        with self.assertRaises(ValueError): self.validate(bytes(modified),binding)


if __name__=='__main__': unittest.main(verbosity=2)
