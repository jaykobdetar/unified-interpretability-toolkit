"""Pure hosted supervisor/provider/router checks; no threads/processes/FDs/sockets."""
import json
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atlas_host.supervisor import Supervisor, CpuLedger, AdmissionGrant, HostedLegacyLane
from atlas_host.profile_platform import SupervisedProfile
from atlas_host.profile_api import PrivateProfileAPI, production_capabilities
from profile_worker_primitives import Base, selected, Platform as PrimitiveHooks


class Clock:
    def __init__(self): self.now=0.;self.parts={'admission':0.,'owner':0.,'watchdog':0.}
    def ledger(self): return CpuLedger({k:lambda k=k:self.parts[k] for k in self.parts})
    def grant(self): return AdmissionGrant(lambda:self.now,self.ledger())


class SharedTests(unittest.TestCase):
    def test_all_numeric_kinds_share_one_slot_and_busy_has_no_queue(self):
        s=Supervisor()
        for first in ['profile','inference','analytics','tile','calibration','overview']:
            token=s.acquire(first,'c')
            for other in ['profile','inference','analytics','tile','calibration']:
                with self.assertRaises(ValueError):s.acquire(other,'c')
            with self.assertRaises(ValueError):token.release()
            s.admission_aborted(token,no_child_created=True);token.release()
        self.assertFalse(s.busy())

    def test_native_disconnect_retains_until_exact_ack_or_renderer_reap(self):
        s=Supervisor();t=s.acquire('tile','c');command=s.native_command(t,{'tensor':0},7000)
        self.assertEqual(command['remaining_ms'],7000)
        self.assertNotIn('capability',json.dumps(command));s.transport_lost(t)
        self.assertTrue(s.busy());self.assertFalse(t.current())
        ack={'version':1,'operation_id':'wrong','complete':True,'numeric_idle':True}
        with self.assertRaises(ValueError):s.native_ack(t,ack)
        with self.assertRaises(ValueError):t.release()
        ack['operation_id']=t.operation;s.native_ack(t,ack);t.release()
        t2=s.acquire('calibration','c')
        with self.assertRaises(ValueError):s.native_ack(t2,ack)
        s.transport_lost(t2);s.renderer_reaped(t2,reaped=True,descendants_clear=True);t2.release()

    def test_existing_inference_analytics_uncertain_cleanup_still_blocks(self):
        s=Supervisor();session=types.SimpleNamespace(process=None,status='stopping');jobs=types.SimpleNamespace(busy=False)
        lane=HostedLegacyLane(s,session,jobs)
        with self.assertRaises(ValueError):lane.reserve('analytics','c')
        session.status='idle';jobs.busy=True
        with self.assertRaises(ValueError):lane.reserve('inference','c')
        jobs.busy=False;t=lane.reserve('inference','c');session.process=object()
        with self.assertRaises(ValueError):lane.finish(t,reaped=True,descendants_clear=True,finalized=True)
        session.process=None
        with self.assertRaises(ValueError):lane.finish(t,reaped=True,descendants_clear=False,finalized=True)
        lane.finish(t,reaped=True,descendants_clear=True,finalized=True)

    def test_outer_grant_includes_admission_transfer_and_final_child_replacement(self):
        c=Clock();g=c.grant();c.now=.4;c.parts['admission']=.2;c.parts['owner']=.1;c.parts['watchdog']=.05
        g.sample_child(.5);g.sample_child(.6)
        remaining=g.remaining(800)
        self.assertEqual(remaining['wall_ms'],3800);self.assertEqual(remaining['cpu_ms'],3050)
        with self.assertRaises(ValueError):g.sample_child(.59)
        c.parts['owner']=3.2
        with self.assertRaises(ValueError):g.remaining()

    def test_bad_or_missing_context_cpu_meter_refuses(self):
        with self.assertRaises(ValueError):CpuLedger({'owner':lambda:0})
        c=Clock();g=c.grant();c.parts['watchdog']=float('nan')
        with self.assertRaises(ValueError):g.remaining()


class Hooks(PrimitiveHooks):
    def __init__(self,os,clock):super().__init__(os);self.clockdata=clock;self.resource_rss=50*1024**2;self.closed_at=None
    def start_gate(self):return 6*1024**3,30*1024**3
    def resources(self):return {'rss_bytes':self.resource_rss,'available_bytes':6*1024**3,'all_owned_accounted':True,'descendants_clear':True}
    def watch(self,callback,deadline):
        self.callback=callback;self.original_deadline=deadline
        def close():
            self.watch_closed=True
            if self.closed_at is not None:self.clockdata.now=self.closed_at
        return types.SimpleNamespace(close=close)
    def no_child_created(self):return not hasattr(self,'child')
    def snapshot_closed(self,store):return not self.os.files


class ProviderTests(Base):
    def make(self,delay=0):
        self.clock=Clock();self.grant=self.clock.grant();self.supervisor=Supervisor();self.hooks=Hooks(self.os,self.clock)
        self.clock.now=delay
        self.runtime=SupervisedProfile(self.supervisor,self.grant,self.hooks,selected(),17,'a'*64,'context')
        return self.runtime
    def finish(self,r):
        self.hooks.child.reaped=True
        for _ in range(25):
            r.tick()
            if r.platform.closed:return
        self.fail('Provider did not finalize')

    def test_actual_primitive_provider_reaps_before_common_slot_release(self):
        r=self.make(.3);r.start(5);self.assertTrue(self.supervisor.busy())
        self.assertEqual(self.hooks.original_deadline,5)
        self.finish(r);self.assertEqual(r.job.state,'partial');self.assertFalse(self.supervisor.busy())
        self.assertEqual(r.page(*r.auth,r.job.accepted['revision'],'rows',0,2)['end'],2)
        self.assertEqual(self.supervisor.snapshot_reservation,2*r.job.store.frame_bytes)
        r.close();self.assertEqual(self.supervisor.snapshot_reservation,0)

    def test_expired_before_start_never_spawns_and_releases_reservation(self):
        r=self.make();self.clock.now=6
        with self.assertRaises(ValueError):r.start(5)
        self.assertFalse(hasattr(self.hooks,'child'));self.assertTrue(r.platform.closed);self.assertFalse(self.supervisor.busy());r.close()

    def test_independent_watchdog_stops_without_http_or_owner_tick(self):
        r=self.make();r.start(5);self.clock.now=6;self.hooks.callback()
        self.assertTrue(self.hooks.child.stopped);self.assertTrue(r.job.expired);self.assertTrue(self.supervisor.busy())
        r.tick();self.assertTrue(self.supervisor.busy());self.finish(r);self.assertEqual(r.job.state,'error');r.close()

    def test_old_new_retiring_frames_stay_fully_reserved(self):
        r=self.make();r.start(5);self.finish(r);f=r.job.store.frame_bytes
        # Role fields may say only latest while an old close is still pending.
        self.hooks.resource_rss=768*1024**2-f
        self.assertFalse(r.platform.resources_ok(f))
        self.assertEqual(self.supervisor.snapshot_reservation,2*f)
        self.hooks.resource_rss=50*1024**2;r.close()

    def test_page_requires_final_accepted_revision_even_if_store_latest_exists(self):
        r=self.make();r.start(5);self.finish(r);rev=r.job.accepted['revision']
        r.job.state='error';r.job.accepted=None
        with self.assertRaises(ValueError):r.page(*r.auth,rev,'rows',0,1)
        r.close()

    def test_outer_finalization_overrun_never_exposes_page(self):
        r=self.make();r.start(5);self.hooks.closed_at=6;self.finish(r)
        self.assertIsNone(r.job.accepted);self.assertEqual(r.job.state,'error')
        with self.assertRaises(ValueError):r.page(*r.auth,'a'*64,'rows',0,1)
        r.close()

    def test_new_profile_session_cannot_hide_existing_snapshot_allocation(self):
        r=self.make();r.start(5);self.finish(r)
        with self.assertRaises(ValueError):SupervisedProfile(self.supervisor,self.clock.grant(),self.hooks,selected(),17,'b'*64,'c2')
        self.assertFalse(self.supervisor.busy());self.assertIs(self.supervisor.profile_session,r);r.close()

    def test_wrong_owner_cannot_page_or_close_via_public_api(self):
        r=self.make();r.start(5);self.finish(r)
        with self.assertRaises(ValueError):r.page('b'*64,r.auth[1],r.auth[2],r.job.accepted['revision'],'rows',0,1)
        r.close()

    def test_uncertain_retiring_handle_blocks_outer_finalization(self):
        r=self.make();r.job.state='error'
        # Simulate the corrected primitive's retained close-disposition charge.
        r.job.store._owned=(123,);r.job.store._uncertain=frozenset({123})
        self.assertFalse(r.platform.finish(r.job))
        self.assertTrue(self.supervisor.busy());self.assertTrue(r.platform.reserved.poisoned)
        self.assertEqual(self.supervisor.snapshot_reservation,2*r.job.store.frame_bytes)
        with self.assertRaises(ValueError):r.close()


class RouterTests(unittest.TestCase):
    def test_production_disabled_and_control_reads_never_get_a_grant(self):
        self.assertEqual(production_capabilities(),{'profiles_enabled':False,'resume_available':False})
        calls=[];service=types.SimpleNamespace(status=lambda d:calls.append(d) or {'state':'running'})
        api=PrivateProfileAPI(service,lambda *args:None)
        data={'version':1,'model_id':'m_'+'a'*64,'context_id':'c','tab_capability':'b'*64,'job_id':'d'*32,'job_capability':'e'*64}
        self.assertEqual(api.handle('status',data)[2],{'Cache-Control':'no-store'})
        with self.assertRaises(ValueError):api.handle('status',data,admission=Clock().grant())
        self.assertEqual(len(calls),1)

    def test_resume_unknown_fields_and_foreign_owner_refuse_before_service(self):
        api=PrivateProfileAPI(None,lambda *args:(_ for _ in ()).throw(ValueError('not owner')))
        data={'version':1,'model_id':'m_'+'a'*64,'context_id':'c','tab_capability':'b'*64,'binding':selected(),'seed':17,'values':1,'restart':False}
        with self.assertRaises(ValueError):api.handle('start',{**data,'resume':True},admission=Clock().grant())
        with self.assertRaises(ValueError):api.handle('start',data,admission=Clock().grant())

if __name__=='__main__':unittest.main(verbosity=2)
