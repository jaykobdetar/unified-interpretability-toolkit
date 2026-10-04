"""Small pure/mock production-adapter fixtures; no processes/listeners/SVD."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from email.message import Message

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from analytics.service import AnalyticsJobs, route
from analytics.source import fingerprint
from analytics.worker import analyze_request, catalog_from_payload, validate_request

spec = importlib.util.spec_from_file_location('analytics_live_fixture', ROOT/'tools/live_inference.py')
live = importlib.util.module_from_spec(spec); spec.loader.exec_module(live)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        header = json.dumps({'fixture': {'dtype':'BF16','shape':[2,3],'data_offsets':[0,12]}}).encode()
        self.path = self.root/'tiny.safetensors'
        self.path.write_bytes(struct.pack('<Q',len(header))+header+b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',x))[0]>>16) for x in [1.,-2.,0.,4.,5.,-6.]))
        self.tensor = {'id':0,'name':'fixture','dtype':'BF16','shape':[2,3],'shard':self.path.name,'byte_offset':8+len(header)}
        self.model = {'source_directory':str(self.root),'source_identity':'trusted-fixture','catalog':[self.tensor]}
        self.data = {'tensor':0,'region':{'row':0,'col':0,'rows':2,'cols':3},'seed':1,'svd':False}
        self.clock = Mock(return_value=10.)
        self.busy = Mock(return_value=False)
        self.reap = Mock(return_value=True)
        self.fetch = Mock(return_value=self.model)
        self.jobs = AnalyticsJobs('unused-python', self.root, inference_busy=self.busy, fetch_model=self.fetch,
                                  available=lambda:8*1024**3,reap=self.reap,clock=self.clock)
        self.fake = SimpleNamespace(pid=-999, stdout=SimpleNamespace(fileno=lambda:123, close=Mock()))

    def tearDown(self): self.temp.cleanup()

    def payload(self):
        return {'root':str(self.root),'model':self.model,'fingerprints':{self.path.name:fingerprint(self.path.stat())},'request':self.data}

    def start(self):
        with patch('analytics.service.subprocess.Popen',return_value=self.fake) as popen, patch('analytics.service.os.set_blocking'), patch('analytics.service.shutil.disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
            result = self.jobs.start(self.data)
            self.assertEqual(popen.call_count,1)
        return result

    def test_real_adapter_function_on_tiny_source(self):
        result = analyze_request(self.payload())
        self.assertEqual([r['mean_abs'] for r in result['original']['rows']],[1,5])
        self.assertFalse(result['heads']['available'])
        self.assertEqual(result['tensor_id'],0)
        self.assertEqual(result['host_source_identity'],'trusted-fixture')

    def test_changed_header_or_admission_identity_rejected(self):
        data=self.payload(); data['model']={**self.model,'catalog':[{**self.tensor,'shape':[3,2]}]}
        with self.assertRaisesRegex(ValueError,'Header layout'): catalog_from_payload(data)
        data=self.payload(); data['fingerprints'][self.path.name]=[0]*5
        with self.assertRaisesRegex(ValueError,'identity changed'): catalog_from_payload(data)

    def test_mutual_exclusion_before_source_lookup(self):
        self.busy.return_value=True
        with self.assertRaisesRegex(ValueError,'compute busy'): self.jobs.start(self.data)
        self.fetch.assert_not_called()
        session=live.Session('unused',self.root); session.analytics=self.jobs
        self.jobs.status='stopping'
        with patch.object(live,'verify_model') as verify:
            with self.assertRaisesRegex(ValueError,'Analysis owns'): session.start({'prompt':'x'})
            verify.assert_not_called()

    def test_single_admission_and_capability_privacy(self):
        snapshot=self.start()
        self.assertEqual(snapshot['status'],'running')
        self.assertTrue(self.jobs.owns(snapshot['job']))
        self.assertFalse(self.jobs.owns('other'))
        self.assertNotIn('job',self.jobs.metadata()); self.assertNotIn('result',self.jobs.metadata())
        with patch('analytics.service.os.read',side_effect=BlockingIOError):
            with self.assertRaisesRegex(ValueError,'compute busy'): self.jobs.start(self.data)

    def test_reap_failure_retains_ownership_and_excludes_replacement(self):
        self.start(); self.reap.return_value=False
        self.assertFalse(self.jobs.stop())
        self.assertEqual(self.jobs.status,'stopping'); self.assertIs(self.jobs.process,self.fake)
        self.assertTrue(self.jobs.busy)
        self.reap.return_value=True; self.jobs.tick()
        self.assertFalse(self.jobs.busy); self.assertEqual(self.jobs.status,'cancelled')

    def test_completion_reaps_before_publishing_result(self):
        self.start(); event=json.dumps({'ok':True,'result':{'test':1}}).encode()+b'\n'
        self.reap.return_value=False
        with patch('analytics.service.os.read',return_value=event): self.jobs.tick()
        self.assertEqual(self.jobs.status,'stopping'); self.assertIsNone(self.jobs.snapshot()['result'])
        self.reap.return_value=True; self.jobs.tick()
        self.assertEqual(self.jobs.status,'complete'); self.assertEqual(self.jobs.snapshot()['result'],{'test':1})
        self.assertFalse(self.jobs.busy)

    def test_time_memory_and_output_limits(self):
        for failure in ['time','memory','output']:
            self.jobs.status='idle'; self.start()
            self.clock.return_value=16. if failure=='time' else 10.
            self.jobs.available=lambda:3*1024**3 if failure=='memory' else 8*1024**3
            if failure=='output': self.jobs.buffer=bytearray(b'x'*(2*1024**2))
            with patch('analytics.service.os.read',return_value=b'x'): self.jobs.tick()
            self.assertIsNone(self.jobs.process); self.assertTrue(self.jobs.error)
            self.clock.return_value=10.; self.jobs.available=lambda:8*1024**3

    def test_index_appearance_after_admission_is_rejected(self):
        payload = self.payload()
        (self.root/'model.safetensors.index.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'identity changed'):
            catalog_from_payload(payload)

    def test_request_limits_fail_before_process_spawn(self):
        for data in [{**self.data,'root':'/other'}, {**self.data,'seed':True}, {**self.data,'svd':1},
                     {**self.data,'tensor':512}, {'scope':'model','svd':True}]:
            with self.assertRaises(ValueError): validate_request(data,[self.tensor])
        self.fetch.side_effect=[self.model,{**self.model,'source_identity':'changed'}]
        with patch('analytics.service.subprocess.Popen') as spawn, patch('analytics.service.shutil.disk_usage',return_value=SimpleNamespace(free=30*1024**3)):
            with self.assertRaisesRegex(ValueError,'identity changed'): self.jobs.start(self.data)
            spawn.assert_not_called()

    def handler(self, action, data, extra=None):
        handler=live.Handler.__new__(live.Handler)
        session=live.Session('unused',self.root); session.analytics=self.jobs
        handler.server=SimpleNamespace(server_port=8796,session=session,analytics=self.jobs)
        handler.command='POST'; handler.path='/api/analytics/'+action
        raw=json.dumps(data).encode(); handler.rfile=io.BytesIO(raw); handler.headers=Message()
        for key,value in {'Host':'127.0.0.1:8796','Origin':'http://127.0.0.1:8796','X-Atlas-Local':'1','Content-Length':str(len(raw)),**(extra or {})}.items(): handler.headers[key]=value
        handler.send=Mock()
        return handler

    def test_existing_origin_and_local_header_gate_before_routing(self):
        for extra in [{'Origin':'http://other.invalid'}, {'X-Atlas-Local':'0'}, {'Host':'other.invalid'}]:
            handler=self.handler('start',self.data,extra)
            with patch.object(self.jobs,'start') as start:
                handler.handle_action(); start.assert_not_called()
                self.assertEqual(handler.send.call_args.args[0],400)

    def test_poll_cancel_require_independent_job_capability(self):
        self.start()
        for action in ['poll','cancel']:
            handler=self.handler(action,{'job':'wrong'})
            with patch.object(self.jobs,'stop') as stop, patch('analytics.service.os.read',side_effect=BlockingIOError):
                handler.handle_action(); stop.assert_not_called()
                self.assertEqual(handler.send.call_args.args[0],409)

    def test_worker_uses_kernel_timer_and_never_raises_inherited_limits(self):
        from analytics.worker import configure
        import resource
        import signal
        with patch('analytics.worker.os.sched_getaffinity',return_value={3,4}), patch('analytics.worker.os.sched_setaffinity') as affinity, patch('analytics.worker.os.nice'), patch('analytics.worker.resource.getrlimit',return_value=(2,3)), patch('analytics.worker.resource.setrlimit') as limits, patch('analytics.worker.Path.read_text',return_value='MemAvailable: 8388608 kB'), patch('signal.signal') as handler, patch('signal.alarm') as alarm:
            configure()
            affinity.assert_called_once_with(0,{3})
            alarm.assert_called_once_with(5)
            handler.assert_called_once_with(signal.SIGALRM,signal.SIG_DFL)
            self.assertEqual([c.args for c in limits.call_args_list],[(resource.RLIMIT_AS,(2,2)),(resource.RLIMIT_CPU,(2,2))])

    def test_analytics_only_cannot_launch_inference(self):
        session=live.Session('unused',self.root); session.inference_enabled=False
        with patch.object(live,'verify_model') as verify:
            with self.assertRaisesRegex(ValueError,'disabled'): session.start({'prompt':'x'})
            verify.assert_not_called()

    def test_only_verified_inference_mode_advertises_source_mapping(self):
        for enabled in (False, True):
            handler=self.handler('start', self.data)
            handler.command='GET'; handler.path='/api/model'
            handler.headers.replace_header('Content-Length', '0')
            handler.server.atlas_port=8797
            handler.server.session.inference_enabled=enabled
            metadata={'source_directory':str(self.root),'revision':live.MANIFEST['revision']}
            with patch.object(live,'proxy_request',return_value=(200,json.dumps(metadata).encode(),'application/json')):
                handler.handle_action()
            code, body, mime=handler.send.call_args.args
            self.assertEqual(code,200)
            result=json.loads(body)
            self.assertEqual('inference_source_model' in result,enabled)

    def test_shutdown_retains_both_owned_cleanup_obligations(self):
        session=live.Session('unused',self.root); session.analytics=self.jobs
        self.jobs.stop=Mock(return_value=False)
        session.stop=Mock(return_value=True)
        server=SimpleNamespace(server_close=Mock())
        with patch.object(live,'signal_and_reap',return_value=True), patch.object(live.time,'sleep'):
            self.assertFalse(live.cleanup_owned(session,object(),server))
        self.assertEqual(self.jobs.stop.call_count,8)
        self.assertEqual(session.stop.call_count,8)
        server.server_close.assert_called_once()


if __name__=='__main__': unittest.main()
