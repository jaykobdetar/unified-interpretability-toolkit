"""Pure error-boundary and source-assumption checks; no OS worker/thread/socket."""
import ast
from contextlib import ExitStack
import errno,json
from pathlib import Path
import sys,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atlas_host.profile_os import ProcessBook,ThreadMeter,proc_stat
from atlas_host.profile_observation import SpawnObservationPending
from atlas_host.startup_diagnostics import StartupDiagnostics,diagnose,error_chain
import profile_runtime_doubles as doubles

class Internals(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for target in ('subprocess.Popen','socket.socketpair','threading.Thread.start','os.pidfd_open',
                       'os.kill','signal.pidfd_send_signal','os.memfd_create'):
            self.stack.enter_context(patch(target,side_effect=AssertionError('Live work forbidden')))
        self.d=StartupDiagnostics()
    def book(self,**overrides):
        opts={'stats':lambda pid:{'start':1,'cpu':0.,'rss':1024},
              'descendants':lambda pid:set(),'available':lambda:6*1024**3,'diagnostics':self.d}
        return ProcessBook(**{**opts,**overrides})
    def test_probe_exception_keeps_cause_and_sentinel_is_explicitly_uncertain(self):
        book=self.book()
        def broken(pid):
            try:raise PermissionError(errno.EPERM,'private-token','/owner/secret')
            except PermissionError as cause:raise ValueError('private-wrapper') from cause
        book.stats=broken
        result=book.resources()
        self.assertEqual(result,{'rss_bytes':768*1024**2+1,'available_bytes':0,
                                'all_owned_accounted':False,'descendants_clear':False})
        event=self.d.snapshot()['events'][0]
        self.assertEqual(event['site'],'profile_os.resource_observation_failed')
        self.assertEqual(event['error_chain']['exceptions'][1]['errno'],errno.EPERM)
        self.assertNotIn('private',json.dumps(event));self.assertNotIn('/owner',json.dumps(event))
    def test_known_spawn_interleaving_retains_pending_ownership_and_measured_rss(self):
        book=self.book();book.spawning=True;book.spawn_started=0.;book.clock=lambda:0.
        book.descendants=lambda pid:{42} if pid==book.root else set()
        with self.assertRaises(SpawnObservationPending) as captured:book.resources()
        observed=captured.exception.observed
        self.assertFalse(observed['all_owned_accounted'])
        self.assertEqual(observed['rss_bytes'],2048)
        self.assertEqual(observed['available_bytes'],6*1024**3)
        self.assertTrue(self.d.snapshot()['events'][0]['facts']['spawning'])
        self.assertFalse(book.settled())
    def test_real_memory_measurement_is_preserved_without_exception_event(self):
        book=self.book(stats=lambda pid:{'start':1,'cpu':0.,'rss':800*1024**2})
        result=book.resources()
        self.assertEqual(result['rss_bytes'],800*1024**2)
        self.assertTrue(result['all_owned_accounted']);self.assertTrue(result['descendants_clear'])
        self.assertEqual(self.d.snapshot()['events'],[])
    def test_event_byte_count_limit_deduplication_and_private_fields(self):
        error=ValueError('capability-secret')
        for _ in range(10):self.d.event('test.error',error,tab_capability='secret',rss_bytes=10)
        snapshot=self.d.snapshot();self.assertEqual(len(snapshot['events']),1)
        self.assertEqual(snapshot['events'][0]['count'],10)
        self.assertEqual(snapshot['events'][0]['facts'],{'rss_bytes':10})
        for n in range(100):self.d.event('test.error'+str(n),error)
        snapshot=self.d.snapshot();self.assertEqual(len(snapshot['events']),64)
        self.assertGreater(snapshot['events_dropped'],0)
        self.assertLessEqual(snapshot['event_bytes'],512*1024)
        self.assertNotIn('capability',json.dumps(snapshot))
    def test_thread_meter_reads_registered_clock_and_freezes_before_context_exit(self):
        meter=ThreadMeter(diagnostics=self.d)
        with patch('time.pthread_getcpuclockid',return_value=77),patch('threading.get_ident',return_value=42),patch('time.clock_gettime',side_effect=[.1,.2]) as read:
            meter.register();self.assertEqual(meter.read(),.1);meter.freeze()
            self.assertEqual(meter.read(),.2);self.assertEqual([c.args for c in read.call_args_list],[(77,),(77,)])
        unregistered=ThreadMeter(diagnostics=self.d)
        with self.assertRaises(ValueError):unregistered.read()
        self.assertIn('CPU context unavailable',self.d.snapshot()['events'][0]['error_chain']['exceptions'][0]['message'])
    def test_proc_stat_indexes_and_spaces_in_command_name(self):
        fields=['0']*22;fields[11]='10';fields[12]='5';fields[19]='123';fields[21]='7'
        with patch('pathlib.Path.read_text',return_value='42 (worker name (nested)) '+' '.join(fields)),patch('os.sysconf',side_effect=lambda key:100 if key=='SC_CLK_TCK' else 4096):
            self.assertEqual(proc_stat(42),{'start':123,'cpu':.15,'rss':28672})
    def test_all_existing_profile_reduction_catches_record_before_other_actions(self):
        root=Path(__file__).resolve().parents[1]/'tools/atlas_host'
        count=0
        for module in ('profile_worker','profile_platform','profile_service','profile_snapshot'):
            tree=ast.parse((root/(module+'.py')).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ExceptHandler) and isinstance(node.type,ast.Name) and node.type.id=='BaseException':
                    count+=1;first=node.body[0]
                    self.assertIsInstance(first,ast.Expr)
                    self.assertIsInstance(first.value,ast.Call)
                    self.assertEqual(first.value.func.id,'diagnose')
        self.assertEqual(count,17)
    def test_worker_fd_receipt_omits_binding_paths_and_capabilities(self):
        argv=['/private/bin','profile-worker']
        for key in ('model','revision','tensor','slice','seed','values','wall-ms','cpu-ms','binding'):
            argv += ['--'+key,'private-secret']
        argv += ['--output-fd','19']
        with patch('fcntl.fcntl',return_value=1),patch('os.get_inheritable',return_value=False),patch('os.fstat',return_value=types.SimpleNamespace(st_mode=0o100600)):
            record=self.d.prepare(argv,(19,))
        receipt=record.snapshot()
        self.assertEqual(receipt['passed_fds'][0]['role'],'profile output memfd')
        self.assertFalse(receipt['passed_fds'][0]['parent_is_socket'])
        self.assertEqual(receipt['argv'][-2:],['--output-fd','19'])
        self.assertNotIn('private-secret',json.dumps(receipt));self.assertNotIn('/private/bin',json.dumps(receipt))

    def test_runtime_error_type_is_preserved(self):
        self.assertEqual(error_chain(RuntimeError('private-secret'))['exceptions'][0]['type'],'RuntimeError')

class ServiceReduction(unittest.TestCase):
    def test_inner_platform_failure_survives_fixed_public_error(self):
        case=doubles.ServiceTests('test_second_start_never_queues');case.setUp()
        self.addCleanup(case.doCleanups)
        d=StartupDiagnostics();case.service.diagnostics=d;case.service.watchdog.diagnostics=d
        original=case.service.hooks_factory
        def factory(source,slot):
            h=original(source,slot);h.diagnostics=d
            def fail():
                try:raise PermissionError(errno.EACCES,'capability-secret','/owner/private')
                except PermissionError as cause:raise ValueError('private-wrapper') from cause
            h.start_gate=fail
            return h
        case.service.hooks_factory=factory
        grant,_=case.start();case.service.finish_admission(grant);case.service.step()
        status=case.service.status(case.owner)
        self.assertEqual(status['error'],'runtime_error')
        self.assertNotIn('error_chain',status)
        events=d.snapshot()['events'];self.assertGreaterEqual(len(events),3)
        self.assertTrue(any(any(n['errno']==errno.EACCES for n in e.get('error_chain',{}).get('exceptions',[])) for e in events))
        for text in ('capability-secret','/owner/private','private-wrapper'):
            self.assertNotIn(text,json.dumps(d.snapshot()));self.assertNotIn(text,json.dumps(status))

if __name__=='__main__':unittest.main(verbosity=2)
