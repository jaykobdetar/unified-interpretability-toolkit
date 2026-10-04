"""Pure PROVIDER-2 regressions. Real process/socket/FD/thread actions forbidden."""
from contextlib import ExitStack
import os,signal,socket,subprocess,threading,resource,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))

def forbidden(*args,**kwargs):raise AssertionError('Real OS action forbidden in pure regression')

with ExitStack() as guard:
    for owner,name in [(subprocess,'Popen'),(socket,'socket'),(socket,'socketpair'),(os,'memfd_create'),
        (os,'pidfd_open'),(os,'wait4'),(os,'kill'),(signal,'pidfd_send_signal'),(threading.Thread,'start'),
        (os,'sched_setaffinity'),(resource,'setrlimit')]:
        guard.enter_context(patch.object(owner,name,side_effect=forbidden))
    from profile_runtime_doubles import NativeTests
    class NativeCleanup(unittest.TestCase):
        def finish(self,t,c):
            self.assertFalse(t.s.busy())
            self.assertTrue(c.slot.closed)
            self.assertFalse(c.slot.failed)
            for _ in range(3):
                c.slot.callback();t.watch.pulse()
                self.assertTrue(c.stop())
            self.assertFalse(t.watch.slots)
            t.watch.close()
        def test_deadline_reap_before_reader_exception_does_not_rearm(self):
            t=NativeTests();c=t.make();original=c._io
            def receive(data,size,deadline):
                value=original(data,size,deadline)
                if data is None and size==2:t.clock.now=6;t.child.reaped=True
                return value
            c._io=receive
            with self.assertRaisesRegex(ValueError,'Native operation expired'):c.read('/api/model','ctx')
            self.finish(t,c)
        def test_reap_during_reader_cleanup_is_terminal_and_idempotent(self):
            t=NativeTests();c=t.make();original=c._io;calls=[]
            def receive(data,size,deadline):
                value=original(data,size,deadline)
                if data is None and size==2:t.clock.now=6
                return value
            def stop():
                calls.append(True);t.child.stopped=True
                if len(calls)==2:
                    t.child.reaped=True
                    c.slot.callback()  # Deterministic cleanup-boundary interleaving.
            c._io=receive;t.child.stop=stop
            with self.assertRaisesRegex(ValueError,'Native operation expired'):c.read('/api/model','ctx')
            self.finish(t,c)
        def test_late_guard_after_successful_ack_cannot_touch_released_token(self):
            t=NativeTests();c=t.make();c.read('/api/model','ctx')
            t.clock.now=6;t.child.reaped=True
            self.finish(t,c)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeCleanup))
raise SystemExit(not result.wasSuccessful())
