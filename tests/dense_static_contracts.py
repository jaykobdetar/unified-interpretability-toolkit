"""Inert policy artifacts, tiny synthetic metadata, fake clocks/handles. No live jobs."""
from copy import deepcopy
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).parent)]
from atlas_host.common import canonical
from atlas_host import dense_static_admission as policy
from atlas_host.hosted_runtime import HostedApplication,NativeChannel,StaticOperation
from atlas_host.profile_http import HostedHandler
from atlas_host.profile_service import WatchdogLoop
from atlas_host.runtime_adapter import FixtureHost,dispatch
from atlas_host.supervisor import AdmissionGrant,CpuLedger,Supervisor
from profile_runtime_doubles import Socket,Meter
import static_model_contracts as tiny
import static_atlas


def H(v):return hashlib.sha256(canonical(v)).hexdigest()

def skeleton(schema):
    if '$ref' in schema:return skeleton(policy.SCHEMAS[schema['$ref'].split('/')[-1]])
    if 'const' in schema:return deepcopy(schema['const'])
    t=schema['type']
    if t=='object':return {k:skeleton(v) for k,v in schema['properties'].items()}
    if t=='array':return [skeleton(schema['items']) for _ in range(schema['minItems'])]
    if t=='integer':return schema.get('minimum',0)
    if t=='string':
        if schema.get('pattern')=='^[0-9a-f]{64}$':return 'a'*64
        if schema.get('pattern')=='^[0-9a-f]{40}$':return 'a'*40
        return 'x'*max(1,schema.get('minLength',1))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.fs=skeleton(policy.SCHEMAS['filesystemReceipt'])
        self.fs['platform']['python_version']='3.14.7'
        self.roots={}
        for role in ('source','cache'):
            root={'canonical_root':str(self.root/role),'device':1,'directory_inode':10,'filesystem_type':'ext4'}
            self.roots[root['canonical_root']]=root;self.fs[role]=root
            probe=self.fs['observations'][role];scratch=dict(root,canonical_root=str(self.root/('scratch-'+role)),directory_inode=11)
            probe['probe_root']=scratch;self.roots[scratch['canonical_root']]=scratch
            before={'device':1,'inode':20,'bytes':8,'mtime_ns':1,'ctime_ns':1}
            probe['open_identity']={'before':before,'opened':before,'after_fd':before,'after_path':before,'bytes_read':8,'expected_sha256':'a'*64,'readback_sha256':'a'*64}
            probe['same_length_write']={'before':before,'after':dict(before,ctime_ns=2),'before_bytes_sha256':'a'*64,'after_bytes_sha256':'b'*64}
            replacement=dict(before,inode=21)
            probe['ordinary_replace']={'before':before,'replacement':replacement,'after':dict(replacement,ctime_ns=2),'expected_sha256':'a'*64,'readback_sha256':'a'*64}
            if role=='cache':probe['atomic_publish']={'before':before,'temporary':replacement,'after':dict(replacement,ctime_ns=2),'expected_sha256':'a'*64,'readback_sha256':'a'*64,'temporary_remaining':False}
        self.recipe=skeleton(policy.SCHEMAS['sourceRecipe'])
        self.recipe['runtime_inventory']=[{'path':'only.py','bytes':8,'sha256':'a'*64}]
        self.entry={'enabled':True,'model_id':'m_'+'a'*64,'root':self.fs['source']['canonical_root'],
            'manifest':{'version':1,'repository':policy.REPOSITORY,'revision':policy.REVISION,'provenance':'owner_expected',
              'license':{'id':'Apache-2.0','file':'LICENSE','accepted':True},'files':[
                  {'name':'config.json','bytes':704,'sha256':policy.CONFIG_SHA},
                  {'name':'model.safetensors','bytes':269060552,'sha256':policy.WEIGHTS_SHA},
                  {'name':'LICENSE','bytes':8,'sha256':'b'*64}]}}
        # These are dictionary mocks, not registration/license or filesystem evidence.
        self.registry=NS(owner_receipt=Mock(side_effect=lambda _:deepcopy(self.entry)))
        self.addCleanup(patch.stopall)
        patch.object(policy,'_root_identity',side_effect=lambda p:deepcopy(self.roots[str(p)])).start()
        patch.object(policy,'_platform',return_value=self.fs['platform']).start()

    def test_closed_supporting_schemas_and_semantics(self):
        policy.filesystem_checks(self.fs,self.fs['source']['canonical_root'],self.fs['cache']['canonical_root'])
        policy.validate_support(self.recipe,'sourceRecipe')
        for kind,value in [('filesystemReceipt',self.fs),('sourceRecipe',self.recipe)]:
            for key in list(value):
                altered=deepcopy(value);altered.pop(key)
                with self.assertRaises(ValueError):policy.validate_support(altered,kind)
            altered=deepcopy(value);altered['approved']=True
            with self.assertRaises(ValueError):policy.validate_support(altered,kind)
        for value in (True,1.,'1'):
            altered=deepcopy(self.fs);altered['version']=value
            with self.assertRaises(ValueError):policy.validate_support(altered,'filesystemReceipt')
        altered=deepcopy(self.recipe);altered['smol_qualification_sha256']='a'*64
        with self.assertRaises(ValueError):policy.validate_support(altered,'sourceRecipe')

    def test_filesystem_root_device_platform_and_observations_refuse(self):
        for alter in [lambda v:v['source'].update(device=2),lambda v:v['cache'].update(filesystem_type='fuse'),
                      lambda v:v['platform'].update(kernel_release='wrong'),
                      lambda v:v['observations']['source']['open_identity']['after_fd'].update(inode=99),
                      lambda v:v['observations']['cache']['atomic_publish'].update(temporary_remaining=True),
                      lambda v:v['observations']['source']['same_length_write'].update(after_bytes_sha256='a'*64)]:
            bad=deepcopy(self.fs);alter(bad)
            with self.assertRaises(ValueError):policy.filesystem_checks(bad,self.fs['source']['canonical_root'],self.fs['cache']['canonical_root'])

    def bundle(self):
        (self.root/'only.py').write_bytes(b'source00');binary=self.root/'reader.data';binary.write_bytes(b'not-code')
        (self.root/'receipts').mkdir()
        self.recipe.update(runtime_head='a'*40,runtime_tree='b'*40,prior_tiny_native_evidence_sha256=policy.PRIOR_TINY_SHA)
        self.recipe['runtime_inventory']=[{'path':'only.py','bytes':8,'sha256':hashlib.sha256(b'source00').hexdigest()}]
        self.recipe['pure_evidence']['source_inventory_sha256']=H(self.recipe['runtime_inventory'])
        self.recipe['native_binary'].update(sha256=hashlib.sha256(b'not-code').hexdigest(),bytes=8)
        for v in (self.fs,self.recipe):(self.root/'receipts'/(H(v)+'.json')).write_bytes(canonical(v))
        owner={'version':1,'policy':'pinned_smol_dense_static_v1','enabled':True,'model_id':self.entry['model_id'],
               'owner_receipt_sha256':H(self.entry),'filesystem_receipt_sha256':H(self.fs),'source_recipe_sha256':H(self.recipe),
               'native_binary_sha256':self.recipe['native_binary']['sha256']}
        file=self.root/'policy.json';file.write_bytes(canonical(owner))
        patch.object(policy,'package_root',return_value=self.root).start()
        patch.object(policy,'runtime_names',return_value=['only.py']).start()
        patch.object(policy,'_git_version',return_value=['a'*40,'b'*40]).start()
        patch.object(policy,'_current',return_value=None).start()
        patch.object(policy,'QUALIFIED_READER_SHA',self.recipe['native_binary']['sha256']).start()
        patch.object(policy,'NATIVE_SOURCE_PINS',{'only.py':self.recipe['runtime_inventory'][0]['sha256']}).start()
        return file,owner,binary

    def test_exact_owner_bundle_binds_and_changes_refuse(self):
        file,owner,binary=self.bundle()
        bound=policy.bind_dense_policy(file,H(owner),self.registry,self.fs['cache']['canonical_root'],binary)
        self.assertTrue(bound.allows(self.entry['model_id']));self.assertFalse(bound.allows('m_'+'b'*64))
        self.assertEqual(bound.check(),self.entry)
        bound.check_binary(NS(path=binary,check=lambda:None))
        with self.assertRaises(ValueError):bound.check_binary(NS(path=self.root/'only.py',check=lambda:None))
        self.entry['verified_at']='changed full bookkeeping'
        self.assertFalse(bound.allows(self.entry['model_id']))
        with self.assertRaises(ValueError):bound.check()

    def test_missing_approval_noncanonical_digest_and_source_binary_refuse(self):
        file,owner,binary=self.bundle()
        for approved in (None,'0'*64):
            with self.assertRaises(ValueError):policy.bind_dense_policy(file,approved,self.registry,self.fs['cache']['canonical_root'],binary)
        file.write_bytes(canonical(owner)+b'\n')
        with self.assertRaises(ValueError):policy.bind_dense_policy(file,hashlib.sha256(file.read_bytes()).hexdigest(),self.registry,self.fs['cache']['canonical_root'],binary)
        file.write_bytes(canonical(owner));binary.write_bytes(b'different')
        with self.assertRaises(ValueError):policy.bind_dense_policy(file,H(owner),self.registry,self.fs['cache']['canonical_root'],binary)

    def test_pin_and_license_dictionaries_do_not_create_authority(self):
        policy.target_entry(self.entry)
        for change in [lambda e:e.update(enabled=False),lambda e:e['manifest'].update(revision='b'*40),
                       lambda e:e['manifest']['license'].update(accepted=False),
                       lambda e:e['manifest']['files'][1].update(sha256='b'*64)]:
            bad=deepcopy(self.entry);change(bad)
            with self.assertRaises(ValueError):policy.target_entry(bad)
        unsealed=object.__new__(policy.BoundDenseStaticAdmission);object.__setattr__(unsealed,'_seal',None)
        with self.assertRaises(ValueError):unsealed.check()


class CachePreflightTests(unittest.TestCase):
    def test_existing_native_lock_is_checked_without_writes_and_busy_refused(self):
        with tempfile.TemporaryDirectory() as t:
            cache=Path(t);lock=cache/'atlas.lock';lock.write_bytes(b'inert')
            before=lock.stat()
            with patch.object(policy.fcntl,'flock') as flock:policy.check_cache_available(cache)
            self.assertEqual([call.args[1] for call in flock.call_args_list],
                             [policy.fcntl.LOCK_EX|policy.fcntl.LOCK_NB,policy.fcntl.LOCK_UN])
            self.assertEqual(before.st_mtime_ns,lock.stat().st_mtime_ns)
            with patch.object(policy.fcntl,'flock',side_effect=BlockingIOError()):
                with self.assertRaises(ValueError):policy.check_cache_available(cache)


class MountParsingTests(unittest.TestCase):
    def test_bounded_local_mount_metadata_and_octal_path_encoding(self):
        with tempfile.TemporaryDirectory(prefix='atlas mount ') as temp:
            root=Path(temp)
            encoded=str(root).replace(' ',r'\040')
            raw=('1 0 0:1 / / rw - ext4 /dev/root rw\n'
                 +'2 1 0:2 / '+encoded+' rw - ext4 /dev/local rw\n').encode()
            with patch.object(Path,'open',return_value=io.BytesIO(raw)):
                identity=policy._root_identity(root)
            self.assertEqual(identity['canonical_root'],str(root));self.assertEqual(identity['filesystem_type'],'ext4')
            with patch.object(Path,'open',return_value=io.BytesIO(b'x'*(1024**2+1))):
                with self.assertRaises(ValueError):policy._root_identity(root)


class Child:
    def __init__(self):self.cpu=0.;self.reaped=False;self.unexpected=set();self.stopped=False
    def sample(self):return {'cpu_seconds':self.cpu,'reaped':self.reaped}
    def stop(self):self.stopped=True


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.child=Child();self.supervisor=Supervisor();self.watch=WatchdogLoop(Meter(),clock=lambda:self.now)
        self.host=NS(reader=None,entry=None,static_prepared=None,static_bound=None,cache_root=Path('/tmp'),stopping=False,view_kind='static',context='ctx',leases={'lease':15},clock=lambda:self.now)
        self.host._revoke_readiness=Mock()
        self.book=NS(resources=lambda:{'rss_bytes':100,'available_bytes':6*1024**3,'all_owned_accounted':True,'descendants_clear':True},settled=lambda:self.child.reaped)
        self.service=NS(admission=Mock(side_effect=AssertionError('No nested grant')),watchdog=self.watch)
        self.app=NS(host=self.host,supervisor=self.supervisor,book=self.book,profiles=self.service)
        patcher=patch('atlas_host.hosted_runtime.shutil.disk_usage',return_value=NS(free=100*1024**3));patcher.start();self.addCleanup(patcher.stop)
    def grant(self):return AdmissionGrant(lambda:self.now,CpuLedger({'admission':lambda:0.,'owner':lambda:0.,'watchdog':lambda:0.}))
    def operation(self,kind='metadata'):
        grant=self.grant();op=StaticOperation(self.app,grant,kind)
        self.channel=NativeChannel(Socket(),self.child,self.supervisor,self.service,self.book,clock=lambda:self.now,wait=lambda *args:None)
        self.reader=NS(child=self.child,channel=self.channel)
        op.add_reader(self.reader);op.mutated=True
        return op
    def test_native_ack_retains_original_grant_token_watch_until_publication(self):
        op=self.operation();grant=op.grant;self.child.cpu=.2
        self.assertEqual(self.channel.read('/api/model','ctx',operation=op),(200,b'{}','application/json'))
        self.assertTrue(self.supervisor.busy());self.assertFalse(op.slot.closed)
        self.assertEqual(self.channel.settled_cpu,0.);self.service.admission.assert_not_called()
        writes=[];op.publish(lambda:writes.append(op.token.current()))
        self.assertEqual(writes,[True]);self.assertIs(op.grant,grant)
        self.assertFalse(self.supervisor.busy());self.assertTrue(op.slot.closed);self.assertEqual(self.channel.settled_cpu,.2)
    def test_route_specific_native_kind_preserved(self):
        for path,kind in [('/api/inspect','inspect'),('/tile','tile'),('/api/calibrate','calibration')]:
            op=self.operation(kind);self.channel.read(path,'ctx',operation=op)
            self.assertEqual(json.loads(self.channel.stream.sent[4:])['kind'],kind)
            op.publish(lambda:None)
        op=self.operation('metadata')
        with self.assertRaises(ValueError):self.channel.read('/api/inspect','ctx',operation=op)
        self.assertFalse(self.channel.stream.sent);self.child.reaped=True;op.abort()
    def test_post_ack_expiry_publishes_nothing_and_retains_until_reap(self):
        op=self.operation();self.channel.read('/api/model','ctx',operation=op);self.now=6
        write=Mock()
        with self.assertRaises(ValueError):op.publish(write)
        write.assert_not_called();self.assertTrue(self.supervisor.busy());self.assertTrue(self.child.stopped)
        self.assertEqual(self.channel.settled_cpu,0.)
        self.child.reaped=True;self.watch.pulse();self.assertFalse(self.supervisor.busy());self.assertTrue(op.finished)
    def test_precommand_expiry_never_sends_and_cpu_overrun_after_write_revokes(self):
        op=self.operation();self.now=6
        with self.assertRaises(ValueError):self.channel.read('/api/model','ctx',operation=op)
        self.assertFalse(self.channel.stream.sent);self.child.reaped=True;op.abort()
        self.now=0.;self.child=Child();op=self.operation();self.host.reader=self.reader
        self.channel.read('/api/model','ctx',operation=op)
        with self.assertRaises(ValueError):op.publish(lambda:setattr(self.child,'cpu',4.1))
        self.assertTrue(self.host.stopping);self.host._revoke_readiness.assert_called();self.assertEqual(self.channel.settled_cpu,0.)
    def test_old_cleanup_cpu_series_is_not_dropped_when_new_child_is_added(self):
        op=self.operation();self.channel.settled_cpu=.1;op.children[0][1:]=[.1,.1];self.child.cpu=.3;self.child.reaped=True
        new=Child();new.cpu=.2;op.add_child(new);op.check()
        self.assertAlmostEqual(op.grant.child_cpu,.4)
    def test_settled_watermark_carries_idle_debt_and_catalog_cannot_reset_it(self):
        op=self.operation();self.child.cpu=.2;self.channel.read('/api/model','ctx',operation=op);op.publish(lambda:None)
        self.child.cpu=.7;self.channel.idle_pulse();self.assertEqual(self.channel.settled_cpu,.2)
        self.host.reader=self.reader;next_op=StaticOperation(self.app,self.grant(),'metadata')
        self.assertAlmostEqual(next_op.grant.child_cpu,.5)
        next_op.publish(lambda:None);self.assertEqual(self.channel.settled_cpu,.7)
        self.child.cpu=4.7
        with self.assertRaises(ValueError):self.channel.idle_pulse()
    def test_cpu_regression_refuses_and_uncertain_cleanup_keeps_slot(self):
        op=self.operation();self.child.cpu=.2;op.check();self.child.cpu=.1
        with self.assertRaises(ValueError):op.check()
        self.assertFalse(op.abort());self.assertTrue(self.supervisor.busy())
        self.child.reaped=True;self.watch.pulse();self.assertTrue(op.finished)
    def test_factory_recognizes_only_its_own_startup_metadata_reservation(self):
        op=self.operation();op.children.clear();op.channels.clear();op.mutated=False
        app=HostedApplication.__new__(HostedApplication)
        app.lifetime=NS(stopping=False);app.supervisor=self.supervisor;app.book=NS(settled=lambda:True)
        app.profiles=self.service;app.host=self.host;app.diagnostics=None;app.binary=NS(check=lambda:None)
        app.dense_policy=object.__new__(policy.BoundDenseStaticAdmission)
        op.app=app;app.book.resources=self.book.resources
        prepared=NS(entry={'root':'mock'},check=lambda:None);op.prepared=prepared
        with patch.object(policy.BoundDenseStaticAdmission,'admit',return_value=True),patch.object(policy.BoundDenseStaticAdmission,'check_binary'),patch('atlas_host.hosted_runtime.check_cache_available'),patch('atlas_host.hosted_runtime.available_bytes',return_value=6*1024**3) as available,patch('atlas_host.hosted_runtime.HostedRenderer') as spawn:
            app._renderer(prepared.entry,Path('/tmp/cache'),prepared=prepared,operation=op)
            self.assertIs(spawn.call_args.kwargs['operation'],op)
            prepared.entry['model_id']='m_'+'a'*64;app.static_operations=[op]
            self.assertTrue(app._admit_static(prepared))
            available.return_value=4*1024**3
            with self.assertRaises(ValueError):app._renderer(prepared.entry,Path('/tmp/cache'),prepared=prepared,operation=op)
            available.return_value=6*1024**3;app.book.settled=lambda:False
            with self.assertRaises(ValueError):app._renderer(prepared.entry,Path('/tmp/cache'),prepared=prepared,operation=op)
            op.token.kind='inspect'
            with self.assertRaises(ValueError):app._renderer(prepared.entry,Path('/tmp/cache'),prepared=prepared,operation=op)
            self.assertEqual(spawn.call_count,1)

    def test_final_watch_join_failure_keeps_cleanup_owned_by_lifetime(self):
        op=self.operation();self.channel.read('/api/model','ctx',operation=op)
        old_close=op.slot.close
        def late_close():old_close();self.now=6
        op.slot.close=late_close
        with self.assertRaises(ValueError):op.publish(lambda:None)
        self.assertTrue(op.slot.closed);self.assertTrue(self.supervisor.busy())
        self.child.reaped=True
        app=HostedApplication.__new__(HostedApplication);app.lifetime=NS(pulse=lambda:None)
        app.dense_policy=object();app.host=self.host;app.supervisor=self.supervisor;app.static_operations=[op]
        app._lifetime_pulse();self.assertFalse(self.supervisor.busy());self.assertTrue(op.finished)
    def test_stop_error_revokes_before_retaining_uncertain_slot(self):
        op=self.operation();self.host.reader=self.reader
        self.child.stop=Mock(side_effect=OSError('inert stop error'))
        self.assertFalse(op.abort());self.assertTrue(self.host.stopping)
        self.host._revoke_readiness.assert_called();self.assertTrue(self.supervisor.busy())
    def test_prospective_source_refusal_does_not_stop_valid_old_reader(self):
        op=self.operation();op.mutated=False
        op.prepared=NS(check=Mock(side_effect=ValueError('unadmitted prospective source')))
        self.watch.pulse();self.assertFalse(op.failed)
        self.assertTrue(op.abort());self.assertFalse(self.child.stopped)
        op.prepared.check.assert_not_called()
    def test_constructor_failure_retains_operation_for_independent_cleanup(self):
        self.host.reader=NS(channel=NS(child=self.child,settled_cpu=0.,stream=NS(close=lambda:None)))
        self.child.cpu=5.;self.app.static_operations=[]
        with self.assertRaises(ValueError):StaticOperation(self.app,self.grant(),'metadata')
        self.assertEqual(len(self.app.static_operations),1);self.assertTrue(self.supervisor.busy())
        self.child.reaped=True;self.watch.pulse();self.assertFalse(self.supervisor.busy())


class AdapterAndHttpTests(unittest.TestCase):
    def test_hook_forwards_original_grant_and_default_preserves_dispatch(self):
        grant=object();operation=object()
        app=NS(host=NS(view_kind='static',registry=NS(owner_receipt=lambda _: {'manifest':{'provenance':'owner_expected'}})),dense_policy=NS(allows=lambda _:True),begin_static=Mock(return_value=operation))
        h=HostedHandler.__new__(HostedHandler);h.server=NS(application=app);h.profile_admission=grant
        with patch('atlas_host.profile_http.dispatch',return_value=(200,b'{}','application/json')) as route:
            h.dispatch_host('POST','/api/view-contexts',{'model_id':'m_'+'a'*64})
            app.begin_static.assert_called_once_with(grant,'metadata');self.assertIs(route.call_args.kwargs['operation'],operation)
        del h.static_finalizer;app.begin_static.reset_mock()
        with patch('atlas_host.profile_http.dispatch'):
            h.dispatch_host('GET','/api/models/m_a/view?context=ctx');app.begin_static.assert_called_once_with(grant,'metadata')
        from host_atlas import HostHandler
        base=HostHandler.__new__(HostHandler);base.server=NS(host='fixture')
        with patch('host_atlas.dispatch') as route:
            base.dispatch_host('GET','/api/models');route.assert_called_once_with('fixture','GET','/api/models',None)
    def test_inherited_guarded_acquire_uses_dispatch_hook(self):
        from host_atlas import HostHandler
        h=HostHandler.__new__(HostHandler);h.command='POST';h.path='/api/view-contexts';h.headers=Message();h.headers['X-Atlas-Local']='1'
        h.rfile=io.BytesIO(b'{"model_id":"mock"}');h.validated=lambda:('/api/view-contexts',19);h.send=Mock()
        h.dispatch_host=Mock(return_value=(202,b'{}','application/json'))
        h.handle_action();h.dispatch_host.assert_called_once_with('POST','/api/view-contexts',{'model_id':'mock'})
    def test_launcher_default_and_pair_are_closed_without_starting(self):
        with patch.object(static_atlas,'load_config') as config,patch.object(static_atlas,'run') as run:
            static_atlas.main(['--config','mock','--binary-sha256','a'*64])
            self.assertIsNone(run.call_args.kwargs['policy_path']);self.assertIsNone(run.call_args.kwargs['approved_sha'])
            with self.assertRaises(ValueError):static_atlas.main(['--config','mock','--binary-sha256','a'*64,'--dense-static-policy','mock'])
            self.assertEqual(run.call_count,1)
    def test_initial_acquire_binds_full_model_before_active_lease_with_same_operation(self):
        case=tiny.StaticContracts();case.setUp();self.addCleanup(case.doCleanups)
        calls=[]
        class Op:
            context=None;prepared=None;mutated=False
            def check(self):calls.append('check')
        op=Op()
        class Reader:
            def initialize(self):calls.append('initialize-owned');assert host.reader is self
            def alive(self):return True
            def ready(self):return True
            def stop(self):return True
            def read(self,path,*,operation):
                assert operation is op;calls.append('native-model')
                return 200,canonical(case.native_model(op.prepared)),'application/json'
        def factory(entry,cache,*,prepared,operation):
            assert prepared is op.prepared and operation is op;calls.append('factory');return Reader()
        host=FixtureHost(case.registry,case.root/'cache',factory,static_policy=case.policy,static_admission=lambda _:True)
        lease=host.acquire({'model_id':case.identifier},operation=op)
        self.assertEqual(lease['state'],'active');self.assertIsNotNone(host.static_bound)
        self.assertLess(calls.index('initialize-owned'),calls.index('native-model'))
    def test_native_name_and_empty_saved_statistics_are_explicit(self):
        prepared=NS(entry={'name':'Owner'})
        model={'name':'Owner','fresh_source_hashes':None,'calibration_complete':False,'global_max':None,
               'coverage':{'source_complete':True,'active_tensor':None,'all_requested':False,'materialized_bytes':0,'materialized_tiles':0,'cache_budget_bytes':2*1024**3,'fine_tile_file_cap':1000,'calibrated_tensors':0,'values_streamed':0,'statistics_complete':False,'calibration_error':None,
                           **{k:0 for k in ('sha_hashed_shards','sha_expected_matched_shards','sha_missing_expected_shards','sha_verified_shards')}},
               'catalog':[{'calibration_complete':False,'rule_status':{'tensor_signed_percentile':'calibration pending'},
                           **{k:None for k in ('max_abs','median_nonzero_abs','q99','quantile_order_statistics','quantile_interpolation','robust_clipped_count','exact_zero_count','unique_bit_patterns','calibration_method')}}]}
        policy.check_native_model(prepared,model)
        for change in [lambda m:m.update(name='Other'),lambda m:m['catalog'][0].update(max_abs=1),lambda m:m.update(global_max=1),lambda m:m['coverage'].update(values_streamed=True)]:
            bad=deepcopy(model);change(bad)
            with self.assertRaises(ValueError):policy.check_native_model(prepared,bad)

    def test_valid_saved_statistics_are_nonzero_without_fresh_recompute_claim(self):
        t={'calibration_complete':True,'count':5,'max_abs':1.,'median_nonzero_abs':.5,'q99':.9,
           'robust_clipped_count':1,'exact_zero_count':0,'unique_bit_patterns':2,'calibration_method':'exact-16-bit-histogram',
           'quantile_order_statistics':'exact','quantile_interpolation':'linear in F64; final floating-point rounding possible',
           'rule_status':{'tensor_signed_percentile':'supported dtype; requires a valid exact histogram'}}
        c={'source_complete':True,'active_tensor':None,'all_requested':False,'calibration_error':None,
           'materialized_bytes':0,'materialized_tiles':0,'cache_budget_bytes':2*1024**3,'fine_tile_file_cap':1000,
           'calibrated_tensors':1,'values_streamed':5,'statistics_complete':True,
           **{k:0 for k in ('sha_hashed_shards','sha_expected_matched_shards','sha_missing_expected_shards','sha_verified_shards')}}
        model={'name':'Owner','fresh_source_hashes':None,'global_max':1.,'calibration_complete':True,'coverage':c,'catalog':[t]}
        policy.check_native_model(NS(entry={'name':'Owner'}),model)
        model['global_max']=True
        with self.assertRaises(ValueError):policy.check_native_model(NS(entry={'name':'Owner'}),model)
    def test_handler_post_write_failure_sends_no_second_response(self):
        h=HostedHandler.__new__(HostedHandler);h.connection=NS(settimeout=Mock())
        op=NS(finished=False,grant=NS(remaining=lambda:{'wall_ms':500}),check=lambda:None,abort=Mock())
        def publish(write):write();raise ValueError('inert late failure')
        op.publish=publish;h.static_finalizer=op
        from host_atlas import HostHandler
        with patch.object(HostHandler,'send') as write:
            with self.assertRaises(ValueError):h.send(200,b'{}')
            h.send(409,b'{}');self.assertEqual(write.call_count,1)
        self.assertTrue(h.close_connection);op.abort.assert_called()


if __name__=='__main__':unittest.main()
