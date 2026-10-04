"""Loopback profile bridge installed only by the explicit owner profile launcher."""
from urllib.parse import urlsplit, parse_qs, urlencode
from .startup_diagnostics import diagnose
from .common import canonical, require
from .profile_os import strict_json
from .runtime_adapter import dispatch
from host_atlas import HostHandler, ROOT, BUNDLE


class HostedHandler(HostHandler):
    def handle(self):
        app=self.server.application
        grant,meter=app.profiles.admission()  # Before receive/framing/body work.
        self.profile_admission=grant
        try:
            super().handle()
        finally:
            meter.freeze()
            app.profiles.finish_admission(grant)

    def handle_action(self):
        app=self.server.application
        try:
            path,length=self.validated()
            for key in ('Host','Origin','X-Atlas-Local','Content-Type','Sec-Fetch-Site'):
                require(len(self.headers.get_all(key,[]))<=1,'Duplicate guarded header')
            if path.startswith('/api/profiles/'):
                require(getattr(self.server,'profile_controls',False) is True, 'Profile HTTP controls disabled')
                require(self.command=='POST' and self.path==path,'Private POST controls required')
                require(self.headers.get('Origin')=='http://'+self.headers.get('Host','')
                        and self.headers.get('X-Atlas-Local')=='1'
                        and self.headers.get('Content-Type','').split(';')[0]=='application/json',
                        'Same-origin local JSON control required')
                require(0<length<=8192,'Private body exceeds bound')
                action=path.rsplit('/',1)[-1]
                status,body,_=app.api.handle(action,strict_json(self.rfile.read(length)),
                    admission=self.profile_admission if action=='start' else None)
                return self.send(status,body)
            if self.command=='GET' and path=='/api/models':
                require(not length and self.path==path, 'Catalog request unavailable')
                with app.lock:catalog=app.host.catalog()
                catalog.update(profiles_enabled=getattr(self.server,'profile_controls',False) is True,
                               resume_available=False)
                return self.send(200,catalog)
            if self.command=='GET' and path=='/viewer.js':
                require(not length and self.path==path, 'Asset request unavailable')
                bundle=list(BUNDLE)
                if getattr(self.server,'profile_controls',False) is True:
                    bundle.insert(bundle.index('host-client.js'),'profile-client.js')
                return self.send(200,b'\n;\n'.join((ROOT/'web'/name).read_bytes() for name in bundle),'text/javascript')
            if self.command=='GET' and path.startswith('/api/models/') and path.endswith('/binding'):
                parsed=urlsplit(self.path);parts=path.strip('/').split('/')
                query=parse_qs(parsed.query,keep_blank_values=True,max_num_fields=4)
                require(len(parts)==4 and set(query)<= {'context','tensor','slice'}
                        and 'context' in query and all(len(v)==1 for v in query.values()),'Binding query')
                with app.lock:
                    app.host._validate_context(parts[2],query['context'][0])
                    native='/api/binding?'+urlencode({k:v[0] for k,v in query.items() if k!='context'})
                    status,raw,mime=app.host.reader.read(native)
                    require(status==200 and mime=='application/json','Binding unavailable')
                    body=strict_json(raw)
                    app.contexts.remember(parts[2],query['context'][0],body['source_binding'])
                    return self.send(200,body)
            # Shared host lock serializes model switches with authorization/page
            # starts. A busy profile refuses source replacement/release; lease
            # expiry and cancel still have independent service execution.
            with app.lock:
                if self.command=='POST' and path.startswith('/api/view-contexts') and not path.endswith('/heartbeat'):
                    require(not app.supervisor.busy() and app.supervisor.profile_session is None,
                            'Cancel and confirm profile cleanup before replacing or releasing source')
                if self.command=='GET' and path.startswith('/api/models/') and path.endswith('/view'):
                    status,raw,mime=dispatch(app.host,'GET',self.path)
                    body=strict_json(raw)
                    context=body['host_context']
                    app.contexts.remember(context['model_id'],context['context_id'],body['source_binding'])
                    return self.send(status,body,mime)
                self.server.host=app.host
                return super().handle_action()
        except (ValueError,KeyError,TypeError,OSError) as error:
            diagnose(getattr(self.server,'application',None), 'profile_http.request_refused', error)
            return self.send(409,{'version':1,'code':'unavailable','error':'Hosted request unavailable or cleanup pending'})

    do_GET=handle_action
    do_POST=handle_action
