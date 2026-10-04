'use strict';
// Fixture host only. Capabilities live in this closure, never URLs or storage.
(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.AtlasHost=api.create({fetchImpl:root.fetch.bind(root),document:root.document,
    schedule:root.setTimeout.bind(root),cancel:root.clearTimeout.bind(root),window:root});
})(globalThis,()=>{
  const abort=()=>new DOMException('Model context changed','AbortError');
  const require=(ok,message)=>{if(!ok)throw new Error(message);};
  function create({fetchImpl,document=null,schedule=setTimeout,cancel=clearTimeout,window=null}){
    let active=null,generation=0,switching=false,timer=null,onChange=async()=>{},onLost=()=>{};
    let profiles=null,profileCapabilities={profiles_enabled:false,resume_available:false},profileEpoch=0;
    let profileBinding=null,sourceUnresolved=false;
    const snapshot=()=>active?{model_id:active.model_id,context_id:active.context_id,generation}:null;
    function current(saved){return !!saved&&!!active&&saved.generation===generation&&saved.model_id===active.model_id&&saved.context_id===active.context_id;}
    function assertCurrent(saved){if(!current(saved))throw abort();}
    async function request(url,options={}){
      const response=await fetchImpl(url,{cache:'no-store',...options});
      const body=await response.json();
      if(!response.ok)throw Object.assign(new Error(body.error||`HTTP ${response.status}`),{code:body.code,status:response.status,responseReceived:true});
      require(body.api_version===1,'Invalid host response version');return body;
    }
    const post=(url,body,extra={})=>request(url,{method:'POST',headers:{'Content-Type':'application/json','X-Atlas-Local':'1'},body:JSON.stringify(body),...extra});
    function note(message){const node=document?.getElementById('host-model-status');if(node)node.textContent=message;}
    function lose(saved,message){if(!current(saved))return;profiles?.configure(profileCapabilities,null,null);profileEpoch++;active=null;generation++;cancel(timer);timer=null;note(message);onLost(message);}
    async function heartbeat(){
      const saved=snapshot();if(!saved)return;
      try{await post(`/api/view-contexts/${saved.context_id}/heartbeat`,{capability:active.capability});}
      catch(e){lose(saved,'Reader lease unavailable. Reopen the selected model.');return;}
      if(current(saved))timer=schedule(heartbeat,5000);
    }
    async function select(model_id){
      require(!switching,'A model selection is already pending');
      require(!sourceUnresolved,'Source ownership is unresolved. Reopen the page after lease cleanup.');switching=true;
      const selectNode=document?.getElementById('host-model-select');if(selectNode)selectNode.disabled=true;
      const previous=active,previousBinding=profileBinding;let acquireSent=false,installed=false;
      sourceUnresolved=true;
      try{
        await prepareProfileChange();
        const body={model_id,...(active?{context_id:active.context_id,capability:active.capability}:{})};
        acquireSent=true;
        const next=await post('/api/view-contexts',body);
        require(next.model_id===model_id&&/^m_[0-9a-f]{64}$/.test(next.model_id)
          &&/^[0-9a-f]{32}$/.test(next.context_id)&&/^[0-9a-f]{64}$/.test(next.capability),'Invalid reader lease');
        active=next;generation++;installed=true;sourceUnresolved=false;profileBinding=null;cancel(timer);timer=schedule(heartbeat,5000);
        if(selectNode)selectNode.value=active.model_id;
        note('Opening selected fixture. No inference or download permission is granted.');
        await onChange(snapshot());
        return snapshot();
      }catch(e){
        // Only a conclusive refusal (or no acquire sent) can restore the old
        // selection, and even then its native binding must still verify.
        if(!installed&&(!acquireSent||e.responseReceived===true)){
          sourceUnresolved=false;
          if(active===previous&&previousBinding)await setProfileSelection(previousBinding);
        }
        if(selectNode&&active)selectNode.value=active.model_id;note(e.message);throw e;
      }
      finally{switching=false;if(selectNode)selectNode.disabled=false;}
    }
    function url(raw,saved=snapshot()){
      assertCurrent(saved);
      const match=/^\/(?:api\/(model|view|inspect|progress|tensor-status)|(?<tile>tile))(\?[^#]*)?$/.exec(raw);
      require(match,'Read-only model route required');
      const name=match.groups?.tile?'tile':match[1];
      const query=new URLSearchParams(raw.includes('?')?raw.slice(raw.indexOf('?')+1):'');
      require(!query.has('context'),'Caller cannot override model context');
      query.set('context',saved.context_id);
      return `/api/models/${saved.model_id}/${name}?${query}`;
    }
    function bindRead(raw){
      const saved=snapshot();
      return {url:url(raw,saved),assertCurrent:()=>assertCurrent(saved),check(data){
        assertCurrent(saved);
        require(data.host_context?.model_id===saved.model_id&&data.host_context?.context_id===saved.context_id,
          'Response belongs to another model context');
        if(active.source_identity)require(data.host_context.source_identity===active.source_identity
          &&data.host_context.model_identity===active.model_identity,'Response source identity changed');
        else{active.source_identity=data.host_context.source_identity;active.model_identity=data.host_context.model_identity;}
      }};
    }
    async function release(){
      require(!switching,'A source transition is already pending');
      require(!sourceUnresolved,'Source ownership is unresolved. Wait for lease cleanup.');switching=true;
      const previous=active,previousBinding=profileBinding;let releaseSent=false;
      sourceUnresolved=true;
      try{
        await prepareProfileChange();
        if(!active){sourceUnresolved=false;return;}
        releaseSent=true;
        const result=await post(`/api/view-contexts/${previous.context_id}/release`,{capability:previous.capability},{keepalive:true});
        active=null;generation++;profileBinding=null;sourceUnresolved=false;cancel(timer);timer=null;
        profiles?.configure(profileCapabilities,null,null);
        return result;
      }catch(error){
        if(!releaseSent||error.responseReceived===true){
          sourceUnresolved=false;
          if(active===previous&&previousBinding)await setProfileSelection(previousBinding);
        }
        throw error;
      }finally{switching=false;}
    }
    async function initialize(changed,lost){
      onChange=changed;onLost=lost;
      const region=document?.getElementById('host-model-picker'),selectNode=document?.getElementById('host-model-select');
      if(region)region.hidden=false;
      const catalog=await request('/api/models');
      profileCapabilities={profiles_enabled:catalog.profiles_enabled===true,resume_available:false};
      const entries=catalog.models;
      require(Array.isArray(entries),'Invalid installed-model catalog');
      if(selectNode){
        selectNode.replaceChildren(...entries.map(item=>{const option=document.createElement('option');option.value=item.model_id;
          option.textContent=item.name+(item.fixture_eligible?'':' · unavailable in fixture mode');option.disabled=!item.fixture_eligible;return option;}));
        selectNode.addEventListener('change',()=>select(selectNode.value).catch(()=>{}));
        document.getElementById('host-model-open')?.addEventListener('click',()=>select(selectNode.value).catch(()=>{}));
      }
      window?.addEventListener('pagehide',()=>release().catch(()=>{}));
      const first=entries.find(item=>item.fixture_eligible);
      if(first)await select(first.model_id);else note('No enabled synthetic fixture. The owner must register and enable one locally.');
    }
    async function postProfile(path,body){
      require(/^\/api\/profiles\/(start|status|page|cancel|heartbeat|reconcile)$/.test(path),'Private profile route required');
      const controller=new AbortController(),timeout=schedule(()=>controller.abort(),3000);
      try{
        const response=await fetchImpl(path,{method:'POST',cache:'no-store',signal:controller.signal,
          headers:{'Content-Type':'application/json','X-Atlas-Local':'1'},body:JSON.stringify(body)});
        const result=await response.json();
        if(!response.ok)throw Object.assign(new Error(result.error||'Profile request unavailable'),{status:response.status});
        require(result.resume_available!==true,'Resume is unavailable');return result;
      }finally{cancel(timeout);}
    }
    function mountProfiles(region,api){
      require(!profiles,'Profile panel already mounted');
      profiles=api.mount(region,{post:postProfile,setTimer:schedule,clearTimer:cancel});
    }
    function suspendProfiles(){profileEpoch++;profiles?.suspend?.(true);}
    async function prepareProfileChange(){
      suspendProfiles();
      if(!profiles)return;
      await profiles.reset();
      require(!profiles.client.snapshot().cleanup_pending,'Profile cleanup pending. Wait for status, then select again.');
    }
    async function setProfileSelection(expected){
      suspendProfiles();
      const epoch=profileEpoch,saved=snapshot();
      if(sourceUnresolved)return;
      profileBinding=null;
      profiles?.configure(profileCapabilities,null,null);
      if(!profiles||!profileCapabilities.profiles_enabled||!saved||!expected)return;
      try{
        const query=new URLSearchParams({context:saved.context_id,tensor:expected.tensor});
        if(expected.slice.leading_indices.length)query.set('slice',expected.slice.leading_indices.join(','));
        const result=await request(`/api/models/${saved.model_id}/binding?${query}`);
        assertCurrent(saved);if(epoch!==profileEpoch||sourceUnresolved)throw abort();
        const canonical=v=>JSON.stringify(v,(_k,x)=>x&&typeof x==='object'&&!Array.isArray(x)?Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])):x);
        require(canonical(result.source_binding)===canonical(expected),'Profile selection binding changed');
        profiles.configure(profileCapabilities,{model_id:saved.model_id,context_id:saved.context_id,
          tab_capability:active.capability},result.source_binding);
        profileBinding=result.source_binding;profiles.suspend?.(false);
      }catch(error){if(epoch===profileEpoch)profiles.unavailable(error.message);}
    }
    return {initialize,select,url,bindRead,snapshot,current,release,mountProfiles,prepareProfileChange,setProfileSelection,isHost:true};
  }
  return {create};
});
