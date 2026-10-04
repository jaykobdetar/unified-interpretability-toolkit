'use strict';
// Browser adapter. app.js owns model/view epochs; this module owns local-only tools.
(()=>{
  const A=globalThis.AtlasTools;if(!A)return;
  const el=id=>document.getElementById(id),say=(id,msg)=>{el(id).textContent=msg;};
  const ids=['region-r0','region-c0','region-r1','region-c1'];
  let activeScope=null,notes=[],editing=null,pendingBookmark=null,writeTimer=null,exportController=null;
  const isViewLink=hash=>hash.startsWith('#wa=');
  let restoring=false,linked=isViewLink(location.hash),lastLocation=location.hash,scopeFailed=false;
  const retries=new Map();
  const bounds=()=>A.region(ids.map(id=>{const v=el(id).value;return /^\d+$/.test(v)?Number(v):NaN;}),state.tensor);
  const setBounds=b=>{cancelExport();ids.forEach((id,i)=>{el(id).value=String(b[i]);});scheduleBookmark();};
  function cancelExport(){if(exportController){exportController.abort();exportController=null;say('region-status','Export cancelled; no file produced.');}el('region-export').disabled=false;el('region-cancel').disabled=true;renderRetries();}
  function download(content,type,name){const blob=new Blob([content],{type});if(blob.size>A.MAX_BYTES)throw new Error('Download exceeds 256 KiB');const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);try{a.click();}finally{a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}}
  function currentViewport(){const v=state.viewers.left;if(!state.current||!v?.world.getItemCount())return [0,0,state.tensor.cols,state.tensor.rows];const b=v.viewport.getBounds(true),item=v.world.getItemAt(0),p=item.viewportToImageCoordinates(b.x,b.y,true),q=item.viewportToImageCoordinates(b.x+b.width,b.y+b.height,true);return [p.x,p.y,q.x-p.x,q.y-p.y];}
  function writeBookmark(push=false){
    if(restoring||!state.model||!state.tensor||(!push&&!linked))return;
    try{const hash=A.bookmark({...settings(),region:bounds(),viewport:currentViewport()},state.model);
      // Drop query text rather than preserving an unrelated prompt/path/capability.
      if(hash!==location.hash||location.search)history[push?'pushState':'replaceState'](null,'',location.pathname+hash);
      linked=true;lastLocation=hash;say('bookmark-status','View link saved in the address bar. Copy the URL to bookmark it; notes and prompts are excluded.');
    }catch(e){say('bookmark-status',e.message);}
  }
  function scheduleBookmark(){clearTimeout(writeTimer);if(linked&&!restoring)writeTimer=setTimeout(()=>writeBookmark(),350);}
  function refreshNotes(){
    try{const s=A.scope(state.model,state.tensor),key=A.noteKey(s);if(key===activeScope?.key)return;
      activeScope={value:s,key};editing=null;notes=[];scopeFailed=false;el('note-text').value='';
      say('note-scope',`${s.name}${s.slice?' · slice ['+s.slice.join(', ')+']':''} · revision ${s.revision} · source ${s.source_identity.slice(0,12)}…`);
      try{notes=A.loadNotes(localStorage,s);say('note-status','Notes for this exact source and tensor.');}catch(e){scopeFailed=true;say('note-status',`Local notes unavailable: ${e.message}. Existing storage was not changed.`);}
      el('note-controls').disabled=false;el('note-save').disabled=scopeFailed;renderNotes();
    }catch(e){activeScope=null;notes=[];el('note-controls').disabled=true;say('note-scope',e.message);}
  }
  function requireNoteScope(){if(!activeScope||activeScope.key!==A.noteKey(A.scope(state.model,state.tensor)))throw new Error('Model/tensor changed; reload the note scope before editing or exporting.');}
  function persist(next){requireNoteScope();if(scopeFailed)throw new Error('Local note storage unavailable');A.saveNotes(localStorage,activeScope.value,next);notes=next;renderNotes();}
  function renderNotes(){
    const list=el('note-list');list.replaceChildren();
    for(const n of notes){const row=document.createElement('article'),p=document.createElement('p'),actions=document.createElement('div');row.className='local-note';actions.className='tool-actions';p.textContent=`${n.kind} [${n.region.join(', ')}]: ${n.text}`;row.append(p,actions);
      const button=(label,action)=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',action);actions.append(b);};
      button('Show pin',()=>{setBounds(n.region);focusRegion();});
      button('Edit',()=>{editing=n.id;setBounds(n.region);el('note-kind').value=n.kind;el('note-text').value=n.text;say('note-status','Editing saved note. Save note applies changes.');el('note-text').focus();});
      button('Delete',()=>{try{persist(notes.filter(x=>x.id!==n.id));if(editing===n.id){editing=null;el('note-text').value='';}say('note-status','Note deleted from this browser.');}catch(e){say('note-status',e.message);}});
      list.append(row);
    }
  }
  function applyViewport(b){if(!state.current)return;const v=state.viewers.left,item=v.world.getItemAt(0),r=item.imageToViewportRectangle(...b);v.viewport.fitBounds(r,true);synchronize('left');scheduleResolution();}
  function focusRegion(){try{const b=bounds();applyViewport([b[1],b[0],b[3]-b[1]+1,b[2]-b[0]+1]);say('region-status',`Native rows ${b[0]}–${b[2]}, columns ${b[1]}–${b[3]} selected.`);scheduleBookmark();window.atlasFocusView?.();}catch(e){say('region-status',e.message);}}
  function configure(model){
    clearTimeout(writeTimer);restoring=true;pendingBookmark=null;
    if(isViewLink(location.hash)){const parsed=A.parseBookmark(location.hash);try{if(!parsed)throw new Error('Invalid or unsupported view link ignored.');pendingBookmark=A.resolveBookmark(parsed,model);
        if(!model.calibration_complete&&SIDES.some(side=>pendingBookmark[side].startsWith('global_')))throw new Error('Bookmarked global rules require complete calibration. Refresh after calibration to restore them.');
        state.tensor=A.withSlice(model.catalog.find(t=>t.id===pendingBookmark.tensor),pendingBookmark.slice||[]);for(const side of SIDES)el(side+'-rule').value=pendingBookmark[side];
        setBounds(pendingBookmark.region);say('bookmark-status','Restoring view link for this exact source.');
      }catch(e){pendingBookmark=null;linked=false;say('bookmark-status',e.message);}}
    el('region-controls').disabled=false;
    ids.forEach((id,i)=>el(id).max=String((i%2?state.tensor.cols:state.tensor.rows)-1));
    if(!pendingBookmark){try{bounds();}catch{setBounds([0,0,0,0]);}}
    refreshNotes();drawCatalog();restoring=false;
  }
  function navigate(){
    if(location.hash===lastLocation)return;lastLocation=location.hash;
    // Ordinary section anchors are navigation, never malformed source bookmarks.
    if(location.hash&&!isViewLink(location.hash)){linked=false;return;}
    clearTimeout(writeTimer);cancelExport();linked=isViewLink(location.hash);
    // Refresh invalidates every prior model/view/inspection/poll callback. No session mutation.
    if(!location.hash){setBounds([0,0,0,0]);state.tensor=null;}
    refreshModel();
  }
  globalThis.atlasWorkspace={
    changed(reason){
      if(reason==='viewport'){scheduleBookmark();return;}
      clearTimeout(writeTimer);cancelExport();retries.clear();say('read-retry','');
      if(reason==='tensor'){pendingBookmark=null;setBounds([0,0,0,0]);if(state.model)configureWithoutURL();}
      if(reason==='deactivate'){el('region-controls').disabled=true;el('note-controls').disabled=true;}
    },
    model:configure,
    rawReady(){el('region-controls').disabled=false;refreshNotes();el('note-controls').disabled=!activeScope;},
    loaded(){el('region-controls').disabled=false;refreshNotes();el('note-controls').disabled=!activeScope;
      if(pendingBookmark){restoring=true;applyViewport(pendingBookmark.viewport);pendingBookmark=null;restoring=false;}scheduleBookmark();},
    retry(event,url,signal,owner=url.split('?')[0]){if(signal?.aborted)return;
      if(event.phase==='started'){for(const [key,value] of retries)if(value.owner===owner&&value.phase==='failed')retries.delete(key);}
      else if(event.phase==='success')retries.delete(event.request);else retries.set(event.request,{...event,signal,owner});
      renderRetries();
    }
  };
  function renderRetries(){
      for(const [key,value] of retries)if(value.signal?.aborted)retries.delete(key);
      const events=[...retries.values()].filter(e=>!e.signal?.aborted),retry=events.find(e=>e.phase==='retrying'),failed=events.find(e=>e.phase==='failed');
      if(retry)say('read-retry',`Read unavailable (${retry.code}). Retry ${retry.attempt}/${retry.limit} in ${retry.delay} ms. Refresh status restarts recovery; navigation cancels retries.`);
      else if(failed)say('read-retry',failed.status===503&&failed.code==='calibration_readiness'?'Calibration not ready. Refresh status after calibration.':`Read unavailable (${failed.code||'request_error'}). Automatic retry stopped. Use Refresh status or retry the local action manually.`);
      else say('read-retry','');
  }
  function configureWithoutURL(){ids.forEach((id,i)=>el(id).max=String((i%2?state.tensor.cols:state.tensor.rows)-1));refreshNotes();}
  el('bookmark-save').addEventListener('click',()=>writeBookmark(true));
  el('bookmark-reset').addEventListener('click',()=>{clearTimeout(writeTimer);if(location.hash||location.search)history.pushState(null,'',location.pathname);lastLocation=null;linked=false;navigate();say('bookmark-status','View link reset. Default tensor and fit restored.');});
  window.addEventListener('popstate',navigate);window.addEventListener('hashchange',navigate);
  ids.forEach(id=>el(id).addEventListener('input',()=>{cancelExport();scheduleBookmark();}));
  el('region-use-cell').addEventListener('click',()=>{if(!state.selected){say('region-status','Inspect a native address first.');return;}const [r,c]=state.selected;setBounds([r,c,r,c]);focusRegion();});
  el('region-focus').addEventListener('click',focusRegion);
  el('region-cancel').addEventListener('click',cancelExport);
  el('region-export').addEventListener('click',async()=>{
    if(exportController)return;const controller=new AbortController();exportController=controller;const signal=controller.signal;
    el('region-export').disabled=true;el('region-cancel').disabled=false;let timedOut=false;
    const deadline=setTimeout(()=>{timedOut=true;controller.abort();},30000);
    try{const csv=await A.collectRegion({model:state.model,tensor:state.tensor,bounds:bounds(),signal,read:(url,signal)=>A.readJSON(url,{signal,onState:event=>globalThis.atlasWorkspace.retry(event,url,signal,'export')}),onProgress:(n,total)=>say('region-status',`Reading original values: ${n}/${total}.`)});
      if(signal.aborted||exportController!==controller)return;download(csv,'text/csv;charset=utf-8','weight-atlas-region.csv');say('region-status','Raw CSV exported with exact source metadata.');
    }catch(e){if(exportController===controller)say('region-status',timedOut?'Export exceeded 30 seconds; no file produced. Try a smaller region.':e.name==='AbortError'?'Export cancelled; no file produced.':e.message);}
    finally{clearTimeout(deadline);renderRetries();if(exportController===controller){exportController=null;el('region-export').disabled=false;el('region-cancel').disabled=true;}}
  });
  el('note-new').addEventListener('click',()=>{editing=null;el('note-text').value='';say('note-status','New note for the selected native pin.');});
  el('note-save').addEventListener('click',()=>{try{
    const b=bounds(),kind=el('note-kind').value,s=activeScope.value;if(kind==='row'){b[1]=0;b[2]=b[0];b[3]=s.cols-1;}else if(kind==='column'){b[0]=0;b[2]=s.rows-1;b[3]=b[1];}
    const n={id:editing||crypto.randomUUID(),kind,region:b,text:el('note-text').value};persist(A.updateNote(notes,s,n));editing=n.id;say('note-status','Saved only in this browser, for this exact model source and tensor.');
  }catch(e){say('note-status',e.message);}});
  el('note-export').addEventListener('click',()=>{try{requireNoteScope();if(scopeFailed)throw new Error('Local note storage unavailable');download(JSON.stringify({format:'weight-atlas-notes-v1',scope:activeScope.value,notes},null,2),'application/json','weight-atlas-notes.json');say('note-status','Private notes exported. This file contains your note text.');}catch(e){say('note-status',e.message);}});
  window.addEventListener('pagehide',()=>{cancelExport();clearTimeout(writeTimer);state.modelController?.abort();deactivate();});
})();
