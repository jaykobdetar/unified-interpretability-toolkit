'use strict';
function createFixture(){
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').join(__dirname,'../../web/inference.js'),'utf8');
class Element{constructor(){this.value='';this.textContent='';this.disabled=false;this.hidden=false;this.listeners={};this.width=512;this.height=288;}replaceChildren(...items){this.children=items;}append(...items){this.children=(this.children||[]).concat(items);}addEventListener(k,f){this.listeners[k]=f;}getContext(){return {clearRect(){},fillRect(){}};}getBoundingClientRect(){return {top:0,left:0,width:512,height:288};}}
const elems=new Map(),get=id=>{if(!elems.has(id))elems.set(id,new Element());return elems.get(id);};
get('infer-rate').value='2';get('infer-limit').value='4';get('infer-layer').value='0';get('infer-mode').value='step';get('infer-prompt').value='Hello';
let timerId=0;const timers=new Map(),requests=[];
const context=vm.createContext({console,document:{getElementById:get,createElement:()=>new Element()},window:{addEventListener(){}},fetch:(url,options)=>new Promise(resolve=>requests.push({url,options,resolve:(body,status=200)=>resolve({ok:status<400,status,json:async()=>body})})),setTimeout:(f,ms)=>{timers.set(++timerId,{f,ms});return timerId;},clearTimeout:id=>timers.delete(id)});
vm.runInContext(source,context);
const tick=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
const take=part=>{const i=requests.findIndex(r=>r.url.endsWith(part));assert(i>=0,part);return requests.splice(i,1)[0];};
const click=id=>get('infer-'+id).listeners.click();
const submit=()=>get('infer-form').listeners.submit({preventDefault(){}});

return {fs,vm,assert,source,Element,elems,get,timers,requests,context,tick,take,click,submit};
}
module.exports={createFixture};
