'use strict';
// Pure asynchronous fake-file checks; no disk, model, browser or process.
const assert=require('node:assert/strict'),crypto=require('node:crypto'),{hashFile}=require('./acceptance/support.cjs');
(async()=>{
 for(const data of [Buffer.alloc(0),Buffer.from([0,255,1]),Buffer.from('abcdefghi0123456789')]){
  const buffer=Buffer.alloc(7,42),seen=[];let offset=0,closed=0;
  const open=async(file,mode)=>{assert.equal(mode,'r');return {async read(target,start,length,position){assert.strictEqual(target,buffer);assert.equal(start,0);assert.equal(length,7);assert.equal(position,null);seen.push(target.buffer);const bytesRead=Math.min(3,data.length-offset);data.copy(target,start,offset,offset+bytesRead);offset+=bytesRead;return {bytesRead};},async close(){closed++;}};};
  assert.equal(await hashFile('fixture',buffer,open),crypto.createHash('sha256').update(data).digest('hex'));assert.equal(closed,1);assert.equal(new Set(seen).size,1);assert.equal(offset,data.length);
 }
 let closed=0;await assert.rejects(hashFile('read-error',Buffer.alloc(7),async()=>({read:async()=>{throw Error('controlled read error');},close:async()=>{closed++;}})),/controlled read error/);assert.equal(closed,1);
 await assert.rejects(hashFile('open-error',Buffer.alloc(7),async()=>{throw Error('controlled open error');}),/controlled open error/);
 console.log('PASS: full bytes, short final reads, empty files, single backing buffer, handle close on success/read failure, open failure');
})().catch(e=>{console.error(e);process.exitCode=1;});
