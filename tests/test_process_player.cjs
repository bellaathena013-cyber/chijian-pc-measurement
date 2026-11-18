const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
function setup(){
 const timers=new Map();let id=0;
 const element=()=>({append(){},replaceChildren(){},setAttribute(){},hidden:false});
 const document={hidden:false,addEventListener(){},createElement:element};
 const context={document,window:{},Image:class{decode(){return Promise.resolve()}},matchMedia:()=>({matches:false}),performance:{now:()=>0},requestAnimationFrame:()=>++id,cancelAnimationFrame(){},setTimeout:fn=>{timers.set(++id,fn);return id},clearTimeout:id=>timers.delete(id)};
 vm.runInNewContext(fs.readFileSync('app/process-player.js','utf8'),context);
 const player=new context.window.ProcessPlayer(element(),{});player.paint=()=>{};
 return {player,document,timers};
}
const result={measurement_mode:'photo_model',images:{original:'local.png'},markers:[]};
test('hidden response waits for visibility before autoplay',async()=>{
 const {player,document,timers}=setup();document.hidden=true;await player.load(result,true);
 assert.equal(timers.size,0);assert.equal(player.step,0);assert.equal(player.resumeWhenVisible,true);
 document.hidden=false;player.visibility();assert.equal(timers.size,1);
});
test('playing resumes after hiding, manual pause stays paused',async()=>{
 const {player,document,timers}=setup();await player.load(result,true);
 document.hidden=true;player.visibility();assert.equal(timers.size,0);
 document.hidden=false;player.visibility();assert.equal(timers.size,1);
 player.pause();document.hidden=true;player.visibility();document.hidden=false;player.visibility();assert.equal(timers.size,0);
});
test('new upload cancels pending hidden replay',async()=>{
 const {player,document,timers}=setup();document.hidden=true;await player.load(result,true);player.reset();
 document.hidden=false;player.visibility();assert.equal(timers.size,0);assert.equal(player.data,null);
});
