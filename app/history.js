const $=s=>document.querySelector(s);
const names={EXPERIMENTAL_MEASUREMENT:'试验尺寸',PASS:'本轮通过',REVIEW:'需要复核',MANUAL_REQUIRED:'需人工复核或重拍',RETAKE:'请重新拍摄',INVALID_INPUT:'请检查输入'};
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e}
const canvas=$('#canvas'),player=new ProcessPlayer($('#process'),canvas,()=>{++imageGeneration;canvas.hidden=false;$('#views').querySelectorAll('button').forEach(b=>b.classList.remove('active'))});
let generation=0,imageGeneration=0;
async function loadHistory(){
 const host=$('#history');$('#refresh').disabled=true;
 try{
  const response=await fetch('/api/runs');if(!response.ok)throw Error('记录读取失败，请点击刷新重试。');const data=await response.json();host.replaceChildren();
  if(!data.items.length){host.append(text('p','暂无检测记录。返回“照片测量”上传第一张照片。'));return}
  const table=document.createElement('table'),head=document.createElement('thead'),heading=document.createElement('tr');
  ['构件 / 时间','来源','状态','长 × 宽 / mm','操作'].forEach(v=>heading.append(text('th',v)));head.append(heading);table.append(head);
  const body=document.createElement('tbody');
  for(const item of data.items){
   const row=document.createElement('tr'),name=text('td',item.component_id||'未命名构件');name.append(text('small',new Date(item.created_at*1000).toLocaleString('zh-CN')));
   row.append(name,text('td',item.source_type==='simulation'?'模拟':item.measurement_mode==='photo_model'?'照片模型':'布局测量'),text('td',names[item.result]||item.result),text('td',item.measured_length_mm==null?'—':`${item.measured_length_mm} × ${item.measured_width_mm}`));
   const cell=document.createElement('td'),button=text('button','查看详情');button.type='button';button.setAttribute('aria-label',`查看 ${item.component_id||'未命名构件'} 的检测记录`);button.onclick=()=>openRecord(item.run_id);cell.append(button);row.append(cell);body.append(row);
  }table.append(body);host.append(table);
 }catch(e){host.replaceChildren(text('p',e.message))}finally{$('#refresh').disabled=false}
}
async function openRecord(id){
 const current=++generation;++imageGeneration;player.reset();$('#history-list').hidden=true;$('#record').hidden=false;canvas.hidden=true;$('#record-result').replaceChildren();$('#views').replaceChildren();$('#error').textContent='';$('#record-message').textContent='正在读取检测详情…';$('#download').hidden=true;
 $('#record').scrollIntoView({block:'start'});
 try{
  const response=await fetch(`/api/runs/${encodeURIComponent(id)}/result.json`);if(!response.ok)throw Error('无法读取这条记录，请刷新列表后重试。');const r=await response.json();if(current!==generation)return;
  history.replaceState(null,'',`/history?run=${encodeURIComponent(id)}`);
  $('#record-title').textContent=`检测详情 · ${r.component_id||'未命名构件'}`;$('#record-message').textContent='';
  $('#download').href=r.download_url;$('#download').download='result.json';$('#download').hidden=false;
  const host=$('#record-result');host.append(text('strong',names[r.result]||r.result,'status warn'),text('p',r.source_type==='simulation'?'模拟图片 · 非现场精度结论':'上传照片 · 待现场核验'),text('p',r.message||''));
  if(r.success){const table=document.createElement('table');const rows=r.measurement_mode==='photo_model'?[['尺寸','试验测量值 / mm'],['长边',r.measured_length_mm],['短边',r.measured_width_mm]]:[['尺寸 / mm','设计值','测量值','偏差'],['长边',r.design_length_mm,r.measured_length_mm,r.length_error_mm],['短边',r.design_width_mm,r.measured_width_mm,r.width_error_mm]];for(const values of rows){const tr=document.createElement('tr');values.forEach(v=>tr.append(text('td',String(v))));table.append(tr)}host.append(table)}
  for(const warning of r.warnings||[])host.append(text('p',typeof warning==='string'?warning:JSON.stringify(warning)));
  const ready=r.measurement_readiness;if(ready)host.append(text('p',`整图候选 ${ready.detected_marker_count??'未记录'} 个 · 目标内 ${ready.markers_on_target_count??'未确定'} · 实际标定 ${ready.calibration_marker_count} 个 · 黑色外框 ${r.known_marker_size_mm} mm`));
  async function show(key,button){
   player.pause();const imageId=++imageGeneration,picture=new Image();picture.src=r.images[key];await picture.decode();
   if(current!==generation||imageId!==imageGeneration)return;
   canvas.width=picture.naturalWidth;canvas.height=picture.naturalHeight;canvas.getContext('2d').drawImage(picture,0,0);canvas.hidden=false;
   $('#views').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));
  }
  for(const [key,label] of Object.entries({original:'原图',annotated:'边界标注',marker_details:'标志细节',corrected_annotated:'透视校正'})){
   if(!r.images?.[key])continue;const button=text('button',label);button.type='button';button.onclick=()=>show(key,button).catch(()=>$('#error').textContent='图片加载失败，请重新查看记录。');$('#views').append(button);
  }
  if(r.measurement_mode==='photo_model'&&r.images?.original){await player.load(r,false);if(current!==generation)return;player.select(3)}
  else if(r.images?.annotated)await show('annotated');
 }catch(e){if(current===generation){$('#record-message').textContent='';$('#error').textContent=e.message}}
}
$('#back-list').onclick=()=>{++generation;++imageGeneration;player.reset();$('#record').hidden=true;$('#history-list').hidden=false;$('#error').textContent='';history.replaceState(null,'','/history');$('#refresh').focus()};
$('#refresh').onclick=loadHistory;loadHistory();
const requested=new URLSearchParams(location.search).get('run');if(requested&&/^[a-f0-9]{32}$/.test(requested))openRecord(requested);
