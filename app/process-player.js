/* Replays persisted geometry. It never reports inference progress. */
class ProcessPlayer {
  constructor(root, canvas, onActivate=()=>{}) {
    this.root=root; this.canvas=canvas; this.onActivate=onActivate;
    this.generation=0; this.timer=null; this.frame=null; this.data=null; this.step=0;
    this.motion=matchMedia('(prefers-reduced-motion: reduce)');
    this.resumeWhenVisible=false;
    this.visibility=()=>{
      if(document.hidden){const playing=this.timer!==null;this.pause();this.resumeWhenVisible=playing}
      else if(this.resumeWhenVisible){this.resumeWhenVisible=false;this.play()}
    };
    document.addEventListener('visibilitychange',this.visibility);
  }
  pause() {
    this.resumeWhenVisible=false;
    clearTimeout(this.timer); cancelAnimationFrame(this.frame); this.timer=null; this.frame=null;
    if(this.playButton)this.playButton.textContent='播放 / 重播';
  }
  reset() {
    this.pause(); this.generation++; this.data=null; this.root.hidden=true; this.root.replaceChildren();
  }
  async load(data, autoplay=false) {
    this.reset();
    if(data.measurement_mode!=='photo_model'||!data.images?.original)return false;
    const generation=this.generation, picture=new Image(); picture.src=data.images.original;
    await picture.decode();
    if(generation!==this.generation)return false;
    this.data=data; this.picture=picture;
    const saved=data.process_visualization||{};
    this.geometry={polygon:saved.component_polygon_px, boundary:saved.boundary_px||data.boundary_refinement?.quad_original_px};
    this.root.hidden=false;
    const heading=document.createElement('div');heading.className='process-heading';
    const title=document.createElement('strong');title.textContent='识别过程回放';heading.append(title);
    const note=document.createElement('span');note.textContent='本次真实结果 · 非实时运算进度';heading.append(note);this.root.append(heading);
    const steps=document.createElement('div');steps.className='process-steps';steps.setAttribute('role','group');steps.setAttribute('aria-label','选择回放步骤');
    this.buttons=['原始照片','构件识别','边界与标志','测量结果'].map((name,index)=>{
      const button=document.createElement('button');button.type='button';button.textContent=`${index+1}. ${name}`;
      button.onclick=()=>{this.pause();this.select(index)};steps.append(button);return button;
    });this.root.append(steps);
    const controls=document.createElement('div');controls.className='process-controls';
    this.playButton=document.createElement('button');this.playButton.type='button';this.playButton.textContent='播放 / 重播';
    this.playButton.onclick=()=>{if(this.timer!==null){this.pause();this.paint(1)}else this.play()};controls.append(this.playButton);
    const finish=document.createElement('button');finish.type='button';finish.textContent='直接看结果';finish.onclick=()=>{this.pause();this.select(3)};controls.append(finish);this.root.append(controls);
    this.caption=document.createElement('p');this.caption.setAttribute('role','status');this.root.append(this.caption);
    this.step=0;
    if(autoplay&&!this.motion.matches)this.play();else this.select(0);
    return true;
  }
  select(step,animate=false) {
    if(!this.data)return;
    cancelAnimationFrame(this.frame);this.step=step;this.onActivate();
    this.buttons.forEach((button,index)=>button.setAttribute('aria-pressed',String(index===step)));
    const r=this.data, ready=r.measurement_readiness;
    const polygon=this.valid(this.geometry.polygon), boundary=this.valid(this.geometry.boundary);
    this.caption.textContent=[
      '原图：检查构件上表面和标志卡是否完整、清晰。',
      polygon?'绿色区域是本次模型选中的构件区域，还需检查是否包含侧面或其他物体。':(r.process_visualization?'本次没有识别出可展示的构件区域。':'这条历史记录未保存模型区域，不能补画识别结果；可继续查看已保存的边界与标志。'),
      `${boundary?'橙色线为拟合边界。':'本次没有可靠的四角边界。'} 蓝框为标志候选${ready?`；实际标定 ${ready.calibration_marker_count} 个`:''}，C 编号不是卡片印刷数字。`,
      r.success?`试验尺寸 ${r.measured_length_mm} × ${r.measured_width_mm} mm · ${r.result==='REVIEW'?'需要复核':'试验尺寸，仍需核验'}。请阅读下方复核提示。`:'本次没有可靠尺寸。已展示实际识别信息，请按下方提示复核或重拍。'
    ][step];
    const start=performance.now();
    const tick=now=>{const t=animate&&!this.motion.matches?Math.min(1,(now-start)/800):1;this.paint(t);if(t<1)this.frame=requestAnimationFrame(tick)};
    this.frame=requestAnimationFrame(tick);
  }
  play() {
    if(!this.data)return;
    this.pause();
    if(document.hidden){this.resumeWhenVisible=true;return}
    const start=this.step>=3?0:this.step;
    const advance=step=>{
      this.select(step,true);
      if(step<3){this.playButton.textContent='暂停';this.timer=setTimeout(()=>advance(step+1),2400)}
      else{this.timer=null;this.playButton.textContent='播放 / 重播'}
    };advance(start);
  }
  valid(points){return Array.isArray(points)&&points.length>=3&&points.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite))}
  paint(progress=1) {
    if(!this.data)return;
    const canvas=this.canvas,ctx=canvas.getContext('2d'),image=this.picture;
    const ratio=Math.min(1,1280/Math.max(image.naturalWidth,image.naturalHeight));
    canvas.width=Math.round(image.naturalWidth*ratio);canvas.height=Math.round(image.naturalHeight*ratio);
    ctx.drawImage(image,0,0,canvas.width,canvas.height);
    const sx=canvas.width/image.naturalWidth,sy=canvas.height/image.naturalHeight;
    const polygon=(points,color,fill,amount=1)=>{
      if(!this.valid(points))return;
      const vertices=[...points,points[0]].map(p=>[p[0]*sx,p[1]*sy]);
      if(fill){ctx.beginPath();vertices.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle=fill;ctx.fill()}
      const segments=vertices.slice(1).map((p,i)=>Math.hypot(p[0]-vertices[i][0],p[1]-vertices[i][1]));
      let remaining=segments.reduce((a,b)=>a+b,0)*amount;ctx.beginPath();ctx.moveTo(...vertices[0]);
      for(let i=0;i<segments.length&&remaining>0;i++){const part=Math.min(1,remaining/(segments[i]||1));ctx.lineTo(vertices[i][0]+(vertices[i+1][0]-vertices[i][0])*part,vertices[i][1]+(vertices[i+1][1]-vertices[i][1])*part);remaining-=segments[i]}
      ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke();
    };
    if(this.step>=1)polygon(this.geometry.polygon,'#42db87','rgba(20,180,100,.22)',this.step===1?progress:1);
    if(this.step>=2){
      polygon(this.geometry.boundary,'#ffad32',null,this.step===2?progress:1);
      for(const [i,marker] of (this.data.markers||[]).entries()){
        polygon(marker.corners_px,'#40cfff',null,this.step===2?progress:1);
        if(this.valid(marker.corners_px)){const [x,y]=marker.corners_px[0];ctx.font='bold 19px sans-serif';ctx.lineWidth=4;ctx.strokeStyle='#102a35';ctx.strokeText(`C${i+1}`,x*sx,y*sy-8);ctx.fillStyle='#b8efff';ctx.fillText(`C${i+1}`,x*sx,y*sy-8)}
      }
    }
    if(this.step===3){
      const r=this.data, font=Math.max(15,Math.min(25,canvas.width/29));ctx.font=`bold ${font}px "Microsoft YaHei",sans-serif`;
      const label=r.success?`${r.measured_length_mm} × ${r.measured_width_mm} mm · ${r.result==='REVIEW'?'需要复核':'试验尺寸'}`:'未得到可靠尺寸 · 请复核或重拍';
      ctx.fillStyle='rgba(10,30,20,.9)';ctx.fillRect(0,0,canvas.width,font*2.5);ctx.fillStyle='white';ctx.fillText(label,14,font*1.6,canvas.width-28);
    }
  }
}
window.ProcessPlayer=ProcessPlayer;
