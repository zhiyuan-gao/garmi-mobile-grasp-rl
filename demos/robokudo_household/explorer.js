'use strict';
const $=id=>document.getElementById(id);
const steps=[
 ['rgb','RGB-D input','Head camera · actual snapshot','The camera image used by this run; the outlined region projects the detected target cluster.'],
 ['crop','Counter region','Known counter crop','The observed point cloud within the configured counter region.'],
 ['plane','Support plane','PlaneAnnotator','The support plane extracted by RoboKudo, shown in grey.'],
 ['cluster','Target point cloud','PointCloudClusterExtractor','The measured surface points assigned to the selected object, shown in coral.'],
 ['fit','Model alignment','Custom fixed-orientation fit','The known object model in green, aligned to the measured points in coral. Position is estimated; orientation is fixed.']
];
let selectedObservation='',lastRun=null;
$('observation-picker').onchange=()=>{selectedObservation=$('observation-picker').value;loaded='';};
const query=()=>selectedObservation?'?observation='+encodeURIComponent(selectedObservation):'';
let current='rgb',visual=null,estimate=null,run=null,image=new Image(),loaded='',stageData={};
for(const [id,label,source] of steps){const b=document.createElement('button');b.id='node-'+id;b.className=id==='fit'?'custom':'';const state=document.createElement('span');state.className='state';state.textContent='○';const l=document.createElement('span');l.className='label';l.textContent=label;const s=document.createElement('small');s.textContent=source;b.append(state,l,s);b.onclick=()=>select(id);$('nodes').append(b)}
function tab(which){$('image-view').classList.toggle('hidden',which!=='image');$('cloud-view').classList.toggle('hidden',which!=='cloud');$('image-tab').classList.toggle('active',which==='image');$('cloud-tab').classList.toggle('active',which==='cloud');if(which==='cloud')resize()}
function select(id){current=id;const s=steps.find(x=>x[0]===id);$('step-title').textContent=s[1];$('step-description').textContent=s[3];for(const [n] of steps)$('node-'+n).classList.toggle('active',n===id);tab(id==='rgb'?'image':'cloud');if(visual)drawCloud()}
$('image-tab').onclick=()=>tab('image');$('cloud-tab').onclick=()=>tab('cloud');
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('three-canvas').append(renderer.domElement);
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,.001,30),group=new THREE.Group();scene.add(group);
let radius=.45,theta=-Math.PI/2+.4,phi=1.,drag=null,dirty=true;
function resize(){const el=$('three-canvas');if(!el.clientWidth)return;renderer.setSize(el.clientWidth,el.clientHeight);camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();dirty=true}
function reset(){radius=current==='crop'||current==='plane'?1.:.45;theta=-Math.PI/2+.4;phi=1.;dirty=true}
$('reset-view').onclick=reset;
renderer.domElement.onpointerdown=e=>{drag=[e.clientX,e.clientY];renderer.domElement.setPointerCapture(e.pointerId)};
renderer.domElement.onpointermove=e=>{if(!drag)return;theta-=(e.clientX-drag[0])*.008;phi=Math.max(.08,Math.min(Math.PI-.08,phi+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];dirty=true};
renderer.domElement.onpointerup=()=>drag=null;renderer.domElement.onpointercancel=()=>drag=null;
renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();radius=Math.max(.10,Math.min(4,radius*Math.exp(e.deltaY*.001)));dirty=true},{passive:false});
new ResizeObserver(resize).observe($('cloud-view'));
function points(list,color,size){const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(list.flat(),3));group.add(new THREE.Points(geom,new THREE.PointsMaterial({color,size,sizeAttenuation:true})))}
function drawCloud(){while(group.children.length){const o=group.children[0];group.remove(o);o.geometry?.dispose();o.material?.dispose()}if(!visual)return;
 if(current==='crop')points(visual.counter,0x9eafbf,.003);
 else{
 if($('show-plane').checked)points(visual.plane,0x697a90,.002);
 if($('show-cluster').checked&&current!=='plane')points(visual.cluster,0xff9b79,.002);
 if($('show-model').checked&&current==='fit')points(visual.fitted_model,0x70e5c1,.0011);
 }
 group.add(new THREE.AxesHelper(.1));reset();}
for(const id of ['show-plane','show-cluster','show-model'])$(id).onchange=drawCloud;
function animate(){requestAnimationFrame(animate);if(!dirty||$('cloud-view').classList.contains('hidden'))return;camera.up.set(0,0,1);camera.position.set(radius*Math.sin(phi)*Math.cos(theta),radius*Math.sin(phi)*Math.sin(theta),.035+radius*Math.cos(phi));camera.lookAt(0,0,.035);renderer.render(scene,camera);dirty=false}animate();
image.onload=()=>{const c=$('image-canvas');c.width=image.naturalWidth;c.height=image.naturalHeight;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);if(visual){const [lo,hi]=visual.image_box;ctx.lineWidth=2;ctx.strokeStyle='#79efd1';ctx.strokeRect(lo[0],lo[1],hi[0]-lo[0],hi[1]-lo[1]);ctx.font='16px system-ui';ctx.fillStyle='#142a29';ctx.fillRect(lo[0],Math.max(0,lo[1]-27),160,25);ctx.fillStyle='#a4ffe9';ctx.fillText(visual.object_name+' · RoboKudo',lo[0]+7,Math.max(18,lo[1]-9))}$('image-view').querySelector('.empty').hidden=true};
function apply(s){stageData=s;const picker=$('observation-picker');const ids=(s.observations||[]).map(v=>v.id);if(JSON.stringify(ids)!==picker.dataset.ids){picker.innerHTML='<option value="">Follow task</option>';for(const v of s.observations||[]){const o=document.createElement('option');o.value=v.id;o.textContent=v.id;picker.append(o)}picker.dataset.ids=JSON.stringify(ids);picker.value=selectedObservation;}$('connection').textContent=s.status==='completed'?'Task completed':s.status==='failed'?'Run failed':s.active?'Live task':'Ready';$('control-status').textContent=s.perception_id==='mug-initial'?'Initial mug pose for the CRAM world. The mug is observed again after returning to the counter.':s.message;
 $('result').previousElementSibling.textContent=s.perception_id==='mug-initial'?'Subsequent mug grasp':'Physical grasp';const e=s.perception||{};estimate=e;
 for(const [id] of steps){const b=$('node-'+id);b.classList.toggle('done',!!e.position_map);b.querySelector('.state').textContent=e.position_map?'✓':'○'}
 $('object').textContent=e.object_name||s.object||'—';$('pose').textContent=e.position_map?e.position_map.map(x=>x.toFixed(3)).join(', '):'—';$('fit').textContent=e.surface_fit_rmse_m?(e.surface_fit_rmse_m*1000).toFixed(2)+' mm':'—';$('points').textContent=e.cluster_points?.toLocaleString()||'—';
 const events=s.events||[];let station=[...events].reverse().find(e=>e.event==='pickup_station_selected');let ref=[...events].reverse().find(e=>e.event==='pickup_reference_selected');
 if(s.object==='mug'){const seq=s.sequence?.events||[];const i=seq.findIndex(e=>e.event==='mug_policy_selected');ref=i>=0?seq[i]:null;const nav=[...seq.slice(0,i<0?seq.length:i)].reverse().find(e=>e.event==='base_motion_done');station=nav?{station:nav.goal}:null;}
 $('station').textContent=$('reference').textContent='—';if(station)$('station').textContent=station.station.slice(0,2).map(x=>x.toFixed(3)).join(', ');
 if(ref)$('reference').textContent=ref.reference;
 let result=s.state?.result,measure=s.state?.snapshot?.grasp_measurements;
 if(s.object==='bowl'){const e=[...events].reverse().find(e=>e.event==='rl_grasp_result');result=e?.result;measure=e?.snapshot?.grasp_measurements;}
 if(s.object==='mug'){const e=(s.sequence?.events||[]).find(e=>e.event==='mug_rl_result');result=e?{success:e.result==='success',reason:e.result}:null;measure=e?.diagnostics;}
 $('result').textContent=result?(result.success?'Verified':'Failed: '+result.reason)+(result.success&&measure?' · lift '+(measure.height*100).toFixed(1)+' cm':''):'Awaiting grasp';
 $('raw').hidden=!s.perception_ready;if(s.perception_ready){$('raw').href='runs/'+encodeURIComponent(s.run_id)+'/'+s.perception_path+'/estimate.json'}
}
async function poll(){try{const r=await fetch('api/status'+query(),{cache:'no-store'});if(!r.ok)throw Error('Service unavailable');const s=await r.json();if(lastRun!==s.run_id){lastRun=s.run_id;selectedObservation='';}const key=s.run_id+'/'+s.perception_id;if(run!==key){run=key;loaded='';visual=null;drawCloud();const c=$('image-canvas');c.getContext('2d').clearRect(0,0,c.width,c.height);$('image-view').querySelector('.empty').hidden=false;$('capture-time').textContent='Awaiting capture';$('station').textContent=$('reference').textContent='—'}apply(s);
 if(s.perception_ready&&loaded!==run){const v=await(await fetch('api/perception'+(s.perception_id?'?observation='+encodeURIComponent(s.perception_id):''),{cache:'no-store'})).json();if(v.origin_map){visual=v;loaded=run;image.src='api/perception-image?observation='+encodeURIComponent(s.perception_id)+'&run='+encodeURIComponent(s.run_id);drawCloud();const c=await(await fetch('runs/'+encodeURIComponent(s.run_id)+'/'+s.perception_path+'/camera.json')).json();$('capture-time').textContent=c.captured_at_utc?'Captured '+c.captured_at_utc:'Captured before approach';select(current)}}
 }catch(e){$('connection').textContent=e.message}setTimeout(poll,1200)}
const base=location.origin+(location.pathname.match(/^(\/user\/[^/]+\/)/)?.[1]||'/');$('lab').href=base+'files/demos/web_ui/index.html';select('rgb');poll();
