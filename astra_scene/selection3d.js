import * as THREE from 'three';
import {OrbitControls} from '/web-assets/OrbitControls.js';
import {TransformControls} from '/web-assets/TransformControls.js';
const $=id=>document.getElementById(id), run=location.pathname.split('/').pop(), api='/api/regions/'+run;
$('manual').href='/select/'+run;$('taskLink').href='/tasks/'+run;
let draftRevision=0,draftGeometry=null,draftTimer=null,draftSaving=false,draftAgain=false,draftReady=false,draftConflict=false,submitting=false,allowNavigation=false;
let lastMode='add',hasReconstruction=false,backendLabel='VGGT + 位姿校正 / GaME 局部训练',previewing=false;
let scene,camera,renderer,orbit,transform,base,current,cameraGroup,selection,sceneBounds;
let expandedProposals=false,numericDirty=false,regions=[],activeId=null,nextRegion=1,future=[],otherBoxes=[],renderProof=null;
let proposals=[],proposalMeshes=[],selectedId=null,label='手动区域',preview=null,revision=0,checkTimer=null,abortCheck=null,placing=false,tool='browse',history=[],dragStart=null,frames=[];
const boxGeometry=new THREE.BoxGeometry(1,1,1),raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
const escapeText=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function request(path,body,signal){const r=await fetch(path,{...(body!==undefined?{method:'POST',headers:{'Content-Type':'application/json','X-Astra-Request':'1'},body:JSON.stringify(body)}:{}),signal});const d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');return d;}
function status(text,kind=''){const e=$('coverage');e.textContent=text;e.className='status '+kind;}
function boxState(){return {center:selection.position.toArray(),size:selection.scale.toArray(),quaternion_xyzw:selection.quaternion.toArray(),label,selectedId};}
function workspace(){const copy=regions.map(r=>({...r}));if(activeId&&selection.visible){const i=copy.findIndex(r=>r.id===activeId);if(i>=0)copy[i]={...boxState(),id:activeId};}return structuredClone({regions:copy,activeId,mode:$('mode').value});}
function commitSelection(){if(!activeId||!selection.visible)return;const i=regions.findIndex(r=>r.id===activeId);if(i>=0)regions[i]={...boxState(),id:activeId};}
function restoreWorkspace(w){$('mode').value=w.mode||'add';lastMode=$('mode').value;modeUI();regions=structuredClone(w.regions);activeId=w.activeId;numericDirty=false;transform.enabled=true;const r=regions.find(r=>r.id===activeId);selection.visible=!!r;document.querySelectorAll('[data-tool]').forEach(b=>b.disabled=b.dataset.tool!=='browse'&&!r);$('focus').disabled=!r;$('numericFields').disabled=!r;if(r){restoreBox(r);setTool(tool);}else{transform.detach();$('selectionName').textContent='尚未选择目标';$('dimensions').textContent='添加一个或多个区域';}renderRegionList();renderOtherBoxes();syncNumbers();invalidate();if(regions.length)schedule();}
function activate(id){if(numericDirty){status('请先应用数值或撤销输入。');return;}commitSelection();activeId=id;restoreBox(regions.find(r=>r.id===id));selection.visible=true;setTool(tool);renderRegionList();renderOtherBoxes();focus();}
function renderRegionList(){const target=$('regions');target.replaceChildren();$('regionCount').textContent='已选 '+regions.length+' / 8 块';for(const r of regions){const row=document.createElement('div');row.className='region-row';const b=document.createElement('button');const verdict=preview?.regions?.find(s=>s.region_id===r.id)?.coverage;b.textContent=r.label+(verdict?verdict.ready?' ✓':' · 需补拍':'');b.title=verdict?.direction_hint||r.label;b.className=r.id===activeId?'active':'';b.onclick=()=>activate(r.id);const remove=document.createElement('button');remove.textContent='×';remove.title='移除 '+r.label;remove.setAttribute('aria-label','移除选区 '+r.label);remove.onclick=()=>{saveHistory(workspace());regions=regions.filter(x=>x.id!==r.id);restoreWorkspace({regions,activeId:activeId===r.id?regions.at(-1)?.id??null:activeId});};row.append(b,remove);target.append(row);}}
function renderOtherBoxes(){for(const m of otherBoxes){scene.remove(m);m.material.dispose();m.children.forEach(c=>{c.geometry.dispose();c.material.dispose();});}otherBoxes=[];for(const r of regions.filter(r=>r.id!==activeId)){const m=makeBox([[0,0,0],[1,1,1]],0x64c9ef,.04);m.position.fromArray(r.center);m.scale.fromArray(r.size);m.quaternion.fromArray(r.quaternion_xyzw);m.children[0].material.depthTest=false;m.children[0].renderOrder=90;otherBoxes.push(m);}}
function orientedBox(){const {center,size,quaternion_xyzw}=boxState();return {center,size,quaternion_xyzw};}
function bounds(){selection.updateMatrixWorld(true);const b=new THREE.Box3().setFromObject(selection);return [b.min.toArray(),b.max.toArray()];}
function restoreBox(s){selection.position.fromArray(s.center);selection.scale.fromArray(s.size);selection.quaternion.fromArray(s.quaternion_xyzw);label=s.label??label;selectedId=s.selectedId??null;$('selectionName').textContent=label;selection.updateMatrixWorld(true);numericDirty=false;transform.enabled=true;dimensions();renderProposals();}
function syncNumbers(){if(!regions.length||numericDirty)return;const angles=new THREE.Euler().setFromQuaternion(selection.quaternion,'XYZ');for(const a of ['x','y','z']){$('pos_'+a).value=Number(selection.position[a].toFixed(6));$('size_'+a).value=Number(selection.scale[a].toFixed(6));$('rot_'+a).value=Number(THREE.MathUtils.radToDeg(angles[a]).toFixed(6));}}
function applyNumbers(event){event?.preventDefault();if(!selection.visible)return;const pos=[],size=[],rot=[];for(const a of ['x','y','z']){pos.push($('pos_'+a).valueAsNumber);size.push($('size_'+a).valueAsNumber);rot.push($('rot_'+a).valueAsNumber);}
 if(![...pos,...size,...rot].every(Number.isFinite)||size.some(v=>v<.03||v>4)||size.reduce((p,v)=>p*v,1)>12){status('请输入有效数字；每边 0.03–4 米，总体积不超过 12 立方米。','error');return;}
 saveHistory(workspace());numericDirty=false;transform.enabled=true;selection.position.fromArray(pos);selection.scale.fromArray(size);selection.rotation.set(...rot.map(THREE.MathUtils.degToRad),'XYZ');selection.updateMatrixWorld(true);invalidate();schedule();}

function setBounds(b){selection.quaternion.identity();selection.position.fromArray(b[0].map((v,i)=>(v+b[1][i])/2));selection.scale.fromArray(b[0].map((v,i)=>b[1][i]-v));selection.updateMatrixWorld(true);dimensions();}
function dimensions(){if(selection?.visible)$('dimensions').textContent=selection.scale.toArray().map(v=>v.toFixed(2)).join(' × ')+' 米';syncNumbers();clipNearby();}
function clipNearby(){if(!current||!base)return;let planes=[];if(selection?.visible&&$('nearby').checked){const [lo,hi]=bounds();const margin=.65;planes=[new THREE.Plane(new THREE.Vector3(1,0,0),-lo[0]+margin),new THREE.Plane(new THREE.Vector3(-1,0,0),hi[0]+margin),new THREE.Plane(new THREE.Vector3(0,1,0),-lo[1]+margin),new THREE.Plane(new THREE.Vector3(0,-1,0),hi[1]+margin),new THREE.Plane(new THREE.Vector3(0,0,1),-lo[2]+margin),new THREE.Plane(new THREE.Vector3(0,0,-1),hi[2]+margin)];}
current.material.clippingPlanes=planes;base.material.clippingPlanes=planes;}
function invalidate(){commitSelection();renderProof=null;$('modelImages').replaceChildren();$('modelStatus').textContent='';revision++;preview=null;$('commit').textContent='检查并生成候选';$('commit').onclick=commit;$('commit').disabled=true;$('photos').replaceChildren();if(abortCheck)abortCheck.abort();clearTimeout(checkTimer);renderRegionList();dimensions();saveDraftSoon();$('check').disabled=!regions.length||numericDirty;status(regions.length?'范围已改变，正在等待检查…':'请添加至少一个选区。');}
function schedule(){if(numericDirty)return;clearTimeout(checkTimer);checkTimer=setTimeout(check,450);}
function saveHistory(b){history.push(b);future=[];if(history.length>30)history.shift();$('undo').disabled=false;$('redo').disabled=true;}
function setTool(mode){if(numericDirty){status('请先应用数值，或点击“撤销输入”恢复当前框。');return;}tool=mode;placing=false;if(renderer)renderer.domElement.style.cursor='auto';$('hint').textContent=mode==='browse'?'左键拖动旋转 · 滚轮缩放 · 右键拖动平移':mode==='translate'?'拖动彩色箭头移动范围，或在右侧输入中心位置':mode==='rotate'?'拖动彩色圆环旋转框，或在右侧输入角度':'拖动彩色手柄调整大小，或在右侧输入尺寸';document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===mode));if(selection.visible&&mode!=='browse'){transform.attach(selection);transform.setMode(mode);transform.setSpace(mode==='translate'?'world':'local');}else transform.detach();}
function choose(b,name='手动区域',id=null){const existing=id&&regions.find(r=>r.selectedId===id);if(existing){activate(existing.id);return;}if(regions.length>=8){status('一次最多 8 块区域，请先移除不需要的选区。','error');return;}saveHistory(workspace());commitSelection();numericDirty=false;transform.enabled=true;$('numericFields').disabled=false;selectedId=id;label=name;activeId='region_'+nextRegion++;selection.visible=true;setBounds(b);regions.push({...boxState(),id:activeId});$('selectionName').textContent=label;document.querySelectorAll('[data-tool]').forEach(e=>e.disabled=false);$('focus').disabled=false;$('check').disabled=false;setTool('translate');invalidate();renderOtherBoxes();renderProposals();focus();schedule();}
function focus(direction){const target=selection.visible?selection.position.clone():new THREE.Vector3().fromArray(sceneBounds[0].map((v,i)=>(v+sceneBounds[1][i])/2));const size=selection.visible?Math.max(...selection.scale.toArray())*2.7:new THREE.Vector3().fromArray(sceneBounds[1]).sub(new THREE.Vector3().fromArray(sceneBounds[0])).length()*.8;
 const dir=direction==='top'?new THREE.Vector3(.001,0,1):direction==='side'?new THREE.Vector3(0,-1,.05):new THREE.Vector3(1,-1,.65).normalize();
 orbit.target.copy(target);camera.position.copy(target).addScaledVector(dir,Math.max(size,1.5));camera.near=.01;camera.far=200;camera.updateProjectionMatrix();orbit.update();}
function renderProposals(){const q=$('search').value.trim().toLowerCase(),matched=proposals.filter(p=>(p.label+' '+p.source_label).toLowerCase().includes(q)),shown=q||expandedProposals?matched:matched.slice(0,5);$('moreProposals').hidden=!!q||matched.length<=5;$('moreProposals').textContent=expandedProposals?'收起其他建议':'显示其他 '+(matched.length-5)+' 个建议';for(const m of proposalMeshes)m.visible=selection.visible?false:shown.some(p=>p.id===m.userData.proposal.id);$('proposals').replaceChildren();for(const p of shown){const b=document.createElement('button');b.className='proposal'+(selectedId===p.id?' active':'');b.innerHTML='<img alt="" src="/region-images/'+run+'/region_proposals/'+p.thumbnail+'"><div class="caption"><b>'+escapeText(p.label)+'</b><small>'+p.support_views+' 个视角提供预框</small></div>';b.onclick=()=>choose(p.bounds,p.label,p.id);$('proposals').append(b);}if(!shown.length)$('proposals').innerHTML='<p class="empty">没有匹配的建议。可以自己圈选区域。</p>';}
function makeBox(b,color,opacity){const mesh=new THREE.Mesh(boxGeometry,new THREE.MeshBasicMaterial({color,transparent:true,opacity,depthWrite:false}));mesh.position.fromArray(b[0].map((v,i)=>(v+b[1][i])/2));mesh.scale.fromArray(b[0].map((v,i)=>b[1][i]-v));const edge=new THREE.LineSegments(new THREE.EdgesGeometry(boxGeometry),new THREE.LineBasicMaterial({color,transparent:true,opacity:.8}));mesh.add(edge);scene.add(mesh);return mesh;}
async function loadProposals(start=false){try{let d=await request(api+'/proposals');if(d.state==='not_prepared'&&start){await request(api+'/detect',{});d={state:'running'};}if(d.state==='ready'){proposals=(d.proposals||[]).filter(p=>p.bounds);for(const m of proposalMeshes){scene.remove(m);m.material.dispose();m.children.forEach(c=>{c.geometry.dispose();c.material.dispose();});}proposalMeshes=proposals.map(p=>{const m=makeBox(p.bounds,0x63bca0,.035);m.userData.proposal=p;return m;});renderProposals();}else if(d.state==='running'||d.state==='pending'){ $('proposals').innerHTML='<p class="empty">正在自动找物体…<br>你可以先手动选区。</p>';setTimeout(()=>loadProposals(),3500);}else{$('proposals').innerHTML='<p class="empty">'+escapeText(d.error||'预框暂未完成，可以手动选区。')+'</p>';}}catch(e){$('proposals').innerHTML='<p class="empty">'+escapeText(e.message)+'</p>';}}
async function check(){if(!selection?.visible||numericDirty)return;clearTimeout(checkTimer);const version=revision;if(abortCheck)abortCheck.abort();abortCheck=new AbortController();$('commit').disabled=true;$('check').disabled=true;status('正在检查这个范围的多视角表面和留出画面…');try{const d=await request(api+'/preview',{regions:workspace().regions.map(r=>({id:r.id,label:r.label,oriented_box:{center:r.center,size:r.size,quaternion_xyzw:r.quaternion_xyzw}})),mode:$('mode').value},abortCheck.signal);if(version!==revision)return;preview=d;renderRegionList();const c=d.coverage;status($('mode').value==='delete'?'已选 '+d.regions.length+' 个删除区域。下一步：查看删除预览。':c.message+'。'+(d.regions?.length>1?' 每块区域已单独检查。':'\n'+c.direction_hint),c.ready?'good':'error');$('coverage').style.whiteSpace='pre-line';$('commit').disabled=!c.ready; $('commit').textContent=!renderProof&&($('mode').value==='delete'||hasReconstruction)?'预览并检查候选':'检查并生成候选';const ids=c.supported_views;let selected=ids.length<=3?ids:[ids[0],ids[Math.floor(ids.length/2)],ids.at(-1)];$('photos').replaceChildren();for(const i of selected){const img=document.createElement('img');img.className='photo';img.alt='拍摄视角 '+i+' 的选区核对';img.src='/region-images/'+run+'/region_previews/'+d.preview_id+'/view_'+String(i).padStart(3,'0')+'.jpg';img.onclick=()=>{$('largePhoto').src=img.src;$('photoDialog').showModal();};const caption=document.createElement('div');caption.className='photo-label';caption.textContent='真实视频 · 视角 '+i;$('photos').append(img,caption);}if(ids.length>3){const detail=document.createElement('details');const sum=document.createElement('summary');sum.textContent='查看其他 '+(ids.length-selected.length)+' 个方向';detail.append(sum);for(const i of ids.filter(i=>!selected.includes(i))){const img=document.createElement('img');img.className='photo';img.loading='lazy';img.alt='拍摄视角 '+i;img.src='/region-images/'+run+'/region_previews/'+d.preview_id+'/view_'+String(i).padStart(3,'0')+'.jpg';img.onclick=()=>{$('largePhoto').src=img.src;$('photoDialog').showModal();};detail.append(img);}$('photos').append(detail);}
 if(cameraGroup)for(const m of cameraGroup.children)m.material.color.set(c.supported_views.includes(m.userData.view)?0x59e0a0:0x65818c);
}catch(e){if(e.name!=='AbortError'&&version===revision)status(e.message,'error');}finally{if(version===revision)$('check').disabled=false;}}
async function addCloud(desc,isBase){const r=await fetch('/cloud-data/'+run+'/'+desc.file);if(!r.ok)throw Error('点云读取失败');const data=new Float32Array(await r.arrayBuffer());if(data.length!==desc.count*6)throw Error('点云数据不完整');const positions=new Float32Array(desc.count*3),colors=new Float32Array(desc.count*3);for(let i=0;i<desc.count;i++){positions.set(data.subarray(i*6,i*6+3),i*3);colors.set(isBase?[.35,.43,.41]:data.subarray(i*6+3,i*6+6),i*3);}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));const cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:isBase?.018:.025,vertexColors:true,sizeAttenuation:true}));scene.add(cloud);return cloud;}
async function init(){try{
 scene=new THREE.Scene();scene.background=new THREE.Color(0x152323);camera=new THREE.PerspectiveCamera(50,1,.01,200);camera.up.set(0,0,1);renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.localClippingEnabled=true;$('stage').prepend(renderer.domElement);
 orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.minDistance=.25;orbit.maxDistance=60;
 transform=new TransformControls(camera,renderer.domElement);transform.setSize(.8);scene.add(transform.getHelper());
 selection=makeBox([[0,0,0],[1,1,1]],0xffb43b,.075);selection.visible=false;selection.renderOrder=100;selection.material.depthTest=false;selection.children[0].material.depthTest=false;selection.children[0].material.opacity=1;selection.children[0].renderOrder=101;
 transform.addEventListener('dragging-changed',e=>{orbit.enabled=!e.value;if(e.value){dragStart=workspace();invalidate();}else{if(dragStart)saveHistory(dragStart);dragStart=null;for(const a of ['x','y','z'])selection.scale[a]=Math.max(.03,Math.min(4,Math.abs(selection.scale[a])));commitSelection();dimensions();saveDraftSoon();schedule();}});
 transform.addEventListener('objectChange',()=>{dimensions();if(!transform.dragging){invalidate();schedule();}});
 const resize=()=>{const w=$('stage').clientWidth,h=$('stage').clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe($('stage'));resize();renderer.setAnimationLoop(()=>{orbit.update();renderer.render(scene,camera);});
 const [manifest,f,task]=await Promise.all([request(api+'/cloud'),request(api+'/frames'),request('/api/tasks/'+run)]);hasReconstruction=task.has_reconstruction;backendLabel=task.backend;modeUI();frames=f.frames;sceneBounds=manifest.bounds;[base,current]=await Promise.all([addCloud(manifest.base,true),addCloud(manifest.current,false)]);
 cameraGroup=new THREE.Group();for(const c of manifest.cameras){const m=new THREE.Mesh(new THREE.SphereGeometry(.018,8,6),new THREE.MeshBasicMaterial({color:0x65a8db}));m.position.fromArray(c.position);m.userData.view=c.view;cameraGroup.add(m);}scene.add(cameraGroup);focus();await restoreDraft();$('loading').remove();
 let down=null;renderer.domElement.addEventListener('pointerdown',e=>{down={x:e.clientX,y:e.clientY};});
 renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>5||tool!=='browse'||e.button!==0)return;const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
 if(placing){raycaster.params.Points.threshold=.045;const source=$('mode').value==='delete'?base:current;const hits=source.visible?raycaster.intersectObject(source):[];const hit=hits.find(h=>{if(!selection.visible||!$('nearby').checked)return true;const [lo,hi]=bounds(),p=h.point.toArray();return p.every((v,i)=>v>=lo[i]-.65&&v<=hi[i]+.65);});if(!hit){$('hint').textContent='请点击可见点云；删除模式请显示原场景，可先放大目标';return;}const p=hit.point;choose([[p.x-.35,p.y-.35,p.z-.5],[p.x+.35,p.y+.35,p.z+.5]]);setTool('translate');}
 else{const hit=raycaster.intersectObjects(proposalMeshes.filter(m=>m.visible),false)[0];if(hit){const p=hit.object.userData.proposal;choose(p.bounds,p.label,p.id);}}});
 loadProposals(false);
}catch(e){$('loading').textContent='三维场景暂时无法打开：'+e.message+'。可使用右上角二维框选。';}}
$('numericForm').onsubmit=applyNumbers;$('numericForm').addEventListener('input',()=>{numericDirty=true;transform.enabled=false;invalidate();$('check').disabled=true;status('数值尚未应用。点击“应用数值”或按 Enter 更新范围。');});
$('cancelNumbers').onclick=()=>{numericDirty=false;transform.enabled=true;syncNumbers();invalidate();schedule();};
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>sceneBounds&&focus(b.dataset.view));
$('focus').onclick=()=>focus();$('reset').onclick=()=>{if(!sceneBounds)return;const visible=selection.visible;selection.visible=false;focus();selection.visible=visible;$('nearby').checked=false;clipNearby();};
$('newBox').onclick=()=>{if(!current)return;if(numericDirty){status('请先应用数值或撤销输入。');return;}setTool('browse');$('nearby').checked=false;clipNearby();placing=true;renderer.domElement.style.cursor='crosshair';status('在场景中点击目标，放置新的选区。');$('hint').textContent='点击要添加的区域：更新选手机表面，删除选原场景';};
$('undo').onclick=()=>{if(!history.length)return;future.push(workspace());restoreWorkspace(history.pop());$('undo').disabled=!history.length;$('redo').disabled=false;};
$('redo').onclick=()=>{if(!future.length)return;history.push(workspace());restoreWorkspace(future.pop());$('redo').disabled=!future.length;$('undo').disabled=false;};
$('search').oninput=renderProposals;$('moreProposals').onclick=()=>{expandedProposals=!expandedProposals;renderProposals();};$('nearby').onchange=clipNearby;$('detect').onclick=()=>loadProposals(true);$('mode').onchange=()=>{const previous=workspace();previous.mode=lastMode;saveHistory(previous);lastMode=$('mode').value;modeUI();if(selection.visible){invalidate();schedule();}else saveDraftSoon();};
function modeUI(){const del=$('mode').value==='delete';$('deleteNote').hidden=!del;$('renderModel').textContent=del?'预览删除前后':hasReconstruction?'预览更新前后':'查看原模型';
if(base&&current){base.visible=true;$('showBase').checked=true;current.visible=!del;$('showCurrent').checked=!del;}}

$('check').onclick=check;$('closePhoto').onclick=()=>$('photoDialog').close();
$('showBase').onchange=()=>{if(base)base.visible=$('showBase').checked;};$('showCurrent').onchange=()=>{if(current)current.visible=$('showCurrent').checked;};$('showCameras').onchange=()=>{if(cameraGroup)cameraGroup.visible=$('showCameras').checked;};
async function commit(){
 if(submitting||previewing||numericDirty||!preview?.coverage.ready)return;
 if(($('mode').value==='delete'||hasReconstruction)&&!renderProof){await $('renderModel').onclick();if(!renderProof)return;}
 const mode=$('mode').value,count=regions.length;
 $('confirmSummary').textContent=({add:'新增',replace:'替换',delete:'删除'})[mode]+' '+count+' 个区域。'+(mode==='delete'?'模型按范围删除；导航只有腾空证据通过的地方才会清除。':'只修改所选区域，范围外保留原场景。');
 $('confirmMethod').textContent='实际方法：'+(mode==='delete'?'原模型裁剪':backendLabel);
 $('confirmError').textContent='';$('confirmImages').replaceChildren(...[...$('modelImages').children].map(x=>x.cloneNode(true)));$('confirmDialog').showModal();
}
$('cancelCommit').onclick=()=>$('confirmDialog').close();
$('confirmCommit').onclick=async()=>{
 if(submitting||!preview?.coverage.ready||numericDirty)return;
 const version=revision,id=preview.preview_id;
 submitting=true;$('confirmCommit').disabled=true;
 try{
  const d=await request(api+'/update',{preview_id:id,render_id:renderProof});
  await persistDraft();
  allowNavigation=true;location.href='/tasks/'+d.id;
 }catch(e){$('confirmError').textContent=e.message;$('confirmCommit').disabled=false;submitting=false;}
};

$('commit').onclick=commit;
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='Escape')setTool('browse');if(selection?.visible&&!e.ctrlKey&&!e.metaKey){if(e.key.toLowerCase()==='w')setTool('translate');if(e.key.toLowerCase()==='e')setTool('rotate');if(e.key.toLowerCase()==='r')setTool('scale');}if((e.ctrlKey||e.metaKey)&&(e.key.toLowerCase()==='z'||e.key.toLowerCase()==='y')){e.preventDefault();$(e.shiftKey||e.key.toLowerCase()==='y'?'redo':'undo').click();}});
$('hideHint').onclick=()=>{$('hintBox').hidden=true;};$('showHelp').onclick=()=>{$('hintBox').hidden=!$('hintBox').hidden;};
$('renderModel').onclick=async()=>{if(!camera||previewing)return;if($('mode').value==='delete'&&!preview){$('modelStatus').textContent='请先等待选区检查完成，再预览删除效果。';return;}const version=revision;previewing=true;$('commit').disabled=true;$('renderModel').disabled=true;$('modelStatus').textContent='正在渲染原模型…';try{camera.updateMatrixWorld(true);const m=camera.matrixWorld.clone().multiply(new THREE.Matrix4().makeScale(1,-1,-1)),e=m.elements;const pose=Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>e[c*4+r]));const width=640,height=Math.min(1024,Math.max(240,Math.round(width/camera.aspect))),focal=height/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)));
 const d=await request(api+'/render',{pose,width,height,fx:focal,preview_id:preview?.preview_id});
 let out=d;while(out.state==='running'){await new Promise(r=>setTimeout(r,1200));out=await request(api+'/render/'+d.id);if(version!==revision)return;}
 if(out.state!=='ready')throw Error(out.error||'渲染失败');
 if(version!==revision)return;renderProof=d.id;$('modelImages').replaceChildren();for(const [file,title] of ($('mode').value==='delete'||out.candidate_preview?[['before.jpg','原 3DGS'],['after.jpg',$('mode').value==='delete'?'删除后候选':'更新后候选']]:[['before.jpg','原 3DGS']])){const div=document.createElement('div');const img=document.createElement('img');img.className='photo';img.alt=title;img.src='/region-images/'+run+'/render_previews/'+d.id+'/'+file;img.onclick=()=>{$('largePhoto').src=img.src;$('photoDialog').showModal();};const p=document.createElement('p');p.textContent=title;div.append(p,img);$('modelImages').append(div);}
 $('modelStatus').textContent=out.deleted_gaussians?'删除预览已生成。请确认目标已移除且背景完整；没建模的遮挡面不会自动补全。':out.candidate_preview?'候选预览已生成。请检查物体、地面与边界；换方向后可重新预览。':'原模型渲染已完成。旋转观察方向后可重新渲染。';if(preview?.coverage.ready&&($('mode').value!=='delete'||out.deleted_gaussians>0)){$('commit').disabled=false;$('commit').textContent='检查并生成候选';}
}catch(e){$('modelStatus').textContent=e.message;}finally{previewing=false;$('renderModel').disabled=false;if(preview?.coverage.ready)$('commit').disabled=false;}};
const localKey='astra:editor:v3:'+run;
function saveDraftSoon(){
 if(!draftReady||numericDirty||draftConflict)return;
 try{localStorage.setItem(localKey,JSON.stringify({workspace:workspace(),revision:draftRevision,geometry_key:draftGeometry,updated:Date.now()/1000,pending:true}));$('draftState').textContent='草稿待同步…';}catch{$('draftState').textContent='本机存储不可用，正在同步服务器…';}
 clearTimeout(draftTimer);draftTimer=setTimeout(persistDraft,600);
}
async function persistDraft(){
 if(!draftReady||numericDirty||draftConflict)return;
 if(draftSaving){draftAgain=true;return;}
 draftSaving=true;
 const w=workspace();
 try{const d=await request(api+'/draft',{workspace:w,revision:draftRevision});draftRevision=d.revision;
 try{localStorage.setItem(localKey,JSON.stringify({...d,pending:false}));}catch{}
 $('draftState').textContent='草稿已保存 · 刷新后可继续';
 }catch(e){$('draftState').textContent='草稿未同步：'+e.message;draftConflict=e.message.includes('其他页面');}
 finally{draftSaving=false;if(draftAgain){draftAgain=false;saveDraftSoon();}}
}
async function restoreDraft(){
 try{
 const d=await request(api+'/draft');draftRevision=d.revision||0;draftGeometry=d.geometry_key;
 let local=null;try{local=JSON.parse(localStorage.getItem(localKey)||'null');}catch{}
 let w=d.stale?null:d.workspace;
 if(local?.pending&&local.geometry_key===draftGeometry&&local.revision===draftRevision)w=local.workspace;
 if(w){nextRegion=Math.max(0,...w.regions.map(r=>Number(r.id.replace('region_',''))||0))+1;restoreWorkspace(w);if(regions.length)focus();$('draftState').textContent='已恢复上次选区';}
 else $('draftState').textContent=d.stale?'几何已更新，请重新选择区域':'选区会自动保存';
 draftReady=true;
 if(local?.pending&&w===local.workspace)saveDraftSoon();
 }catch(e){$('draftState').textContent='无法读取草稿：'+e.message;draftReady=false;}
}
window.addEventListener('beforeunload',e=>{if(!allowNavigation&&(numericDirty||submitting||draftSaving||draftConflict)){e.preventDefault();e.returnValue='';}});
init();
