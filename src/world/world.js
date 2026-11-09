import * as THREE from "three";import {segmentDistanceSq,clamp,randomRange,seededRandom} from "../core/math.js";
export class World{
constructor(canvas){this.scene=new THREE.Scene();this.quality=this.detectQuality();this.scene.background=new THREE.Color(0x9bb6bd);this.scene.fog=new THREE.Fog(0x9bb6bd,this.quality.low?72:this.quality.mid?90:110,this.quality.low?190:this.quality.mid?245:300);this.camera=new THREE.PerspectiveCamera(78,innerWidth/innerHeight,.1,this.quality.low?360:480);this.renderer=new THREE.WebGLRenderer({canvas,antialias:false,powerPreference:"high-performance",stencil:false,depth:true,preserveDrawingBuffer:false});this.renderer.setPixelRatio(this.quality.pixelRatio);this.renderer.setSize(innerWidth,innerHeight,false);this.renderer.info.autoReset=true;this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.08;this.renderer.shadowMap.enabled=false;this.obstacles=[];this.rng=seededRandom(981);this.scene.add(new THREE.HemisphereLight(0xdde9e8,0x35422f,this.quality.low?1.25:1.55));const sun=new THREE.DirectionalLight(0xffe2bd,2.4);sun.position.set(-70,110,60);sun.castShadow=false;this.scene.add(sun);this.terrain();this.environment();this.createTransportPlane();addEventListener("resize",()=>this.resize())}
detectQuality(){const mobile=matchMedia("(pointer:coarse)").matches,cores=navigator.hardwareConcurrency||4,memory=navigator.deviceMemory||4;const low=mobile&&(cores<=4||memory<=3),mid=mobile&&(cores<=6||memory<=4);return{tier:low?"LOW":mid?"BALANCED":"HIGH",pixelRatio:Math.min(devicePixelRatio,low?.65:mid?.82:mobile?1:1.35),low,mid,mobile}}
terrain(){const segments=this.quality.low?48:this.quality.mid?64:84;const g=new THREE.PlaneGeometry(420,420,segments,segments),p=g.attributes.position,c=[];for(let i=0;i<p.count;i++){const x=p.getX(i),z=-p.getY(i),h=Math.sin(x*.04)*.2+Math.cos(z*.035)*.18;p.setZ(i,h);const k=(Math.sin(x*.13+z*.09)+Math.cos(z*.15))*.02;c.push(.28+k,.38+k,.22+k*.5)}g.setAttribute("color",new THREE.Float32BufferAttribute(c,3));g.rotateX(-Math.PI/2);g.computeVertexNormals();const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1}));m.receiveShadow=true;this.scene.add(m)}
add(x,z,r,h,kind){this.obstacles.push({x,z,r,height:h,kind})}
tree(x,z){const g=new THREE.Group(),b=new THREE.Mesh(new THREE.CylinderGeometry(.18,.3,2.6,8),new THREE.MeshStandardMaterial({color:0x59422d}));b.position.y=1.3;g.add(b);for(let i=0;i<3;i++){const c=new THREE.Mesh(new THREE.ConeGeometry(1.7-i*.25,2.7,8),new THREE.MeshStandardMaterial({color:[0x214a2d,0x2d5b37,0x37683f][i]}));c.position.y=2.5+i*.62;g.add(c)}g.position.set(x,0,z);this.scene.add(g);this.add(x,z,1.2,3,"tree")}
rock(x,z){const m=new THREE.Mesh(new THREE.DodecahedronGeometry(1,1),new THREE.MeshStandardMaterial({color:0x737a75,roughness:1}));m.position.set(x,.6,z);m.scale.set(1.4,.8,1.1);this.scene.add(m);this.add(x,z,1.3,1.5,"rock")}
building(x,z,w,d,h){
 const g=new THREE.Group(),wallMat=new THREE.MeshStandardMaterial({color:0x77796e,roughness:.94}),insideMat=new THREE.MeshStandardMaterial({color:0x555b50,roughness:1}),roofMat=new THREE.MeshStandardMaterial({color:0x39433e,roughness:.9}),floorMat=new THREE.MeshStandardMaterial({color:0x716d5e,roughness:1}),trim=new THREE.MeshStandardMaterial({color:0x343d38,roughness:.8});
 const addBox=(mat,px,py,pz,sx,sy,sz)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(sx,sy,sz),mat);m.position.set(px,py,pz);m.castShadow=true;m.receiveShadow=true;g.add(m)};
 // Floor and interior ceiling/roof.
 addBox(floorMat,0,.12,0,w,.24,d);
 addBox(roofMat,0,h+.12,0,w+.8,.32,d+.8);
 // Eaves, roof fascia and corner pillars add depth without expensive materials.
 addBox(trim,0,h+.32,-d/2,w+.55,.16,.18);addBox(trim,0,h+.32,d/2,w+.55,.16,.18);
 for(const sx of [-1,1])for(const sz of [-1,1])addBox(trim,sx*(w/2-.12),h/2,sz*(d/2-.12),.22,h,.22);
 // Rear wall and side walls are segmented around window openings.
 const t=.34,winY=h*.56,winH=.78,doorW=1.45,doorH=2.15;
 addBox(wallMat,0,h/2,-d/2,w,h,t);
 for(const side of [-1,1]){
  addBox(wallMat,side*w/2,h/2,0,t,h,d);
  // Window lintels/sills on both side walls.
  addBox(insideMat,side*w/2,winY-winH/2,0,t,.12,d*.72);
  addBox(wallMat,side*w/2,winY+winH/2,0,t,.12,d*.72);
 }
 // Front facade: central doorway, two windows and their lintels/sills.
 const segment=(cx,sx)=>addBox(wallMat,cx,h/2,d/2,sx,h,t);
 segment(-(doorW/2+(w-doorW)/4),(w-doorW)/2);
 segment(doorW/2+(w-doorW)/4,(w-doorW)/2);
 addBox(wallMat,0,doorH+(h-doorH)/2,d/2,doorW,h-doorH,t);
 // Window frames and dark openings give the facade readable depth.
 for(const side of [-1,1]){
  const wx=side*w*.29,wy=h*.62,ww=w*.19,wh=.85;
  addBox(trim,wx,wy,d/2+.025,ww+.16,wh+.16,.08);
  addBox(insideMat,wx,wy,d/2+.07,ww,wh,.04);
  addBox(trim,wx-ww/2,wy,d/2+.12,.07,wh+.12,.08);addBox(trim,wx+ww/2,wy,d/2+.12,.07,wh+.12,.08);
  addBox(wallMat,wx,wy-wh/2,d/2,ww,.1,t+.04);
  addBox(wallMat,wx,wy+wh/2,d/2,ww,.1,t+.04);
 }
 // Repeated storey bands, window bays and balcony rails for 3–4 floor apartment blocks.
 const floors=Math.max(1,Math.round(h/3.1));
 for(let floor=1;floor<floors;floor++){
  const y=floor*3.05;
  addBox(floorMat,0,y,w-.5,d-.5,.16);
  addBox(trim,0,y+1.45,d/2+.13,w*.9,.12,.22);
  for(const bx of [-w*.23,w*.23]){
   addBox(trim,bx,y+1.0,d/2+.32,.09,.9,.09);
   addBox(trim,bx,y+.55,d/2+.32,.09,.08,.09);
  }
  for(const wx of [-w*.3,0,w*.3]){
   addBox(trim,wx,y+1.85,d/2+.035,w*.18,.98,.08);
   addBox(insideMat,wx,y+1.85,d/2+.085,w*.15,.78,.035);
   addBox(trim,wx-w*.09,y+1.85,d/2+.13,.045,.9,.05);
   addBox(trim,wx+w*.09,y+1.85,d/2+.13,.045,.9,.05);
  }
 }
 // Simple interior partition creates rooms while leaving a passage.
 addBox(insideMat,-w*.18,h*.32,0,.18,h*.64,d*.48);
 // Door frame and a recessed, hinged door leaf.
 addBox(trim,-doorW/2-.08,doorH/2,d/2+.12,.14,doorH,.16);addBox(trim,doorW/2+.08,doorH/2,d/2+.12,.14,doorH,.16);
 const door=new THREE.Mesh(new THREE.BoxGeometry(doorW*.72,doorH*.92,.1),insideMat);door.position.set(doorW*.13,doorH*.46,d/2+.08);g.add(door);
 g.position.set(x,0,z);this.scene.add(g);
 // Segment-based perimeter collision; doorway remains passable.
 const halfW=w/2,halfD=d/2;
 for(let i=0;i<5;i++){const px=x-halfW+(i+.5)*w/5; if(Math.abs(px-x)>doorW/2+.45)this.add(px,z+halfD,Math.min(w/10,.9),h,"buildingWall")}
 for(let i=0;i<5;i++){const px=x-halfW+(i+.5)*w/5;this.add(px,z-halfD,Math.min(w/10,.9),h,"buildingWall")}
 for(const side of [-1,1])for(let i=0;i<4;i++){const pz=z-halfD+(i+.5)*d/4;this.add(x+side*halfW,pz,Math.min(d/8,.9),h,"buildingWall")}
}
environment(){
 const buildings=[[-92,-72,15,12,12],[-72,-72,15,12,12],[-52,-72,15,12,15],[-92,-54,15,12,15],[-72,-54,15,12,12],[-52,-54,15,12,12],[62,62,16,13,15],[83,62,16,13,12],[104,62,16,13,15],[62,82,16,13,12],[83,82,16,13,15],[104,82,16,13,12],[105,-108,15,12,12],[125,-108,15,12,15],[145,-108,15,12,12],[105,-89,15,12,15],[125,-89,15,12,12],[145,-89,15,12,15],[-145,105,14,12,12],[-126,105,14,12,15],[-107,105,14,12,12],[-145,124,14,12,15],[-126,124,14,12,12],[-107,124,14,12,15]];this.buildingLots=buildings;
 const treeCount=this.quality.low?150:this.quality.mid?250:390;
 const rockCount=this.quality.low?28:this.quality.mid?44:64;
 const trunkGeo=new THREE.CylinderGeometry(.16,.27,2.5,6);
 const leafGeo=[new THREE.ConeGeometry(1.45,2.35,6),new THREE.ConeGeometry(1.2,2.05,6),new THREE.ConeGeometry(.95,1.8,6)];
 const trunkMat=new THREE.MeshStandardMaterial({color:0x55402c,roughness:1});
 const leafMats=[0x1e472c,0x285735,0x35643b].map(color=>new THREE.MeshStandardMaterial({color,roughness:1}));
 const trunks=new THREE.InstancedMesh(trunkGeo,trunkMat,treeCount);
 const leaves=leafGeo.map((geo,i)=>new THREE.InstancedMesh(geo,leafMats[i],treeCount));
 const dummy=new THREE.Object3D(), positions=[];
 let placed=0,attempts=0;
 while(placed<treeCount&&attempts<treeCount*5){
  attempts++;
  const x=randomRange(this.rng,-198,198),z=randomRange(this.rng,-198,198);
  // Keep the spawn, roads and building footprints readable and navigable.
  if(Math.hypot(x,z)<24||Math.abs(z)<5||Math.abs(x+70)<12||Math.abs(x-75)<13||Math.abs(z-55)<13||Math.abs(z+45)<12)continue;
  let nearBuilding=false;
  for(const [bx,bz,bw,bd] of (this.buildingLots||[]))if(Math.abs(x-bx)<bw/2+5&&Math.abs(z-bz)<bd/2+5){nearBuilding=true;break}
  if(nearBuilding)continue;
  const scale=randomRange(this.rng,.78,1.28),yaw=randomRange(this.rng,0,Math.PI*2);
  dummy.position.set(x,1.25*scale,z);dummy.rotation.set(0,yaw,0);dummy.scale.setScalar(scale);dummy.updateMatrix();trunks.setMatrixAt(placed,dummy.matrix);
  for(let k=0;k<3;k++){dummy.position.set(x,(2.25+k*.58)*scale,z);dummy.rotation.set(0,yaw+k*.13,0);dummy.scale.setScalar(scale*(1-k*.12));dummy.updateMatrix();leaves[k].setMatrixAt(placed,dummy.matrix)}
  positions.push([x,z]);this.add(x,z,.95*scale,3.2*scale,"tree");placed++;
 }
 trunks.count=placed;trunks.instanceMatrix.needsUpdate=true;this.scene.add(trunks);
 for(const mesh of leaves){mesh.count=placed;mesh.instanceMatrix.needsUpdate=true;this.scene.add(mesh)}
 // Rocks share one geometry/material and one draw call.
 const rockGeo=new THREE.DodecahedronGeometry(1,0),rockMat=new THREE.MeshStandardMaterial({color:0x737a75,roughness:1});
 const rocks=new THREE.InstancedMesh(rockGeo,rockMat,rockCount);let rockPlaced=0;
 for(let i=0;i<rockCount*3&&rockPlaced<rockCount;i++){
  const x=randomRange(this.rng,-198,198),z=randomRange(this.rng,-198,198);
  if(Math.hypot(x,z)<22||Math.abs(z)<7)continue;
  const sc=randomRange(this.rng,.55,1.35);dummy.position.set(x,.55*sc,z);dummy.rotation.set(randomRange(this.rng,-.2,.2),randomRange(this.rng,0,6.28),0);dummy.scale.set(1.4*sc,.8*sc,1.1*sc);dummy.updateMatrix();rocks.setMatrixAt(rockPlaced++,dummy.matrix);this.add(x,z,1.1*sc,1.4*sc,"rock");
 }
 rocks.count=rockPlaced;rocks.instanceMatrix.needsUpdate=true;this.scene.add(rocks);
 // Three compact dirt roads and two connecting tracks; no physics-heavy meshes.
 const roadMat=new THREE.MeshStandardMaterial({color:0x716d5e,roughness:1});
 const road=(x,z,length,width,angle=0)=>{const m=new THREE.Mesh(new THREE.PlaneGeometry(width,length),roadMat);m.rotation.set(-Math.PI/2,0,angle);m.position.set(x,.035,z);this.scene.add(m)};
 road(0,0,390,5,0);road(-95,0,175,4,Math.PI/2);road(88,4,190,4,Math.PI/2);road(-42,-88,115,3,Math.PI/2);road(54,105,100,3,Math.PI/2);
 this.createGrass();
 for(const [x,z,w,d,h] of buildings)this.building(x,z,w,d,h);
 const ring=new THREE.Mesh(new THREE.TorusGeometry(178,.18,8,120),new THREE.MeshBasicMaterial({color:0xdac66d}));ring.rotation.x=Math.PI/2;ring.position.y=.3;this.scene.add(ring);this.zoneRing=ring;
}


createGrass(){
 const count=this.quality.low?2600:this.quality.mid?5200:9000;
 const geo=new THREE.PlaneGeometry(.13,.52,1,1);geo.translate(0,.26,0);
 const mat=new THREE.MeshStandardMaterial({color:0x78934e,roughness:1,side:THREE.DoubleSide});
 const grass=new THREE.InstancedMesh(geo,mat,count),dummy=new THREE.Object3D();let placed=0,tries=0;
 while(placed<count&&tries<count*5){
  tries++;const x=randomRange(this.rng,-205,205),z=randomRange(this.rng,-205,205);
  if(Math.abs(z)<3.8||Math.abs(x+95)<2.5||Math.abs(x-88)<2.5||Math.hypot(x,z)<18)continue;
  let blocked=false;for(const [bx,bz,bw,bd] of this.buildingLots)if(Math.abs(x-bx)<bw/2+2&&Math.abs(z-bz)<bd/2+2){blocked=true;break}if(blocked)continue;
  const scale=randomRange(this.rng,.55,1.35);dummy.position.set(x,0,z);dummy.rotation.set(0,randomRange(this.rng,0,Math.PI),randomRange(this.rng,-.12,.12));dummy.scale.set(scale,scale,scale);dummy.updateMatrix();grass.setMatrixAt(placed++,dummy.matrix);
 }
 grass.count=placed;grass.instanceMatrix.needsUpdate=true;grass.frustumCulled=true;this.scene.add(grass);this.grass=grass;
}

createTransportPlane(){
 const plane=new THREE.Group(),mat=(color,rough=.7,metal=.15)=>new THREE.MeshStandardMaterial({color,roughness:rough,metalness:metal});
 const skin=mat(0xb9c0b8,.55,.28),dark=mat(0x303a3a,.65,.3),glass=mat(0x172b34,.2,.55),rubber=mat(0x171a1a,.9),mark=mat(0x46564b,.8);
 const add=(geo,m,x,y,z,rx=0,ry=0,rz=0)=>{const o=new THREE.Mesh(geo,m);o.position.set(x,y,z);o.rotation.set(rx,ry,rz);o.castShadow=true;o.receiveShadow=true;plane.add(o);return o};
 // Long pressurized fuselage with a tapered nose and tail cone.
 add(new THREE.CylinderGeometry(1.8,2.15,24,20,1,true),skin,0,0,0,Math.PI/2);
 // Reinforced cargo belly, cockpit glazing, nose radome and rear loading ramp.
 add(new THREE.BoxGeometry(2.5,1.15,13),dark,0,-1.35,1);
 add(new THREE.SphereGeometry(1.55,16,10),glass,0,.65,-11.3);
 add(new THREE.BoxGeometry(2.4,.22,5.8),dark,0,-1.75,10.4);
 add(new THREE.SphereGeometry(1.8,16,10),skin,0,0,-12);
 add(new THREE.ConeGeometry(1.8,5,16),skin,0,0,13,Math.PI/2);
 add(new THREE.BoxGeometry(1.3,1.6,4),glass,0,1.05,-10.2);
 // High wing, reinforced center section and control surfaces.
 add(new THREE.BoxGeometry(50,.34,6.4),skin,0,.2,-1);
 add(new THREE.BoxGeometry(47,.12,.8),dark,0,-.04,2.15);
 add(new THREE.BoxGeometry(47,.12,.8),dark,0,-.04,-4.1);
 add(new THREE.BoxGeometry(13,.5,6.5),mark,0,.05,-1);
 for(const side of [-1,1]){
  // Four turboprop nacelles with spinner and propeller blades.
  for(const x of [side*8,side*17]){
   add(new THREE.CylinderGeometry(.72,.82,4.2,12),dark,x,-.35,-1,Math.PI/2);
   add(new THREE.ConeGeometry(.72,.8,12),skin,x,-.35,-3.25,Math.PI/2);
   const prop=new THREE.Group();prop.position.set(x,-.35,-3.7);plane.add(prop);
   for(let k=0;k<4;k++){const blade=new THREE.Mesh(new THREE.BoxGeometry(.16,3.5,.36),dark);blade.position.y=1.85;blade.rotation.z=k*Math.PI/2;prop.add(blade)}
   prop.userData.spin=true;this.planeProps??=[];this.planeProps.push(prop);
   add(new THREE.BoxGeometry(1.3,1.1,4.5),dark,x,-1.2,2.8);
  }
  add(new THREE.BoxGeometry(1.1,.55,5),skin,side*20,-.05,-1);
 }
 // Tall tailplane, fin, rear cargo ramp seam and markings.
 add(new THREE.BoxGeometry(19,.32,3.6),skin,0,2.3,10.2);
 add(new THREE.BoxGeometry(5,.22,2.5),dark,0,2.55,10.2);
 add(new THREE.BoxGeometry(.38,6,4.2),skin,0,4.1,11);
 add(new THREE.BoxGeometry(.08,1.2,8),dark,0,0,11.9);
 // Landing gear and wheels.
 for(const x of [-3.5,3.5])for(const z of [2.5]){
  add(new THREE.CylinderGeometry(.14,.14,2.5,8),dark,x,-2,z);
  for(const side of [-1,1])add(new THREE.CylinderGeometry(.55,.55,.32,12),rubber,x+side*.35,-3.1,z,0,0,Math.PI/2);
 }
 // Cabin portholes and side service panels.
 for(let z=-7;z<9;z+=2.5)for(const side of [-1,1])add(new THREE.SphereGeometry(.13,8,6),glass,side*1.79,.28,z);
 this.transportPlane=plane;plane.scale.setScalar(1.0);this.scene.add(plane);
 this.planeFlight=0;this.planeProps??=[];
 this.transportPlane.visible=false;
}
updateTransportPlane(dt){
 this.planeFlight+=dt;
 // A visible cargo aircraft crosses the island at high altitude once per match.
 const cycle=150,phase=(this.planeFlight%cycle)/cycle;
 this.transportPlane.visible=phase<.62;
 if(this.transportPlane.visible){
  const t=phase/.62;this.transportPlane.position.set(-250+t*500,57+Math.sin(t*Math.PI)*2,-62);
  this.transportPlane.rotation.set(0,Math.PI/2,0);
  for(const p of this.planeProps||[])p.rotation.z+=dt*24;
 }
}

updateZone(r){this.zoneRing.scale.setScalar(r/178)}
lineBlocked(ax,az,bx,bz){return this.obstacles.some(o=>segmentDistanceSq(o.x,o.z,ax,az,bx,bz)<(o.r*(o.kind==="buildingWall"?.9:o.kind==="building"?.85:.8))**2)}
move(p,dx,dz,r=.55){let x=clamp(p.x+dx,-209,209),z=clamp(p.z+dz,-209,209);for(const o of this.obstacles){const ox=x-o.x,oz=z-o.z,d=Math.hypot(ox,oz),rr=o.r+r;if(d<rr){x+=ox/(d||.001)*(rr-d);z+=oz/(d||.001)*(rr-d)}}p.x=clamp(x,-209,209);p.z=clamp(z,-209,209)}
resize(){this.camera.aspect=innerWidth/innerHeight;this.camera.updateProjectionMatrix();this.renderer.setSize(innerWidth,innerHeight,false);this.renderer.setPixelRatio(this.quality?.pixelRatio||Math.min(devicePixelRatio,1))}
render(){this.renderer.render(this.scene,this.camera)}
}