import * as THREE from "three";

// First-person weapon viewmodel. Geometry is deliberately shared within this
// instance; replaceable by authored GLB weapon assets without changing Game.
export class FirstPersonWeapon {
 constructor(camera){
  this.root=new THREE.Group();camera.add(this.root);
  this.root.position.set(.31,-.29,-.58);this.root.rotation.set(-.025,-.035,0);
  this.kick=0;this.aimBlend=0;this.reloadTime=0;
  const metal=new THREE.MeshStandardMaterial({color:0x242b2b,metalness:.78,roughness:.32});
  const polymer=new THREE.MeshStandardMaterial({color:0x202722,metalness:.08,roughness:.76});
  const grip=new THREE.MeshStandardMaterial({color:0x343b32,roughness:.92});
  const sight=new THREE.MeshStandardMaterial({color:0x111b1a,metalness:.35,roughness:.2});
  const add=(geo,mat,pos,rot)=>{const m=new THREE.Mesh(geo,mat);m.position.set(...pos);if(rot)m.rotation.set(...rot);m.castShadow=false;m.receiveShadow=false;this.root.add(m);return m};
  const box=(mat,pos,size,rot)=>add(new THREE.BoxGeometry(...size),mat,pos,rot);
  // M416-style receiver, handguard, barrel and muzzle.
  box(polymer,[0,0,0],[.19,.16,.34]);
  box(metal,[0,.015,-.22],[.16,.13,.22]);
  box(polymer,[0,.012,-.43],[.15,.12,.27]);
  add(new THREE.CylinderGeometry(.025,.025,.48,12),metal,[0,.015,-.77],[Math.PI/2,0,0]);
  add(new THREE.CylinderGeometry(.042,.032,.09,12),metal,[0,.015,-1.035],[Math.PI/2,0,0]);
  box(grip,[0,-.17,.015],[.105,.25,.13],[.12,0,0]);
  box(polymer,[0,-.2,-.18],[.105,.23,.12],[.12,0,0]);
  box(polymer,[0,.02,.24],[.13,.12,.3]);
  box(metal,[0,.115,-.08],[.1,.055,.16]);
  // Picatinny rail, optic body and glass.
  box(metal,[0,.105,-.28],[.13,.035,.35]);
  box(polymer,[0,.205,-.22],[.105,.16,.18]);
  box(sight,[0,.205,-.125],[.075,.105,.012]);
  box(metal,[0,.13,-.62],[.035,.045,.1]);
  // Magazine with floor plate and visible ribs.
  box(polymer,[0,-.16,-.12],[.115,.29,.16],[.16,0,0]);
  box(metal,[0,-.31,-.12],[.12,.025,.17]);
  for(let i=0;i<3;i++)box(grip,[0,-.17+i*.065,-.205],[.105,.012,.012]);
  this.hands=new THREE.Group();this.root.add(this.hands);
  const glove=new THREE.MeshStandardMaterial({color:0x3c4438,roughness:1});
  box(glove,[-.19,-.13,-.38],[.16,.13,.2],[.15,0,-.2]);
  box(glove,[.13,-.13,.015],[.13,.12,.17]);
 }
 update(dt,firing,aim,reloading){
  this.kick=Math.max(0,this.kick-dt*3.8);if(firing)this.kick=Math.min(.09,this.kick+dt*1.4);
  const target=aim?1:0;this.aimBlend+=(target-this.aimBlend)*(1-Math.exp(-dt*13));
  const a=this.aimBlend;
  this.root.position.set(.31*(1-a),-.29*(1-a)-.04*a,-.58+.28*a-this.kick);
  this.root.rotation.x=-.025-this.kick*.7;
  this.root.rotation.y=-.035*(1-a);
  this.root.visible=true;
  if(reloading)this.reloadTime+=dt;else this.reloadTime=0;
  if(reloading){this.root.rotation.x=-.12+Math.sin(this.reloadTime*5)*.035;this.root.position.y-=Math.min(.12,this.reloadTime*.05)}
 }
}
