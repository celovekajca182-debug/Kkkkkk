import * as THREE from "three";
import {WEAPONS} from "../core/config.js";
import {segmentDistanceSq} from "../core/math.js";

export class Combat{
 constructor(world,audio,onDamage){
  this.world=world;this.audio=audio;this.onDamage=onDamage;this.bullets=[];this.effects=[];
  this.geo=new THREE.CylinderGeometry(.012,.012,.8,5);
  this.mat=new THREE.MeshBasicMaterial({color:0xffdfa0});
  this.flashGeo=new THREE.ConeGeometry(.12,.48,7);
  this.flashMat=new THREE.MeshBasicMaterial({color:0xffbd59,transparent:true,opacity:.95,depthWrite:false});
  this.particleGeo=new THREE.TetrahedronGeometry(.055);
  this.particleMats=[0xffd18a,0xff9b52,0xc6b8a0].map(color=>new THREE.MeshBasicMaterial({color,transparent:true,opacity:.95}));
  this.playerX=0;this.playerZ=0;this.maxBullets=72;this.maxEffects=36;
 }
 effect(mesh,life,update){
  if(this.effects.length>=this.maxEffects){const old=this.effects.shift();this.world.scene.remove(old.mesh)}
  this.world.scene.add(mesh);this.effects.push({mesh,life,max:life,update});
 }
 muzzle(x,y,z,angle){
  const flash=new THREE.Mesh(this.flashGeo,this.flashMat);flash.position.set(x+Math.sin(angle)*.48,y,z-Math.cos(angle)*.48);flash.rotation.set(Math.PI/2,0,-angle);flash.scale.set(.8,.8,1);
  this.effect(flash,.055,e=>{e.mesh.scale.setScalar(.5+e.life/e.max);e.mesh.material.opacity=e.life/e.max});
 }
 impact(x,z){
  const group=new THREE.Group();
  for(let i=0;i<3;i++){const m=new THREE.Mesh(this.particleGeo,this.particleMats[i]);m.position.set((Math.random()-.5)*.18,.9+Math.random()*.35,(Math.random()-.5)*.18);group.add(m)}
  group.position.set(x,0,z);this.effect(group,.18,e=>{const k=1-e.life/e.max;e.mesh.children.forEach(m=>{m.position.y+=.012;m.scale.setScalar(1-k)})});
 }
 fire(shooter,angle,name,spreadScale=1){
  const w=WEAPONS[name]||WEAPONS.M416,a=angle+(Math.random()-.5)*w.spread*spreadScale*2;
  const x=shooter.position.x+Math.sin(angle)*.5,z=shooter.position.z-Math.cos(angle)*.5;
  if(this.bullets.length>=this.maxBullets){const old=this.bullets.shift();this.world.scene.remove(old.mesh)}
  const mesh=new THREE.Mesh(this.geo,this.mat);mesh.position.set(x,1.45,z);mesh.scale.set(.8,.8,2.4);mesh.rotation.set(Math.PI/2,Math.PI-a,0);this.world.scene.add(mesh);
  this.bullets.push({mesh,x,y:1.45,z,vx:Math.sin(a)*w.speed,vy:0,vz:-Math.cos(a)*w.speed,owner:shooter.id,damage:w.damage,life:1.2});
  this.muzzle(x,1.45,z,angle);
  const d=shooter.id==="player"?0:Math.hypot(shooter.position.x-this.playerX,shooter.position.z-this.playerZ);
  this.audio.gunshot(name,shooter.id==="player"?1:Math.max(.1,1-d/85));
 }
 setPlayerPosition(x,z){this.playerX=x;this.playerZ=z}
 update(dt,actors){
  for(let i=this.effects.length-1;i>=0;i--){const e=this.effects[i];e.life-=dt;if(e.update)e.update(e);if(e.life<=0){this.world.scene.remove(e.mesh);this.effects.splice(i,1)}}
  for(let i=this.bullets.length-1;i>=0;i--){
   const b=this.bullets[i],ax=b.x,az=b.z,bx=ax+b.vx*dt,bz=az+b.vz*dt;b.life-=dt;
   let hit=null,best=Infinity;
   for(const a of actors){if(!a.alive||a.id===b.owner)continue;const d=segmentDistanceSq(a.position.x,a.position.z,ax,az,bx,bz);if(d<.58*.58){const score=Math.hypot(a.position.x-(ax+bx)/2,a.position.z-(az+bz)/2);if(score<best){best=score;hit=a}}}
   if(hit&&!this.world.lineBlocked(ax,az,bx,bz)){this.impact(bx,bz);this.onDamage(hit,b.damage,b.owner);b.life=0}
   else if(this.world.lineBlocked(ax,az,bx,bz)){this.impact(bx,bz);b.life=0}
   b.x=bx;b.z=bz;b.mesh.position.set(bx,1.45,bz);
   if(b.life<=0){this.world.scene.remove(b.mesh);this.bullets.splice(i,1)}
  }
 }
 clear(){for(const b of this.bullets)this.world.scene.remove(b.mesh);for(const e of this.effects)this.world.scene.remove(e.mesh);this.bullets.length=0;this.effects.length=0}
}
