import * as THREE from "three";
import {Actor} from "./actor.js";
import {BOT_COUNT,BOT_NAMES} from "../core/config.js";
import {distance2,normalize2,randomRange,seededRandom} from "../core/math.js";
export class BotSystem{
 constructor(world,combat,audio,onDeath){this.world=world;this.combat=combat;this.audio=audio;this.onDeath=onDeath;this.rng=seededRandom(77381);this.bots=[];for(let i=0;i<BOT_COUNT;i++){const b=new Actor("bot"+i,BOT_NAMES[i], [0x596b52,0x7b5545,0x536475,0x77715a][i%4],world.scene);b.weapon=["M416","AKM","SCARL","KAR98K"][i%4];b.cool=randomRange(this.rng,0,1);b.state="ROAM";b.waypoint=this.randomPoint();b.think=0;this.bots.push(b);this.place(b)}} 
 randomPoint(){return{x:randomRange(this.rng,-155,155),z:randomRange(this.rng,-155,155)}}
 place(b){let p=this.randomPoint(),tries=0;while(this.world.obstacles.some(o=>Math.hypot(o.x-p.x,o.z-p.z)<o.r+3)&&tries++<50)p=this.randomPoint();b.setPosition(p.x,p.z)}
 update(dt,targets,zone){
  for(const b of this.bots){if(!b.alive)continue;b.cool-=dt;b.think-=dt;let target=null,near=Infinity;
   if(b.think<=0){b.think=randomRange(this.rng,.18,.38);for(const t of targets){if(!t.alive||t.id===b.id)continue;const d=distance2(b.position.x,b.position.z,t.position.x,t.position.z);if(d<near&&d<65&&!this.world.lineBlocked(b.position.x,b.position.z,t.position.x,t.position.z)){near=d;target=t}}b.target=target?.id||null}
   target=targets.find(t=>t.id===b.target&&t.alive);let dir={x:0,z:0},speed=3.4;
   if(target){const dx=target.position.x-b.position.x,dz=target.position.z-b.position.z,d=Math.hypot(dx,dz)||1;dir=normalize2(dx,dz);b.yaw=Math.atan2(dx,-dz);if(d<24){dir={x:-dir.x*.55,z:-dir.z*.55};speed=2.5}else if(d<38){dir={x:dir.x*.2+Math.sin(b.id.length+b.anim)*.3,z:dir.z*.2+Math.cos(b.anim)*.3};speed=1.6}else speed=3.7;if(b.cool<=0&&d<55){this.combat.fire(b,b.yaw,b.weapon,2.3);b.cool=60/({M416:720,AKM:600,SCARL:650,KAR98K:42}[b.weapon]) * (b.weapon==="KAR98K"?1:1);}}
   else{const d=distance2(b.position.x,b.position.z,b.waypoint.x,b.waypoint.z);if(d<4)b.waypoint=this.randomPoint();dir=normalize2(b.waypoint.x-b.position.x,b.waypoint.z-b.position.z);b.yaw=Math.atan2(dir.x,-dir.z)}
   const p={...b.position};this.world.move(p,dir.x*speed*dt,dir.z*speed*dt,.55);b.setPosition(p.x,p.z);b.mesh.rotation.y=b.yaw;b.animate(dt,speed,!!target);
   if(Math.hypot(b.position.x,b.position.z)>zone.radius){b.health-=7.5*dt;if(b.health<=0)this.kill(b,"zone")}
  }
 }
 kill(b,source){if(!b.alive)return;b.alive=false;b.health=0;b.mesh.visible=false;this.onDeath(b,source)}
 alive(){return this.bots.filter(b=>b.alive)}
}