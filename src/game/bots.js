import * as THREE from "three";
import {Actor} from "./actor.js";
import {BOT_COUNT,BOT_NAMES,WEAPONS} from "../core/config.js";
import {distance2,normalize2,randomRange,seededRandom} from "../core/math.js";

const ROLES=[
 {id:"RUSHER",label:"Штурмовик",preferred:[9,25],speed:4.35,aggression:1.25,accuracy:.72,strafe:0.12},
 {id:"TACTICAL",label:"Тактик",preferred:[20,42],speed:3.55,aggression:1,accuracy:.84,strafe:.48},
 {id:"SCOUT",label:"Разведчик",preferred:[28,52],speed:4.1,aggression:.82,accuracy:.68,strafe:.72},
 {id:"SNIPER",label:"Снайпер",preferred:[42,78],speed:2.9,aggression:.9,accuracy:.94,strafe:.18},
 {id:"SURVIVOR",label:"Осторожный",preferred:[24,48],speed:3.35,aggression:.62,accuracy:.76,strafe:.3}
];
const WEAPON_VALUE={M416:3,AKM:3,SCARL:3,KAR98K:4};

export class BotSystem{
 constructor(world,combat,audio,onDeath){
  this.world=world;this.combat=combat;this.audio=audio;this.onDeath=onDeath;
  this.rng=seededRandom(77381);this.bots=[];this.loot=null;
  for(let i=0;i<BOT_COUNT;i++){
   const role=ROLES[i%ROLES.length],b=new Actor("bot"+i,BOT_NAMES[i],[0x596b52,0x7b5545,0x536475,0x77715a][i%4],world.scene);
   b.role=role;b.weapon=null;
   b.cool=randomRange(this.rng,0,1);b.state="ROAM";b.waypoint=this.randomPoint();
   b.think=randomRange(this.rng,0,.5);b.target=null;b.strafeSign=i%2?1:-1;
   b.personality={skill:randomRange(this.rng,.72,1.18),courage:randomRange(this.rng,.72,1.28)};
   this.bots.push(b);this.place(b);
  }
 }
 setLoot(loot){this.loot=loot}
 randomPoint(){return{x:randomRange(this.rng,-155,155),z:randomRange(this.rng,-155,155)}}
 place(b){let p=this.randomPoint(),tries=0;while(this.world.obstacles.some(o=>Math.hypot(o.x-p.x,o.z-p.z)<o.r+3)&&tries++<50)p=this.randomPoint();b.setPosition(p.x,p.z)}
 findTarget(b,targets){
  let best=null,score=Infinity;
  for(const t of targets){
   if(!t.alive||t.id===b.id)continue;
   const d=distance2(b.position.x,b.position.z,t.position.x,t.position.z);
   const max=b.role.id==="SNIPER"?82:68;
   if(d>max||this.world.lineBlocked(b.position.x,b.position.z,t.position.x,t.position.z))continue;
   const threat=t.id=== "player"?1:1.12;
   const value=d*threat;
   if(value<score){score=value;best=t}
  }
  return best;
 }
 collectLoot(b){
  if(!this.loot)return;
  const found=this.loot.nearest(b.position.x,b.position.z,2.1);
  if(!found)return;
  const type=found.item.type,current=WEAPON_VALUE[b.weapon]||0,next=WEAPON_VALUE[type]||0;
  if(WEAPON_VALUE[type]&&( !b.weapon||next>current||(next===current&&type!==b.weapon&&this.rng()>.55))){
   this.loot.take(found.item);b.weapon=type;
  }
 }
 update(dt,targets,zone){
  for(const b of this.bots){
   if(!b.alive)continue;
   b.cool-=dt;b.think-=dt;
   if(b.think<=0){b.think=randomRange(this.rng,.16,.34);b.target=this.findTarget(b,targets)?.id||null}
   const target=targets.find(t=>t.id===b.target&&t.alive)||null;
   let dir={x:0,z:0},speed=b.role.speed;
   if(target){
    const dx=target.position.x-b.position.x,dz=target.position.z-b.position.z,d=Math.hypot(dx,dz)||1;
    const toward=normalize2(dx,dz),role=b.role;
    b.yaw=Math.atan2(dx,-dz);b.state="ENGAGE";
    const min=role.preferred[0],max=role.preferred[1];
    if(d<min){
     const retreat=role.id==="RUSHER"?.18:1;
     dir={x:-toward.x*retreat,z:-toward.z*retreat};speed=role.id==="RUSHER"?2.8:role.speed;
    }else if(d>max){
     dir=toward;speed=role.id==="SNIPER"?role.speed*.8:role.speed;
    }else{
     const side={x:-toward.z*b.strafeSign,z:toward.x*b.strafeSign};
     const forward=(d-(min+max)*.5)*.035;
     dir=normalize2(toward.x*forward+side.x*role.strafe,toward.z*forward+side.z*role.strafe);
     speed=role.id==="SNIPER"?.65:role.speed*.58;
     if(this.rng()<.012)b.strafeSign*=-1;
    }
    const weapon=b.weapon?WEAPONS[b.weapon]:null;
    const idealRange=b.weapon==="KAR98K"?48:31;
    const inRange=!!weapon&&d<(b.weapon==="KAR98K"?82:58);
    const courage=b.personality.courage*(role.aggression);
    if(weapon&&b.cool<=0&&inRange&&d>5&&!(d<min*.55&&role.id!=="RUSHER")){
     const error=(1-role.accuracy)+(1-b.personality.skill)*.55;
     const aimError=(this.rng()-.5)*error*(d/idealRange)*.14;
     this.combat.fire(b,b.yaw+aimError,b.weapon,Math.max(.45,1+(d-idealRange)/90));
     b.cool=60/weapon.rpm*randomRange(this.rng,.9,1.22)/Math.max(.75,courage);
    }
   }else{
    b.state="ROAM";
    const d=distance2(b.position.x,b.position.z,b.waypoint.x,b.waypoint.z);
    if(d<5){b.waypoint=this.randomPoint();b.think=0}
    dir=normalize2(b.waypoint.x-b.position.x,b.waypoint.z-b.position.z);
    b.yaw=Math.atan2(dir.x,-dir.z);
   }
   const p={...b.position};
   this.world.move(p,dir.x*speed*dt,dir.z*speed*dt,.55);
   b.setPosition(p.x,p.z);b.mesh.rotation.y=b.yaw;b.animate(dt,speed,!!target);
   this.collectLoot(b);
   if(Math.hypot(b.position.x,b.position.z)>zone.radius){b.health-=7.5*dt;if(b.health<=0)this.kill(b,"zone")}
  }
 }
 kill(b,source){if(!b.alive)return;b.alive=false;b.health=0;b.mesh.visible=false;this.onDeath(b,source)}
 alive(){return this.bots.filter(b=>b.alive)}
}
