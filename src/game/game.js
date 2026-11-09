import * as THREE from "three";
import {World} from "../world/world.js";
import {Input} from "../core/input.js";
import {AudioEngine} from "../core/audio.js";
import {Actor} from "./actor.js";
import {FirstPersonWeapon} from "./viewmodel.js";
import {Combat} from "./combat.js";
import {BotSystem} from "./bots.js";
import {LootSystem} from "./loot.js";
import {PLAYER,WEAPONS,ZONE} from "../core/config.js";
import {clamp,normalize2} from "../core/math.js";
export class Game{
 constructor(canvas){this.canvas=canvas;this.world=new World(canvas);this.input=new Input(canvas);this.audio=new AudioEngine();this.player=new Actor("player","YOU",0x667e59,this.world.scene);this.player.setPosition(0,0);this.player.mesh.visible=false;this.viewModel=new FirstPersonWeapon(this.world.camera);this.player.weapon="M416";this.inventory={M416:{mag:30,reserve:120},AKM:{mag:30,reserve:90},SCARL:{mag:30,reserve:120},KAR98K:{mag:5,reserve:25}};this.weapons=["M416"];this.weaponIndex=0;this.armor={vest:0,helmet:0,backpack:0};this.consumables={bandage:5,medkit:1,energy:1};this.boost=0;this.healCooldown=0;this.kills=0;this.yaw=0;this.pitch=.22;this.recoil=0;this.cool=0;this.reload=0;this.reloadElapsed=0;this.fireTime=0;this.zone={radius:ZONE.start,elapsed:0};this.aim=false;this.running=false;this.started=false;this.over=false;this.phase="LOBBY";this.phaseTimer=0;this.flightDuration=8;this.dropPoint={x:0,z:0};this.last=0;this.stepTimer=0;this.feed=[];this.combat=new Combat(this.world,this.audio,(a,d,owner)=>this.damage(a,d,owner));this.bots=new BotSystem(this.world,this.combat,this.audio,(b,s)=>this.elimination(b,s));this.loot=new LootSystem(this.world);this.bots.setLoot(this.loot);this.combat.setPlayerPosition(0,0);this.bindUI();this.drawHUD();this.resize=()=>this.world.resize();addEventListener("resize",this.resize)}
 bindUI(){const login=document.querySelector("#login"),logout=document.querySelector("#logout"),nick=document.querySelector("#nickname");const saved=localStorage.getItem("lastzone.nickname");if(saved)nick.value=saved;login.addEventListener("click",()=>{const name=nick.value.trim().replace(/[^a-zA-Z0-9_-]/g,"").slice(0,16);if(name.length<3){nick.setCustomValidity("Никнейм должен содержать минимум 3 символа (латиница, цифры, _ или -)");nick.reportValidity();return}nick.setCustomValidity("");localStorage.setItem("lastzone.nickname",name);document.querySelector("#profileName").textContent=name;document.querySelector("#avatar").textContent=name[0].toUpperCase();const stats=JSON.parse(localStorage.getItem("lastzone.stats")||"{\"matches\":0,\"wins\":0,\"kills\":0}");document.querySelector("#profileStats").textContent="ЛОКАЛЬНЫЙ ПРОФИЛЬ · "+stats.matches+" МАТЧЕЙ · "+stats.wins+" ПОБЕД · "+stats.kills+" УБИЙСТВ";document.querySelector("#loginPanel").hidden=true;document.querySelector("#lobbyPanel").hidden=false});logout.addEventListener("click",()=>{document.querySelector("#lobbyPanel").hidden=true;document.querySelector("#loginPanel").hidden=false});document.querySelector("#play").addEventListener("click",()=>{this.audio.init();this.start()});document.querySelector("#sound").addEventListener("click",e=>{this.audio.setEnabled(!this.audio.enabled);e.currentTarget.textContent="ЗВУК: "+(this.audio.enabled?"ВКЛ":"ВЫКЛ")})}
 start(){if(this.started)return;this.started=true;this.over=false;this.phase="PREP";this.phaseTimer=10;document.querySelector("#menu").style.display="none";document.querySelector("#phaseBanner").hidden=false;this.player.name=localStorage.getItem("lastzone.nickname")||"PLAYER";this.player.setPosition(0,0);this.player.health=100;this.player.alive=true;this.last=performance.now();this.bots.bots.forEach((b,i)=>{const a=i/this.bots.bots.length*Math.PI*2,r=9+(i%4)*2;b.setPosition(Math.cos(a)*r,Math.sin(a)*r);b.mesh.visible=true;b.alive=true;b.health=100;b.state="PREP"});this.last=performance.now();requestAnimationFrame(t=>this.loop(t))}
 loop(now){if(!this.started)return;const dt=Math.min(.04,(now-this.last)/1000||0);this.last=now;if(!this.over)this.update(dt);this.render(dt);requestAnimationFrame(t=>this.loop(t))}
 update(dt){const input=this.input.consume(),p=this.player;
  const jump=document.querySelector("#jump"),chute=document.querySelector("#parachute");
  if(jump)jump.style.display=["PREP","FLIGHT"].includes(this.phase)?"block":"none";
  if(chute)chute.style.display=["FREEFALL","PARACHUTE"].includes(this.phase)?"block":"none";
  if(this.phase==="PREP"){
   this.aim=input.aim;this.yaw-=input.look.x*.003;this.pitch=clamp(this.pitch+input.look.y*.002,-.12,.75);
   const mv=normalize2(input.move.x,-input.move.y),forward={x:Math.sin(this.yaw),z:-Math.cos(this.yaw)},right={x:Math.cos(this.yaw),z:Math.sin(this.yaw)};
   const dx=(forward.x*mv.y+right.x*mv.x)*PLAYER.walk*dt,dz=(forward.z*mv.y+right.z*mv.x)*PLAYER.walk*dt;
   if(Math.hypot(input.move.x,input.move.y)>.08){const pos={...p.position};this.world.move(pos,dx,dz,PLAYER.radius);p.setPosition(pos.x,pos.z)}
   this.phaseTimer-=dt;this.updatePhaseUI("ОСТРОВ ОЖИДАНИЯ · НАЖМИ ПРЫЖОК ДЛЯ ПОСАДКИ В САМОЛЁТ",Math.max(0,Math.ceil(this.phaseTimer)));
   this.world.updateTransportPlane(dt);this.drawHUD();
   if(input.jump||this.phaseTimer<=0){this.phase="FLIGHT";this.phaseTimer=this.flightDuration;this.dropPoint={x:Math.random()*220-110,z:Math.random()*220-110};this.player.setPosition(-190,0);this.world.transportPlane.visible=true}
   return;
  }
  if(this.phase==="FLIGHT"){
   this.phaseTimer-=dt;const t=1-this.phaseTimer/this.flightDuration,plane=this.world.transportPlane;
   plane.visible=true;plane.position.set(-190+t*380,55,-62);plane.rotation.set(0,Math.PI/2,0);
   this.world.camera.position.set(plane.position.x,plane.position.y+3,plane.position.z+12);this.world.camera.lookAt(plane.position.x+18,plane.position.y-3,plane.position.z);
   this.updatePhaseUI("В САМОЛЁТЕ · НАЖМИ ПРЫЖОК ДЛЯ ВЫСАДКИ",Math.max(0,Math.ceil(this.phaseTimer)));
   if(input.jump||this.phaseTimer<=0){this.phase="FREEFALL";this.airHeight=55;this.airSpeed=0;this.player.setPosition(this.dropPoint.x,this.dropPoint.z)}
   return;
  }
  if(this.phase==="FREEFALL"||this.phase==="PARACHUTE"){
   this.yaw-=input.look.x*.003;this.pitch=clamp(this.pitch+input.look.y*.002,-.35,.65);
   const mv=normalize2(input.move.x,-input.move.y),forward={x:Math.sin(this.yaw),z:-Math.cos(this.yaw)},right={x:Math.cos(this.yaw),z:Math.sin(this.yaw)};
   const steer=this.phase==="PARACHUTE"?11:5.5,dx=(forward.x*mv.y+right.x*mv.x)*steer*dt,dz=(forward.z*mv.y+right.z*mv.x)*steer*dt;
   const pos={...p.position};this.world.move(pos,dx,dz,PLAYER.radius);p.setPosition(pos.x,pos.z);
   if(input.parachute&&this.airHeight>2)this.phase="PARACHUTE";
   this.airSpeed=this.phase==="PARACHUTE"?Math.min(5,this.airSpeed+dt*5):Math.min(24,this.airSpeed+dt*9);
   this.airHeight=Math.max(0,this.airHeight-this.airSpeed*dt);
   this.updatePhaseUI(this.phase==="PARACHUTE"?"ПАРАШЮТ ОТКРЫТ · УПРАВЛЯЙ НАПРАВЛЕНИЕМ":"СВОБОДНОЕ ПАДЕНИЕ · НАЖМИ ПАРАШЮТ",Math.ceil(this.airHeight)+" М");
   if(this.airHeight<=0){this.phase="PLAYING";this.world.transportPlane.visible=false;this.bots.bots.forEach((b,i)=>{const a=i/this.bots.bots.length*Math.PI*2,r=35+(i%5)*13;b.setPosition(clamp(this.dropPoint.x+Math.cos(a)*r,-185,185),clamp(this.dropPoint.z+Math.sin(a)*r,-185,185));b.state="ROAM";b.waypoint=this.bots.randomPoint()});document.querySelector("#phaseBanner").hidden=true;this.zone.elapsed=0}
   return;
  }
  this.aim=input.aim;this.yaw-=input.look.x*.003;this.pitch=clamp(this.pitch+input.look.y*.002,-.12,.75);this.cool=Math.max(0,this.cool-dt);this.recoil=Math.max(0,this.recoil-dt*.12);let shotFired=false;
  if(input.swap)this.switchWeapon();if(input.interact)this.pickupLoot();if(input.heal)this.useHealing();if(input.reload)this.beginReload();if(this.reload>0){this.reload-=dt;if(this.reload<=0){const w=this.inventory[p.weapon],need=WEAPONS[p.weapon].mag-w.mag,take=Math.min(need,w.reserve);w.mag+=take;w.reserve-=take;this.audio.reload()}}
  const move=normalize2(input.move.x,-input.move.y),moving=Math.hypot(input.move.x,input.move.y)>.08,forward={x:Math.sin(this.yaw),z:-Math.cos(this.yaw)},right={x:Math.cos(this.yaw),z:Math.sin(this.yaw)},mx=(forward.x*move.y+right.x*move.x),mz=(forward.z*move.y+right.z*move.x),speed=input.sprint&&!this.aim?PLAYER.sprint:PLAYER.walk;
  const pos={...p.position};if(moving)this.world.move(pos,mx*speed*dt,mz*speed*dt,PLAYER.radius);p.setPosition(pos.x,pos.z);if(moving)p.yaw=Math.atan2(mx,-mz);p.animate(dt,moving?speed:0,this.aim);if(moving){this.stepTimer-=dt;if(this.stepTimer<=0){this.audio.footstep(input.sprint);this.stepTimer=input.sprint?.31:.46}}else this.stepTimer=0;
  this.loot.update(dt);this.healCooldown=Math.max(0,this.healCooldown-dt);if(this.boost>0){this.boost=Math.max(0,this.boost-dt);p.health=Math.min(100,p.health+1.2*dt)}this.updatePickupPrompt();this.combat.setPlayerPosition(p.position.x,p.position.z);this.bots.update(dt,[p,...this.bots.bots],this.zone);
  if(input.fire&&this.cool<=0&&this.reload<=0&&p.alive){const ammo=this.inventory[p.weapon];if(ammo.mag>0){this.combat.fire(p,this.yaw+this.recoil,p.weapon,moving?2.4:this.aim?.72:1);shotFired=true;ammo.mag--;this.cool=60/WEAPONS[p.weapon].rpm;this.recoil=clamp(this.recoil+WEAPONS[p.weapon].recoil,0,.15)}else this.beginReload()}
  this.combat.update(dt,[p,...this.bots.bots]);this.zone.elapsed+=dt;if(this.zone.elapsed>ZONE.delay)this.zone.radius=Math.max(ZONE.min,this.zone.radius-ZONE.shrink*dt);this.world.updateZone(this.zone.radius);
  if(Math.hypot(p.position.x,p.position.z)>this.zone.radius){p.health-=ZONE.damage*dt;if(p.health<=0)this.end("Ты погиб за пределами безопасной зоны")}
  this.viewModel.update(dt,shotFired,this.aim,this.reload>0);this.drawHUD();if(this.bots.alive().length===0)this.end("ПОБЕДА");if(!p.alive)this.end("ПОРАЖЕНИЕ");
 }
 beginReload(){if(this.reload>0)return;const a=this.inventory[this.player.weapon],w=WEAPONS[this.player.weapon];if(a.mag<w.mag&&a.reserve>0){this.reload=w.reload;this.reloadElapsed=this.reload}}
 updatePhaseUI(label,count){const el=document.querySelector("#phaseBanner");if(el)el.textContent=label+" · "+count+" СЕК"}
 updatePickupPrompt(){const el=document.querySelector("#pickup"),found=this.loot.nearest(this.player.position.x,this.player.position.z);this.nearLoot=found?.item||null;el.style.display=found?"block":"none";if(found)el.textContent="E · "+this.lootLabel(found.item)}
 lootLabel(item){const names={M416:"M416",AKM:"AKM",SCARL:"SCAR-L",KAR98K:"Kar98k",ammo556:"5.56 мм ×30",ammo762:"7.62 мм ×24",bandage:"Бинт ×2",medkit:"Аптечка",energy:"Энергетик",helmet:"Шлем ур. 1",vest:"Бронежилет ур. 1",backpack:"Рюкзак"};return names[item.type]||item.type}
 pickupLoot(){const item=this.nearLoot;if(!item?.active)return;const picked=this.loot.take(item);if(!picked)return;const type=picked.type;
  if(WEAPONS[type]){const inv=this.inventory[type];if(!inv)this.inventory[type]={mag:WEAPONS[type].mag,reserve:WEAPONS[type].reserve};else inv.reserve+=Math.floor(WEAPONS[type].mag*1.5);if(!this.weapons.includes(type)){if(this.weapons.length>=2)this.weapons[this.weaponIndex]=type;else this.weapons.push(type)}this.weaponIndex=this.weapons.indexOf(type);this.player.weapon=type;this.reload=0}
  else if(type==="ammo556"){for(const w of ["M416","SCARL"])this.inventory[w].reserve+=picked.amount}
  else if(type==="ammo762")this.inventory.AKM.reserve+=picked.amount
  else if(type==="bandage")this.consumables.bandage+=picked.amount
  else if(type==="medkit")this.consumables.medkit+=1
  else if(type==="energy")this.consumables.energy+=1
  else if(type==="helmet")this.armor.helmet=Math.min(1,this.armor.helmet+1)
  else if(type==="vest")this.armor.vest=Math.min(1,this.armor.vest+1)
  else if(type==="backpack")this.armor.backpack=Math.min(1,this.armor.backpack+1);
  this.feed.unshift("Подобрано: "+this.lootLabel(picked));this.feed=this.feed.slice(0,5);this.nearLoot=null;this.updatePickupPrompt()}
 useHealing(){if(this.healCooldown>0||this.player.health>=100)return;if(this.consumables.medkit>0){this.consumables.medkit--;this.player.health=Math.min(100,this.player.health+75);this.healCooldown=3;this.feed.unshift("Использована аптечка")}else if(this.consumables.bandage>0){this.consumables.bandage--;this.player.health=Math.min(100,this.player.health+15);this.healCooldown=1.5;this.feed.unshift("Использован бинт")}else if(this.consumables.energy>0){this.consumables.energy--;this.boost=30;this.feed.unshift("Энергетик: восстановление здоровья")}this.feed=this.feed.slice(0,5)}

 switchWeapon(){if(this.weapons.length<2)return;this.weaponIndex=(this.weaponIndex+1)%this.weapons.length;this.player.weapon=this.weapons[this.weaponIndex];this.reload=0}
 damage(target,amount,owner){if(!target.alive)return;const mitigation=target.id==="player"?(this.armor.vest?.22:0)+(this.armor.helmet?.08:0):(target.armorValue||0);target.health-=amount*(1-mitigation);this.audio.hit();if(target.health<=0){if(target.id==="player"){target.alive=false;target.health=0}else{const b=this.bots.bots.find(x=>x.id===target.id);if(b)this.bots.kill(b,owner)}if(owner==="player")this.kills++;}}
 elimination(b,source){this.feed.unshift(b.name+" выбыл");this.feed=this.feed.slice(0,5)}
 end(message){if(this.over)return;this.over=true;const won=message==="ПОБЕДА";const stats=JSON.parse(localStorage.getItem("lastzone.stats")||"{\"matches\":0,\"wins\":0,\"kills\":0}");stats.matches++;stats.wins+=won?1:0;stats.kills+=this.kills;localStorage.setItem("lastzone.stats",JSON.stringify(stats));document.querySelector("#menu").style.display="grid";document.querySelector("#loginPanel").hidden=true;document.querySelector("#lobbyPanel").hidden=false;document.querySelector("#logo").innerHTML="LAST ZONE<span>"+message+" · "+this.kills+" УБИЙСТВ</span>";document.querySelector("#profileStats").textContent="ЛОКАЛЬНЫЙ ПРОФИЛЬ · "+stats.matches+" МАТЧЕЙ · "+stats.wins+" ПОБЕД · "+stats.kills+" УБИЙСТВ";document.querySelector("#play").textContent="НОВЫЙ МАТЧ";document.querySelector("#play").onclick=()=>location.reload()}
 drawHUD(){const p=this.player,a=this.inventory[p.weapon];document.querySelector("#alive").textContent=1+this.bots.alive().length;document.querySelector("#health").textContent=Math.max(0,Math.ceil(p.health));document.querySelector("#kills").textContent=this.kills;document.querySelector("#zone").textContent=Math.round(this.zone.radius/ZONE.start*100)+"%";document.querySelector("#weaponName").textContent=p.weapon;document.querySelector("#ammo").textContent=a.mag+" / "+a.reserve+(this.reload>0?" · ПЕРЕЗАРЯДКА":"");const kit=document.querySelector("#kit");if(kit)kit.textContent="АПТЕЧКА "+this.consumables.medkit+" · БИНТ "+this.consumables.bandage+" · ЭНЕРГЕТИК "+this.consumables.energy;const gear=document.querySelector("#gear");if(gear)gear.textContent="ШЛЕМ "+(this.armor.helmet?"1":"—")+" · БРОНЯ "+(this.armor.vest?"1":"—")+" · РЮКЗАК "+(this.armor.backpack?"1":"—");document.querySelector("#feed").textContent=this.feed.join("\n");const c=document.querySelector("#minimap"),ctx=c.getContext("2d");ctx.clearRect(0,0,256,256);ctx.fillStyle="#293e30";ctx.fillRect(0,0,256,256);ctx.strokeStyle="#9b9b70";ctx.lineWidth=2;ctx.beginPath();ctx.arc(128,128,this.zone.radius/420*256,0,Math.PI*2);ctx.stroke();for(const b of this.bots.alive()){const x=128+b.position.x/420*256,z=128+b.position.z/420*256;ctx.fillStyle="#df6d55";ctx.fillRect(x-2,z-2,4,4)}ctx.fillStyle="#f1d17b";ctx.beginPath();ctx.arc(128+p.position.x/420*256,128+p.position.z/420*256,4,0,Math.PI*2);ctx.fill()}
 render(dt){const p=this.player,aim=this.aim,camera=this.world.camera;if(this.phase==="FREEFALL"||this.phase==="PARACHUTE"){camera.position.set(p.position.x,Math.max(2,this.airHeight)+3,p.position.z+8);camera.lookAt(p.position.x-Math.sin(this.yaw)*18,Math.max(0,this.airHeight)-5,p.position.z-Math.cos(this.yaw)*18)}else if(this.phase!=="FLIGHT"){camera.position.set(p.position.x,1.66,p.position.z);camera.rotation.order="YXZ";const d=24;camera.lookAt(p.position.x-Math.sin(this.yaw)*d,1.66-Math.tan(this.pitch)*d,p.position.z-Math.cos(this.yaw)*d);const fov=aim?48:78;camera.fov+=(fov-camera.fov)*(1-Math.exp(-dt*12));camera.updateProjectionMatrix()}this.world.render()}
}