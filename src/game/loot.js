import * as THREE from "three";
import {randomRange,seededRandom} from "../core/math.js";

const WEAPON_COLORS={M416:0x46554b,SCARL:0x52604d,AKM:0x75523c,KAR98K:0x8b6847};
const TYPES=[
 ["ammo556",14,30],["ammo762",10,22],["bandage",8,14],["medkit",5,7],
 ["energy",7,12],["helmet",5,7],["vest",5,7],["backpack",4,6]
];
function weaponMesh(type){
 const g=new THREE.Group(),mat=new THREE.MeshStandardMaterial({color:WEAPON_COLORS[type],roughness:.82,metalness:.18}),dark=new THREE.MeshStandardMaterial({color:0x252a26,roughness:.7});
 const body=new THREE.Mesh(new THREE.BoxGeometry(.9,.13,.22),mat),stock=new THREE.Mesh(new THREE.BoxGeometry(.24,.18,.28),dark),mag=new THREE.Mesh(new THREE.BoxGeometry(.16,.28,.14),dark);
 stock.position.x=-.48;mag.position.set(.12,-.17,0);g.add(body,stock,mag);return g;
}
function itemMesh(type){
 const colors={ammo556:0xc5a34b,ammo762:0x927348,bandage:0xd9d7c7,medkit:0xb94b3e,energy:0x4b9caa,helmet:0x65745a,vest:0x394b3b,backpack:0x766044};
 const g=new THREE.Group(),m=new THREE.MeshStandardMaterial({color:colors[type],roughness:.78,metalness:type.startsWith("ammo")?.45:.05});
 const shape=type==="helmet"?new THREE.SphereGeometry(.27,12,8,0,Math.PI*2,0,Math.PI*.58):new THREE.BoxGeometry(type==="backpack"?.38:.3,type==="vest"?.3:.22,.24);
 const mesh=new THREE.Mesh(shape,m);g.add(mesh);
 if(type==="medkit"){const white=new THREE.Mesh(new THREE.BoxGeometry(.07,.25,.025),new THREE.MeshStandardMaterial({color:0xf0e8d9}));const red=new THREE.Mesh(new THREE.BoxGeometry(.25,.07,.03),new THREE.MeshStandardMaterial({color:0xf0e8d9}));g.add(white,red)}
 return g;
}
export class LootSystem{
 constructor(world){
  this.world=world;this.items=[];this.rng=seededRandom(42917);
  for(let i=0;i<18;i++){const type=this.rng()<.36?"M416":this.rng()<.42?"SCARL":this.rng()<.43?"AKM":"KAR98K";this.spawn(type)}
  for(const [type,weight,min,max] of TYPES)for(let i=0;i<max;i++)if(i<min||this.rng()<.35)this.spawn(type);
 }
 spawn(type){
  let x=randomRange(this.rng,-175,175),z=randomRange(this.rng,-175,175),tries=0;
  while((Math.hypot(x,z)<22||this.world.obstacles.some(o=>Math.hypot(o.x-x,o.z-z)<o.r+3))&&tries++<50){x=randomRange(this.rng,-175,175);z=randomRange(this.rng,-175,175)}
  const mesh=type in WEAPON_COLORS?weaponMesh(type):itemMesh(type);
  mesh.position.set(x,type==="helmet"?.34:.25,z);mesh.rotation.y=randomRange(this.rng,0,Math.PI*2);this.world.scene.add(mesh);
  const amount=type==="ammo556"?30:type==="ammo762"?24:type==="bandage"?2:1;
  this.items.push({type,x,z,mesh,active:true,amount});
 }
 nearest(x,z,max=3.6){let best=null,dist=max;for(const item of this.items){if(!item.active)continue;const d=Math.hypot(item.x-x,item.z-z);if(d<dist){dist=d;best=item}}return best?{item:best,distance:dist}:null}
 take(item){if(!item?.active)return null;item.active=false;item.mesh.visible=false;return item}
 update(dt){for(const item of this.items)if(item.active){item.mesh.rotation.y+=dt*.35;item.mesh.position.y=(item.type==="helmet"?.34:.25)+Math.sin(performance.now()*.0015+item.x)*.045}}
}
