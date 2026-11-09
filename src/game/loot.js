import * as THREE from "three";
import {WEAPONS} from "../core/config.js";
import {randomRange,seededRandom} from "../core/math.js";

const TYPES=["AKM","SCARL","KAR98K","M416"];
export class LootSystem{
 constructor(world){
  this.world=world;this.items=[];this.rng=seededRandom(42917);
  for(let i=0;i<18;i++){
   let x=randomRange(this.rng,-175,175),z=randomRange(this.rng,-175,175),tries=0;
   while((Math.hypot(x,z)<22||world.obstacles.some(o=>Math.hypot(o.x-x,o.z-z)<o.r+3))&&tries++<40){x=randomRange(this.rng,-175,175);z=randomRange(this.rng,-175,175)}
   const type=TYPES[i%TYPES.length],group=new THREE.Group();
   const body=new THREE.Mesh(new THREE.BoxGeometry(.9,.13,.22),new THREE.MeshStandardMaterial({color:type==="KAR98K"?0x8b6847:type==="AKM"?0x75523c:0x46554b,roughness:.82,metalness:.18}));
   const stock=new THREE.Mesh(new THREE.BoxGeometry(.24,.18,.28),new THREE.MeshStandardMaterial({color:0x514536,roughness:1}));stock.position.x=-.48;
   const mag=new THREE.Mesh(new THREE.BoxGeometry(.16,.28,.14),new THREE.MeshStandardMaterial({color:0x252a26,roughness:.7}));mag.position.set(.12,-.17,0);
   group.add(body,stock,mag);group.position.set(x,.42,z);group.rotation.y=randomRange(this.rng,0,Math.PI*2);world.scene.add(group);
   this.items.push({type,x,z,mesh:group,active:true});
  }
 }
 nearest(x,z,max=3.6){
  let best=null,dist=max;
  for(const item of this.items){if(!item.active)continue;const d=Math.hypot(item.x-x,item.z-z);if(d<dist){dist=d;best=item}}
  return best?{item:best,distance:dist}:null;
 }
 take(item){
  if(!item?.active)return null;item.active=false;item.mesh.visible=false;return item.type;
 }
 update(dt){for(const item of this.items)if(item.active){item.mesh.rotation.y+=dt*.35;item.mesh.position.y=.42+Math.sin(performance.now()*.0015+item.x)*.045}}
}
