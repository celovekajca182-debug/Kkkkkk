import * as THREE from "three";
export class Actor{
 constructor(id,name,color,scene){this.id=id;this.name=name;this.position={x:0,z:0};this.health=100;this.alive=true;this.yaw=0;this.anim=0;this.mesh=this.build(color);scene.add(this.mesh)}
 build(color){
  const root=new THREE.Group(),cloth=new THREE.MeshStandardMaterial({color,roughness:.9}),armor=new THREE.MeshStandardMaterial({color:0x303a35,roughness:.8}),dark=new THREE.MeshStandardMaterial({color:0x171d1a}),skin=new THREE.MeshStandardMaterial({color:0xb58b70}),helmet=new THREE.MeshStandardMaterial({color:0x475346});
  const add=(geo,mat,x,y,z,parent=root)=>{const m=new THREE.Mesh(geo,mat);m.position.set(x,y,z);m.castShadow=true;parent.add(m);return m};
  add(new THREE.CapsuleGeometry(.34,.72,5,10),cloth,0,1.35,0);add(new THREE.BoxGeometry(.66,.56,.25),armor,0,1.4,-.12);
  add(new THREE.SphereGeometry(.23,14,12),skin,0,2.03,0);add(new THREE.SphereGeometry(.28,14,9),helmet,0,2.2,0);
  const legs=[];for(const x of [-.19,.19]){const p=new THREE.Group();p.position.set(x,.88,0);root.add(p);add(new THREE.CapsuleGeometry(.14,.48,4,8),cloth,0,-.28,0,p);add(new THREE.BoxGeometry(.28,.17,.4),dark,0,-.64,-.08,p);legs.push(p)}
  const arms=[];for(const x of [-.39,.39]){const p=new THREE.Group();p.position.set(x,1.64,0);root.add(p);add(new THREE.CapsuleGeometry(.13,.4,4,8),cloth,0,-.23,-.08,p);arms.push(p)}
  const gun=new THREE.Group();add(new THREE.BoxGeometry(.15,.16,.78),dark,.25,1.45,-.57,gun);add(new THREE.CylinderGeometry(.025,.035,.72,8),armor,.25,1.47,-1.28,gun).rotation.x=Math.PI/2;root.add(gun);
  root.userData={legs,arms,gun};return root
 }
 setPosition(x,z){this.position.x=x;this.position.z=z;this.mesh.position.set(x,0,z)}
 animate(dt,speed,aim=false){this.anim+=dt*(speed>7?12:8);const swing=Math.sin(this.anim)*Math.min(1,speed/7)*.55,p=this.mesh.userData;p.legs[0].rotation.x=swing;p.legs[1].rotation.x=-swing;p.arms[0].rotation.x=aim?-.7:-swing*.25;p.arms[1].rotation.x=aim?-.65:-swing*.25}
}