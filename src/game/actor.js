import * as THREE from "three";

const mat=(color,roughness=.82,metalness=0)=>new THREE.MeshStandardMaterial({color,roughness,metalness});
const shared={
 fabric:mat(0x59634d),fabricDark:mat(0x343d32),armor:mat(0x303a35,.72,.12),
 edge:mat(0x202823,.7,.18),rubber:mat(0x171b19),skin:mat(0x98745e),
 helmet:mat(0x424d3e,.68,.08),glass:mat(0x182b2c,.22,.55),
 metal:mat(0x252c2b,.34,.72),wood:mat(0x574333,.78,.08),
 pouch:mat(0x424a39),sole:mat(0x111513)
};

export class Actor{
 constructor(id,name,color,scene){
  this.id=id;this.name=name;this.position={x:0,z:0};this.health=100;this.alive=true;this.yaw=0;this.anim=0;
  this.mesh=this.build(color);scene.add(this.mesh);
 }
 build(color){
  const root=new THREE.Group(),legs=[],arms=[];
  const uniform=mat(color,.9),vest=shared.armor;
  const add=(parent,geometry,material,x,y,z,options={})=>{
   const mesh=new THREE.Mesh(geometry,material);mesh.position.set(x,y,z);
   if(options.rotation)mesh.rotation.set(...options.rotation);
   if(options.scale)mesh.scale.set(...options.scale);
   mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;
  };
  const box=(parent,material,x,y,z,sx,sy,sz,rx=0,ry=0,rz=0)=>{
   return add(parent,new THREE.BoxGeometry(sx,sy,sz),material,x,y,z,{rotation:[rx,ry,rz]});
  };
  const capsule=(parent,material,x,y,z,r,len,rx=0)=>{
   return add(parent,new THREE.CapsuleGeometry(r,len,5,9),material,x,y,z,{rotation:[rx,0,0]});
  };

  // Torso: layered uniform, plate carrier, shoulder protection and webbing.
  capsule(root,uniform,0,1.42,0,.32,.55);
  box(root,vest,0,1.43,-.045,.66,.58,.34);
  box(root,shared.fabricDark,0,1.77,-.025,.57,.12,.3);
  for(const side of [-1,1]){
   box(root,shared.pouch,side*.245,1.43,-.225,.16,.22,.09,0,0,side*-.05);
   box(root,shared.pouch,side*.245,1.19,-.19,.15,.2,.1);
   box(root,shared.edge,side*.335,1.52,-.02,.055,.4,.32);
   box(root,shared.fabricDark,side*.13,1.66,-.222,.075,.27,.045,0,0,side*.2);
   // Belt and utility pouch.
   box(root,shared.edge,side*.17,1.08,-.015,.22,.12,.36);
  }
  box(root,shared.fabricDark,0,1.09,-.19,.28,.16,.16);
  box(root,shared.pouch,0,1.52,-.232,.2,.15,.055);

  // Neck, head, helmet shell, rim, goggles and ear protection.
  capsule(root,shared.skin,0,1.88,0,.105,.12);
  add(root,new THREE.SphereGeometry(.205,16,12),shared.skin,0,2.08,0);
  add(root,new THREE.SphereGeometry(.226,16,10),shared.helmet,0,2.205,.005,{scale:[1.04,.78,1.08]});
  add(root,new THREE.CylinderGeometry(.228,.228,.035,16),shared.fabricDark,0,2.105,0);
  box(root,shared.glass,0,2.105,-.196,.28,.075,.025);
  for(const side of [-1,1]){
   add(root,new THREE.CylinderGeometry(.065,.065,.07,10),shared.edge,side*.218,2.08,0,{rotation:[0,0,Math.PI/2]});
   box(root,shared.fabricDark,side*.22,1.83,-.015,.08,.23,.12);
  }
  // Backpack, visible from rear.
  box(root,shared.fabricDark,0,1.48,.235,.43,.52,.2);
  box(root,shared.pouch,0,1.22,.28,.3,.18,.12);
  for(const side of [-1,1])box(root,shared.edge,side*.16,1.52,.342,.09,.34,.035);

  // Articulated legs with knee pads and boots.
  for(const side of [-1,1]){
   const pivot=new THREE.Group();pivot.position.set(side*.18,.94,0);root.add(pivot);
   capsule(pivot,uniform,0,-.25,0,.135,.39);
   box(pivot,shared.armor,0,-.43,-.105,.2,.16,.09);
   capsule(pivot,shared.fabricDark,0,-.65,0,.105,.25);
   box(pivot,shared.rubber,0,-.79,-.055,.24,.16,.39);
   box(pivot,shared.sole,0,-.875,-.055,.25,.045,.4);
   legs.push(pivot);
  }

  // Arms are independent pivots; forearms, gloves and elbow protection.
  for(const side of [-1,1]){
   const pivot=new THREE.Group();pivot.position.set(side*.37,1.68,-.005);root.add(pivot);
   capsule(pivot,uniform,side*.015,-.19,-.015,.125,.31,-side*.12);
   add(pivot,new THREE.SphereGeometry(.125,10,8),shared.fabricDark,side*.015,-.43,-.035);
   capsule(pivot,shared.fabricDark,side*.015,-.55,-.08,.095,.2);
   box(pivot,shared.edge,side*.015,-.34,-.115,.17,.12,.07);
   arms.push(pivot);
  }

  // Original compact assault-rifle silhouette with receiver, handguard, barrel,
  // magazine, stock, optic, sights, grip and muzzle device.
  const rifle=new THREE.Group();
  box(rifle,shared.wood,-.28,1.47,-.56,.34,.16,.18,0,0,-.08);
  box(rifle,shared.metal,.015,1.49,-.57,.48,.16,.19);
  box(rifle,shared.edge,.23,1.49,-.57,.34,.19,.22);
  box(rifle,shared.fabricDark,.31,1.49,-.57,.22,.21,.23);
  add(rifle,new THREE.CylinderGeometry(.027,.027,.76,10),shared.metal,.25,1.49,-1.03,{rotation:[Math.PI/2,0,0]});
  add(rifle,new THREE.CylinderGeometry(.045,.045,.09,10),shared.edge,.25,1.49,-1.43,{rotation:[Math.PI/2,0,0]});
  box(rifle,shared.edge,.04,1.28,-.57,.14,.34,.13,0,0,-.08);
  box(rifle,shared.metal,.03,1.62,-.62,.12,.08,.13);
  box(rifle,shared.metal,.03,1.69,-.62,.09,.08,.1);
  box(rifle,shared.edge,.13,1.65,-.57,.2,.09,.1);
  root.add(rifle);
  root.userData={legs,arms,gun:rifle};
  return root;
 }
 setPosition(x,z){this.position.x=x;this.position.z=z;this.mesh.position.set(x,0,z)}
 animate(dt,speed,aim=false){
  this.anim+=dt*(speed>7?12:8);
  const swing=Math.sin(this.anim)*Math.min(1,speed/7)*.48,p=this.mesh.userData;
  p.legs[0].rotation.x=swing;p.legs[1].rotation.x=-swing;
  p.arms[0].rotation.x=aim?-.48:-swing*.2;p.arms[1].rotation.x=aim?-.52:-swing*.2;
  p.gun.rotation.x=aim?-.035:0;
 }
}
