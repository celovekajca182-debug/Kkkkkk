export const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const distance2=(ax,az,bx,bz)=>Math.hypot(bx-ax,bz-az);
export const normalize2=(x,z)=>{const n=Math.hypot(x,z);return n>1e-8?{x:x/n,z:z/n}:{x:0,z:0}};
export const segmentDistanceSq=(px,pz,ax,az,bx,bz)=>{const dx=bx-ax,dz=bz-az,t=dx*dx+dz*dz?clamp(((px-ax)*dx+(pz-az)*dz)/(dx*dx+dz*dz),0,1):0,ex=ax+t*dx-px,ez=az+t*dz-pz;return ex*ex+ez*ez};
export const randomRange=(r,a,b)=>a+(b-a)*r();
export function seededRandom(seed=1){let s=seed>>>0;return()=>((s=(s*1664525+1013904223)>>>0)/4294967296)}