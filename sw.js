const CACHE='last-zone-pwa-v5';
const AUDIO_ASSETS=[
 'https://sfxmint.com/dl/growth-65de718ff53302fbf797166e-retry.mp3',
 'https://spritepilot.com/free-assets/sound-effects/movement/steps-grass.ogg',
 'https://spritepilot.com/free-assets/sound-effects/scifi/reload.ogg',
 'https://spritepilot.com/free-assets/sound-effects/combat/punch.ogg',
 'https://spritepilot.com/free-assets/sound-effects/combat/body-fall.ogg',
 'https://spritepilot.com/free-assets/sound-effects/ambient/forest.ogg'
];
const SHELL=['./','./index.html','./manifest.webmanifest','./icon.svg','https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js',...AUDIO_ASSETS];
self.addEventListener('install',event=>event.waitUntil((async()=>{
 const cache=await caches.open(CACHE);
 for(const url of SHELL){try{await cache.add(url)}catch(error){console.warn('Не удалось сохранить ресурс:',url,error)}}
 await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
 for(const key of await caches.keys())if(key!==CACHE)await caches.delete(key);
 await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
 const req=event.request;if(req.method!=='GET')return;
 const url=new URL(req.url);
 if(AUDIO_ASSETS.includes(req.url)){
  event.respondWith((async()=>{
   const cache=await caches.open(CACHE),cached=await cache.match(req);
   if(cached)return cached;
   try{const response=await fetch(req);if(response.ok&&response.type!=='opaque')await cache.put(req,response.clone());return response}
   catch(error){return cached||Response.error()}
  })());
  return;
 }
 if(url.origin===self.location.origin){
  event.respondWith((async()=>{
   if(req.mode==='navigate'||url.pathname.endsWith('/index.html')){
    try{const fresh=await fetch(req);if(fresh.ok){const cache=await caches.open(CACHE);await cache.put(req,fresh.clone())}return fresh}
    catch(error){return await caches.match(req)||await caches.match('./index.html')||Response.error()}
   }
   const cached=await caches.match(req);if(cached)return cached;
   try{return await fetch(req)}catch(error){return caches.match('./index.html')}
  })());
  return;
 }
 if(url.hostname==='cdn.jsdelivr.net'){
  event.respondWith((async()=>{
   const cache=await caches.open(CACHE);
   try{const response=await fetch(req);if(response.ok)await cache.put(req,response.clone());return response}
   catch(error){return await cache.match(req)||Response.error()}
  })());
 }
});
