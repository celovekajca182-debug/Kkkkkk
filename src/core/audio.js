// Real recorded CC0 firearm reports and field footstep recordings.
// Source: yegors/hard-lines public/audio; source notes and original recordists:
// https://github.com/yegors/hard-lines/blob/main/public/audio/README.md
const BASE="https://raw.githubusercontent.com/yegors/hard-lines/main/public/audio/";
const SOURCES={
 M416:["hk416-report-0.wav","hk416-report-1.wav","hk416-report-2.wav"],
 SCARL:["hk416-report-0.wav","hk416-report-1.wav","hk416-report-2.wav"],
 AKM:["ak47-report-0.wav","ak47-report-1.wav","ak47-report-2.wav"],
 KAR98K:["awm-report-0.wav","awm-report-1.wav","awm-report-2.wav"]
};
export class AudioEngine{
 constructor(){this.enabled=true;this.ctx=null;this.master=null;this.samples=new Map();this.lastShot=0;this.lastBotShot=0}
 init(){if(!this.enabled)return null;const C=window.AudioContext||window.webkitAudioContext;if(!C)return null;if(!this.ctx){this.ctx=new C();this.master=this.ctx.createGain();this.master.gain.value=.8;this.master.connect(this.ctx.destination)}if(this.ctx.state==="suspended")this.ctx.resume();return this.ctx}
 setEnabled(value){this.enabled=value;if(this.master)this.master.gain.setTargetAtTime(value?.8:.0001,this.ctx.currentTime,.05)}
 sample(url){let a=this.samples.get(url);if(!a){a=new Audio(url);a.preload="auto";a.crossOrigin="anonymous";this.samples.set(url,a)}return a}
 play(url,volume=1){if(!this.enabled)return;const a=this.sample(url).cloneNode();a.volume=Math.max(0,Math.min(1,volume));a.play().catch(()=>{});}
 gunshot(name="M416",volume=1){const now=performance.now();if(volume<.99&&now-this.lastBotShot<115)return;if(volume<.99)this.lastBotShot=now;else if(now-this.lastShot<65)return;else this.lastShot=now;const list=SOURCES[name]||SOURCES.M416;const file=list[Math.floor(Math.random()*list.length)];this.play(BASE+file,volume)}
 reload(){/* No fake reload tone: connect a licensed recorded reload sample when added. */}
 hit(){/* No synthesized hit tone. */}
 footstep(run=false){const file="step-hard-"+Math.floor(Math.random()*4)+".wav";this.play(BASE+file,run?.62:.4)}
 pickup(){/* UI stays silent until a recorded interface sample is added. */}
}
