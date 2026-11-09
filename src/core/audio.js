export class AudioEngine{
constructor(){this.ctx=null;this.master=null;this.buffer=null;this.enabled=true}
init(){if(!this.enabled)return null;if(!this.ctx){const C=window.AudioContext||window.webkitAudioContext;if(!C)return null;this.ctx=new C();this.master=this.ctx.createGain();this.master.gain.value=.55;const comp=this.ctx.createDynamicsCompressor();this.master.connect(comp);comp.connect(this.ctx.destination);this.buffer=this.ctx.createBuffer(1,this.ctx.sampleRate,this.ctx.sampleRate);const a=this.buffer.getChannelData(0);for(let i=0;i<a.length;i++)a[i]=Math.random()*2-1}if(this.ctx.state==="suspended")this.ctx.resume();return this.ctx}
setEnabled(v){this.enabled=v;if(v)this.init();if(this.master)this.master.gain.setTargetAtTime(v?.55:.0001,this.ctx.currentTime,.05)}
noise(duration,volume,frequency,filterType="lowpass"){const c=this.init();if(!c)return;const src=c.createBufferSource(),f=c.createBiquadFilter(),g=c.createGain(),t=c.currentTime;src.buffer=this.buffer;f.type=filterType;f.frequency.value=frequency;g.gain.setValueAtTime(volume,t);g.gain.exponentialRampToValueAtTime(.0001,t+duration);src.connect(f);f.connect(g);g.connect(this.master);src.start();src.stop(t+duration)}
tone(a,b,d,v,wave="triangle"){const c=this.init();if(!c)return;const o=c.createOscillator(),g=c.createGain(),t=c.currentTime;o.type=wave;o.frequency.setValueAtTime(a,t);o.frequency.exponentialRampToValueAtTime(b,t+d);g.gain.setValueAtTime(v,t);g.gain.exponentialRampToValueAtTime(.0001,t+d);o.connect(g);g.connect(this.master);o.start();o.stop(t+d)}
gunshot(name="M416",volume=1){if(!this.enabled)return;const heavy=name==="AKM"||name==="KAR98K";this.noise(heavy?.2:.13,.22*volume,heavy?1000:1600);this.noise(.04,.2*volume,4800,"highpass");this.tone(heavy?100:145,40,.12,.09*volume)}
reload(){this.noise(.06,.1,1800,"highpass");this.tone(300,180,.08,.04)}
hit(){this.noise(.06,.12,900);this.tone(210,90,.09,.04)}
footstep(run=false){this.noise(.07,run?.07:.045,220)}
pickup(){this.tone(520,850,.12,.05)}
}