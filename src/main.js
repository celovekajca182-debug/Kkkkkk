import {Game} from "./game/game.js";
const canvas=document.querySelector("#game");
try{const game=new Game(canvas);document.querySelector("#play").disabled=false;window.lastZone=game}catch(error){console.error(error);document.querySelector("#menuText").textContent="Не удалось запустить 3D-движок: "+error.message;document.querySelector("#play").disabled=true}