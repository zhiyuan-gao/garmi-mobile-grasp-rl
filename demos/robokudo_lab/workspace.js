// Compact entry; detailed perception opens separately like the NeemHub arrow.
addEventListener('DOMContentLoaded',()=>{
 const base=location.origin+(location.pathname.match(/^(\/user\/[^/]+\/)/)?.[1]||'/');
 const api=base+'proxy/8782/';
 const header=document.querySelector('header');
 const style=document.createElement('style');style.textContent='header [role=status]{display:inline-block;max-width:190px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}';document.head.append(style);
 const link=document.createElement('a');link.className='tab';link.textContent='RoboKudo ↗';link.href=api+'explorer.html';link.target='_blank';link.rel='noopener';header.append(link);
 const select=document.createElement('select');select.className='tab';select.setAttribute('aria-label','Perception target');
 for(const [value,text] of [['mug','Mug'],['bowl','Bowl']]){const o=document.createElement('option');o.value=value;o.textContent=text;select.append(o)}header.append(select);
 let token='';
 async function poll(){try{const s=await(await fetch(api+'api/status')).json();token=s.csrf_token;select.value=s.object;select.disabled=s.active||s.display_open;}catch(_){} }
 select.onchange=async()=>{const r=await fetch(api+'api/object',{method:'POST',headers:{'X-Demo-Token':token,'Content-Type':'application/json'},body:JSON.stringify({object:select.value})});if(!r.ok)alert((await r.json()).error);poll()};poll();setInterval(poll,1500);
 document.querySelector('.title').textContent='GARMI · RoboKudo + CRAM + RL';
});
