// Compact entry; detailed perception opens separately like the NeemHub arrow.
addEventListener('DOMContentLoaded',()=>{
 const base=location.origin+(location.pathname.match(/^(\/user\/[^/]+\/)/)?.[1]||'/');
 const api=base+'proxy/8782/';
 const header=document.querySelector('header');
 const style=document.createElement('style');style.textContent='header [role=status]{display:inline-block;max-width:190px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}';document.head.append(style);
 const link=document.createElement('a');link.className='tab';link.textContent='RoboKudo ↗';link.href=api+'explorer.html';link.target='_blank';link.rel='noopener';header.append(link);
 document.querySelector('.title').textContent='GARMI · RoboKudo household';
});
