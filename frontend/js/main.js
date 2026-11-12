// Shared navigation, footer and helpers
const PAGES=[["/","Home"],["/detect.html","Detect Disease"],["/diseases.html","Diseases"],["/how-it-works.html","How It Works"],["/about.html","About"],["/dashboard.html","Dashboard"]];
function mountChrome(){
  const here=location.pathname==="/"?"/":location.pathname;
  document.getElementById("nav").innerHTML=`<nav><div class="container"><a class="brand" href="/">🍅 TomatoLeaf&nbsp;AI</a>
   <button class="menu-btn" aria-label="Menu" onclick="document.querySelector('.links').classList.toggle('open')">☰</button>
   <div class="links">${PAGES.map(([h,t])=>`<a href="${h}" class="${h===here?"active":""}">${t}</a>`).join("")}</div></div></nav>`;
  document.getElementById("footer").innerHTML=`<footer><div class="container"><strong>TomatoLeaf AI</strong> — Deep Learning-Based Tomato Leaf Disease Detection and Classification Using Computer Vision.<br>
   B.Tech Machine Learning course project. Predictions are decision support only; confirm serious cases with an agricultural expert.</div></footer>`;
}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const li=a=>`<ul class="clean">${a.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`;
const ICON=s=>s==="healthy"?"🌿":s==="spider-mites"?"🕷️":s.includes("virus")?"🦠":"🍂";
// Disease photo if you add frontend/img/diseases/<slug>.jpg, otherwise an icon placeholder
const photo=s=>`<img class="photo" src="/img/diseases/${s}.jpg" alt="" onerror="this.outerHTML='<div class=&quot;ph&quot;>${ICON(s)}</div>'">`;
async function api(p,o){const r=await fetch(p,o);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||"Request failed");return j}
function diseaseCard(d,full){return `<div class="card dcard">${photo(d.slug)}<span class="chip ${d.healthy?"":"bad"}">${d.healthy?"Healthy":"Disease"}</span>
 <h3 style="margin-top:8px">${esc(d.name)}</h3><p>${esc(d.description)}</p>${full?`<small><b>Type:</b> ${esc(d.cause_type)}</small>
 <h4>Symptoms</h4>${li(d.symptoms)}<h4>Possible causes</h4>${li(d.causes)}<h4>Prevention</h4>${li(d.prevention)}`:""}</div>`}
document.addEventListener("DOMContentLoaded",mountChrome);
