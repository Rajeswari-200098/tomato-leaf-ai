async function load(){
  const [s,h]=await Promise.all([api("/api/stats"),api("/api/history")]);
  const max=Math.max(1,...s.per_class.map(c=>c.count));
  document.getElementById("stats").innerHTML=`<p><b>${s.total}</b> total predictions</p>`+s.per_class.map(c=>`<div class="row"><span>${esc(c.name)}</span><span>${c.count}${c.count?` · avg ${c.avg_confidence}%`:""}</span></div><div class="bar"><div style="width:${c.count/max*100}%"></div></div>`).join("");
  document.getElementById("hist").innerHTML=h.length?`<table><tr><th></th><th>Result</th><th>Confidence</th><th>Date</th></tr>${h.map(r=>`<tr><td><img src="${r.image_url}" alt=""></td><td>${esc(r.name)} <span class="chip ${r.is_healthy?"":"bad"}">${r.is_healthy?"Healthy":"Disease"}</span></td><td>${r.confidence.toFixed(2)}%</td><td>${esc(r.created.replace("T"," "))}</td></tr>`).join("")}</table>`:`<div class="alert warn">No predictions yet. Analyze a leaf first.</div>`;
}
document.addEventListener("DOMContentLoaded",()=>{load().catch(e=>document.getElementById("hist").innerHTML=`<div class="alert err">${esc(e.message)}</div>`);
 document.getElementById("clear").onclick=async()=>{if(confirm("Delete all prediction history and uploaded images?")){await api("/api/history",{method:"DELETE"});load()}}});
