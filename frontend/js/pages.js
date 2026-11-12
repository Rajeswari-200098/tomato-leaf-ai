// Home + Diseases pages
document.addEventListener("DOMContentLoaded",async()=>{
  const el=document.getElementById("disease-grid"); if(!el) return;
  try{const d=await api("/api/diseases");el.innerHTML=d.map(x=>diseaseCard(x,el.dataset.full==="1")).join("")}
  catch(e){el.innerHTML=`<div class="alert err">${esc(e.message)}</div>`}
  const m=document.getElementById("metrics");
  if(m){try{const r=await api("/api/model-metrics");m.innerHTML=`<div class="card"><h3>Measured model performance</h3><p>${esc(r.architecture)} — test accuracy <b>${(r.test_accuracy*100).toFixed(2)}%</b> on ${r.test_samples} held-out images.</p></div>`}
   catch{m.innerHTML=`<div class="alert warn">No trained model yet, so no accuracy is reported. Train it with <code>python model/train.py</code>; real test results will appear here automatically.</div>`}}
});
