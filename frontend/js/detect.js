let file=null,last=null;
const $=id=>document.getElementById(id);
const drop=$("drop"),input=$("file"),analyze=$("analyze");
function setFile(f){
  $("error").innerHTML="";
  if(!f)return;
  if(!/^image\/(jpeg|png|webp)$/.test(f.type))return showErr("Please choose a JPG, PNG or WEBP image.");
  if(f.size>5*1024*1024)return showErr("Image is larger than 5 MB.");
  file=f;$("preview").src=URL.createObjectURL(f);$("upload-view").classList.add("hidden");$("preview-view").classList.remove("hidden");analyze.disabled=false;
}
const showErr=m=>$("error").innerHTML=`<div class="alert err">${esc(m)}</div>`;
drop.onclick=()=>input.click();input.onchange=()=>setFile(input.files[0]);
["dragover","dragenter"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add("over")}));
["dragleave","drop"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove("over")}));
drop.addEventListener("drop",ev=>setFile(ev.dataTransfer.files[0]));
$("reset").onclick=reset;$("again").onclick=reset;
function reset(){file=null;input.value="";analyze.disabled=true;$("upload-view").classList.remove("hidden");$("preview-view").classList.add("hidden");$("result").classList.add("hidden");$("loading").classList.add("hidden");$("error").innerHTML="";window.scrollTo({top:0,behavior:"smooth"})}
analyze.onclick=async()=>{
  if(!file)return;$("error").innerHTML="";analyze.disabled=true;$("loading").classList.remove("hidden");$("result").classList.add("hidden");
  const fd=new FormData();fd.append("image",file);
  try{last=await api("/api/predict",{method:"POST",body:fd});render(last)}catch(e){showErr(e.message)}
  $("loading").classList.add("hidden");analyze.disabled=false;
};
function render(r){
  const p=r.prediction,i=r.info,cls=p.is_healthy?"":p.confidence<60?"warn":"";
  $("result").innerHTML=`<div class="card"><div class="two"><div>
   <div class="result-head"><h2 style="margin:0">${esc(p.name)}</h2><span class="chip ${p.is_healthy?"":"bad"}">${p.is_healthy?"Healthy leaf":"Diseased leaf"}</span></div>
   <div class="row"><span>Confidence</span><b>${p.confidence.toFixed(2)}%</b></div><div class="bar ${p.low_confidence?"warn":""}"><div id="cbar"></div></div>
   ${p.low_confidence?`<div class="alert warn">Low confidence. Try a clearer, well-lit close-up of a single leaf; this model only knows these 10 classes and may misjudge non-tomato images.</div>`:""}
   <h4>Other likely classes</h4>${r.top_predictions.map(t=>`<div class="row"><span>${esc(t.name)}</span><span>${t.confidence.toFixed(2)}%</span></div><div class="bar"><div style="width:${t.confidence}%"></div></div>`).join("")}
   </div><div><p>${esc(i.description)}</p><h4>Symptoms</h4>${li(i.symptoms)}<h4>Possible causes</h4>${li(i.causes)}<h4>Preventive measures</h4>${li(i.prevention)}</div></div>
   <p style="margin-top:18px"><button class="btn" id="report">⬇ Download report</button> <button class="btn ghost" id="again2">Analyze another image</button></p></div>`;
  $("result").classList.remove("hidden");requestAnimationFrame(()=>$("cbar").style.width=p.confidence+"%");
  $("again2").onclick=reset;$("report").onclick=()=>download(r);$("result").scrollIntoView({behavior:"smooth"});
}
function download(r){
  const p=r.prediction,i=r.info,L=a=>`<ul>${a.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`;
  const html=`<!doctype html><meta charset="utf-8"><title>Tomato Leaf Report</title><body style="font-family:sans-serif;max-width:720px;margin:30px auto"><h1>Tomato Leaf Analysis Report</h1>
  <p>Generated: ${esc(r.timestamp)}</p><img src="${r.image_url}" style="max-height:260px"><h2>${esc(p.name)} — ${p.is_healthy?"Healthy":"Diseased"}</h2>
  <p><b>Confidence:</b> ${p.confidence.toFixed(2)}%</p><p>${esc(i.description)}</p><h3>Symptoms</h3>${L(i.symptoms)}<h3>Possible causes</h3>${L(i.causes)}<h3>Prevention</h3>${L(i.prevention)}
  <p><small>AI-assisted result; confirm with an agricultural expert. Print this page to save as PDF.</small></p></body>`;
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([html],{type:"text/html"}));a.download=`tomato-report-${r.id.slice(0,8)}.html`;a.click();
}
api("/api/status").then(s=>{if(!s.model_available)$("error").innerHTML=`<div class="alert warn"><b>Model not trained yet.</b> Predictions are unavailable until you run <code>python model/train.py</code>. See the README.</div>`}).catch(()=>{});
