"use strict";
const $ = id => document.getElementById(id);
let config, selected, state = {}, cursor = 0, activeTrace = null;
const traces = [];
function ansi(parent, text) {
  for (const run of ansiSegments(text)) {
    const span = document.createElement("span");
    span.textContent = run.text;
    Object.assign(span.style, run.style);
    parent.append(span);
  }
}
async function post(path, body={}) {
  $("alert").hidden=true;
  try {
    const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Console-Token":config.token},body:JSON.stringify(body)});
    const data=await response.json(); if(!response.ok)throw Error(data.error);
  } catch(error) { alertError(error.message); }
}
function alertError(message) { $("alert").textContent=message; $("alert").hidden=false; }
function selectDemo(demo) {
  selected=demo;
  $("demo-title").textContent=demo[1];
  $("layout-label").textContent=demo[2]==="mpi"?"TWO MPI RANKS / ONE ODO NODE":"ONE JULIA PROCESS";
  $("gpu-control").hidden=demo[0]!=="05_multigpu";
  $("repl-mode").value=demo[2];
  for(const button of $("demos").children)button.classList.toggle("active",button.dataset.demo===demo[0]);
}
function controls() {
  $("status").textContent=state.status;
  $("connect").disabled=state.connecting||state.connected;
  $("disconnect").disabled=!state.connected&&!state.connecting;
  $("run").disabled=$("eval").disabled=!state.job||state.busy;
  $("cancel").disabled=!state.busy;
}
function event(item) {
  if(item.kind==="output") {
    const out=$("output"), follow=out.scrollTop+out.clientHeight>=out.scrollHeight-30;
    ansi(out,item.text); if(follow)out.scrollTop=out.scrollHeight;
    while(out.childNodes.length>6000)out.firstChild.remove();
  } else if(item.kind==="run") {
    const demo=config.demos.find(d=>d[0]===item.label);if(demo)selectDemo(demo);
    const gpuArg=item.args?.find(a=>a.startsWith("--gpus="));if(gpuArg)$("gpus").value=gpuArg.split("=")[1];
    const pending=document.createElement("p");pending.className="placeholder";pending.textContent="Starting Julia… algorithm source will appear before computation begins.";$("source").replaceChildren(pending);
    ansi($("output"),`\n━━ ${item.label} ━━\n`);
    $("output").scrollTop=$("output").scrollHeight;
  } else if(item.kind==="source") {
    const container=$("source"), heading=document.createElement("h4"), pre=document.createElement("pre");
    container.querySelector(".placeholder")?.remove();
    heading.textContent=item.file; ansi(pre,item.ansi||item.code); container.append(heading,pre);
  } else if(item.kind==="timeline") {
    traces.push(item); if(traces.length>20)traces.shift(); activeTrace=item; draw();
  } else if(item.kind==="error" || (item.kind==="done"&&!item.ok)) alertError(item.error);
}
function svg(tag, attrs={}, text) {
  const element=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const [key,value] of Object.entries(attrs))element.setAttribute(key,value);
  if(text!==undefined)element.textContent=text;
  return element;
}
function draw() {
  if(!activeTrace)return;
  const trace=activeTrace, events=trace.events.filter(e=>$("moves").checked||e.category==="compute");
  $("download").disabled=false;
  $("history").replaceChildren();
  traces.forEach((t,i)=>{const b=document.createElement("button");b.textContent=`${i+1} · ${t.title}`;b.classList.toggle("active",t===trace);b.onclick=()=>{activeTrace=t;draw();};$("history").append(b);});
  const all=trace.events, lanes=[...new Set(events.map(e=>e.lane))].sort();
  const duration=Math.max(0.000001,...all.map(e=>e.start+e.duration));
  const compute=all.filter(e=>e.category==="compute");
  $("metrics").replaceChildren();
  for(const [value,label] of [[duration.toFixed(3),"seconds"],[compute.length,"compute tasks"],[new Set(all.map(e=>e.rank)).size,"ranks"],[lanes.length,"lanes"]]){
    const span=document.createElement("span"), strong=document.createElement("strong");strong.textContent=value;span.append(strong,document.createTextNode(label));$("metrics").append(span);
  }
  const left=235, right=25, row=33, top=35;
  const width=Math.max(700,$("timeline").clientWidth)*Number($("zoom").value), height=top+row*lanes.length+38;
  const plot=svg("svg",{width,height,role:"img","aria-label":`${trace.title} task timeline`}), x=t=>left+t/duration*(width-left-right);
  for(let i=0;i<=8;i++){
    const t=i*duration/8, at=x(t);plot.append(svg("line",{x1:at,x2:at,y1:top-10,y2:height-28,stroke:"#2d4350"}));
    plot.append(svg("text",{x:at,y:height-8,"text-anchor":"middle"},t.toFixed(duration<1?3:2)));
  }
  lanes.forEach((lane,i)=>{
    const y=top+i*row;
    const label=svg("text",{x:5,y:y+14},lane.length>33?lane.slice(0,32)+"…":lane);label.append(svg("title",{},lane));plot.append(label);
    plot.append(svg("line",{x1:left,x2:width-right,y1:y+10,y2:y+10,stroke:"#203541"}));
    events.filter(e=>e.lane===lane).sort((a,b)=>b.duration-a.duration).forEach(e=>{
      const rect=svg("rect",{x:x(e.start),y:y+(e.category==="move"?15:1),width:Math.max(2,x(e.start+e.duration)-x(e.start)),height:e.category==="move"?5:12,rx:2,fill:e.category==="compute"?"#6acde8":"#ffbd75"});
      rect.append(svg("title",{},`${e.name}\n${e.lane}\n${e.category} · ${(e.duration*1000).toFixed(3)} ms\nstart ${e.start.toFixed(6)} s`));plot.append(rect);
    });
  });
  $("timeline").replaceChildren(plot);
  const totals=new Map();compute.forEach(e=>totals.set(e.name,(totals.get(e.name)||0)+e.duration));
  const ordered=[...totals].sort((a,b)=>b[1]-a[1]).slice(0,8), max=Math.max(1e-9,...ordered.map(x=>x[1]));
  const heading=document.createElement("h4");heading.textContent="Compute time by function · summed across lanes, may overlap";$("summary").replaceChildren(heading);
  ordered.forEach(([name,value])=>{const r=document.createElement("div"),n=document.createElement("span"),track=document.createElement("div"),bar=document.createElement("div"),v=document.createElement("span");r.className="bar-row";n.className="bar-name";n.textContent=name;n.title=name;track.className="bar-track";bar.className="bar-fill";bar.style.width=`${100*value/max}%`;v.textContent=`${(value*1000).toFixed(2)} ms`;track.append(bar);r.append(n,track,v);$("summary").append(r);});
}
async function poll() {
  try { const initial=cursor===0;const response=await fetch(`/api/state?after=${cursor}`);state=await response.json();state.events.forEach(event);cursor=state.seq;controls();if(initial)$("output").scrollTop=$("output").scrollHeight; }
  catch(error) { $("status").textContent="Local server unavailable"; }
  setTimeout(poll,700);
}
async function init() {
  config=await(await fetch("/api/config")).json();
  $("remote").textContent=`${config.host}:${config.remote_root}`;
  config.demos.forEach((demo,i)=>{const b=document.createElement("button"),num=document.createElement("span");num.className="num";num.textContent=String(i+1).padStart(2,"0");b.dataset.demo=demo[0];b.append(num,document.createTextNode(demo[1]));b.onclick=()=>selectDemo(demo);$("demos").append(b);});
  selectDemo(config.demos[0]);
  $("connect").onclick=()=>post("/api/connect");$("disconnect").onclick=()=>post("/api/disconnect");
  $("run").onclick=()=>post("/api/run",{op:"demo",demo:selected[0],gpus:$("gpus").value});
  $("eval").onclick=()=>post("/api/run",{op:"eval",code:$("code").value,mode:$("repl-mode").value});
  $("code").onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){e.preventDefault();if(!$("eval").disabled)$("eval").click();}};
  $("cancel").onclick=()=>post("/api/cancel");$("clear").onclick=()=>$("output").replaceChildren();
  $("moves").onchange=$("zoom").oninput=draw;window.addEventListener("resize",draw);
  $("download").onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(activeTrace,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=`dagger-${activeTrace.run||"timeline"}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  poll();
}
init().catch(error=>alertError(error.message));
