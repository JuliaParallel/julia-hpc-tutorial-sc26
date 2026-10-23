"use strict";
// Also normalize saved reports produced before explicit processor ownership.
function normalizeTimeline(trace) {
  return {...trace, events: trace.events.flatMap(event => {
    if(event.logger_rank !== undefined)return [event];
    const match=event.lane.match(/\/ \(MPI: (\d+), (.*)\)$/);
    const owner=match?Number(match[1]):event.rank;
    const category=event.name==="move!"?"move":event.category;
    if(category==="compute"&&owner!==event.rank)return [];
    const processor=match?match[2]:event.lane.split(" / ").slice(1).join(" / ");
    return [{...event, category, logger_rank:event.rank, target_rank:owner, processor,
      source:null,destination:null,
      lane:category==="move"?`rank ${event.rank} / data movement`:`rank ${owner} / ${processor}`}];
  })};
}
function taskDetails(event) {
  const lines=[event.lane,`${event.category} · ${(event.duration*1000).toFixed(3)} ms`, `start ${event.start.toFixed(6)} s`];
  if(event.category==="move") {
    if(event.source&&event.destination) {
      lines.push(`Source: rank ${event.source.rank} / ${event.source.space}`,
        `Destination: rank ${event.destination.rank} / ${event.destination.space}`);
    } else lines.push(`Destination processor: rank ${event.target_rank} / ${event.processor}`, "Source endpoint not recorded");
    lines.push(`Recorded on rank ${event.logger_rank}`);
  }
  return lines.join("\n");
}
if(typeof module!=="undefined")module.exports={normalizeTimeline,taskDetails};
