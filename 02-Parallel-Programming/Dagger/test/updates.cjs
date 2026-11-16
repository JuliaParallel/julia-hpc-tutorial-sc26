// Exercise banner timing without waiting thirty real seconds.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('host/app.js', 'utf8');
const snippet = source.slice(source.indexOf('let updateDismissTimer'), source.indexOf('const panes'));
const storage = new Map();
function setup() {
  const timers = new Map();
  let next = 0;
  const context = vm.createContext({
    setTimeout(fn, ms) { const id = ++next; timers.set(id, {fn, ms}); return id; },
    clearTimeout(id) { timers.delete(id); },
    sessionStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    updateControls() {},
  });
  vm.runInContext('let updateInfo = {}, showUpdateStatus = true;\n' + snippet, context);
  return {timers, run: code => vm.runInContext(code, context)};
}
const page = setup();
page.run('acceptUpdateInfo({status: "current"})');
assert.equal(page.timers.size, 1);
const original = [...page.timers.values()][0];
assert.equal(original.ms, 30000);
page.run('acceptUpdateInfo({status: "current"})');
assert.equal([...page.timers.values()][0], original, 'polling must not extend the timer');
original.fn();
assert.equal(page.run('showUpdateStatus'), false);
page.run('acceptUpdateInfo({status: "current"})');
assert.equal(page.run('showUpdateStatus'), false, 'polling must not restore a dismissed banner');
page.run('acceptUpdateInfo({status: "current", last_update: {id: "update-1", commits: 4}})');
assert.equal(page.run('updateSuccess'), 'Updated from main: pulled 4 commits.');
assert.equal(page.run('showUpdateStatus'), true);
const reload = setup();
reload.run('showUpdateStatus = false; acceptUpdateInfo({status: "current", last_update: {id: "update-1", commits: 4}})');
assert.equal(reload.run('showUpdateStatus'), false, 'reload must not repeat a receipt already displayed');
reload.run('acceptUpdateInfo({status: "current", last_update: {id: "update-2", commits: 1}})');
assert.equal(reload.run('updateSuccess'), 'Updated from main: pulled 1 commit.');
console.log('Banner timeout, polling, receipt persistence and commit counts passed');
