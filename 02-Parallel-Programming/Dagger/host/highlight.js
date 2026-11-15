"use strict";
// Julia syntax highlighting for the console's editors. The textarea stays the
// editor; a highlighted copy of its text sits behind it, in the colours the
// demo sources use (Highlights.jl's Dracula theme).

const JULIA_KEYWORDS = new Set([
  "function", "end", "if", "elseif", "else", "for", "while", "do", "in", "begin",
  "let", "return", "using", "import", "export", "module", "baremodule", "struct",
  "mutable", "const", "global", "local", "try", "catch", "finally", "quote",
  "macro", "break", "continue", "where", "abstract", "primitive", "type", "isa",
]);
const JULIA_CONSTANTS = new Set(["true", "false", "nothing", "missing", "Inf", "NaN", "pi", "im"]);

function escapeCode(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Split Julia source into [class, text] pairs; class "" is plain text.
function juliaTokens(code) {
  const tokens = [];
  const push = (kind, text) => {
    const last = tokens[tokens.length - 1];
    if (last && last[0] === kind) last[1] += text;
    else tokens.push([kind, text]);
  };
  // A quote after these is a transpose, not a character literal.
  const transposable = () => {
    const last = tokens[tokens.length - 1];
    if (!last) return false;
    const tail = last[1].slice(-1);
    if (last[0] === "") return /[\w)\]}'.!]/.test(tail);
    return ["name", "call", "num", "const"].includes(last[0]);
  };
  let i = 0;
  const n = code.length;
  while (i < n) {
    const rest = code.slice(i);
    let m;
    if (rest.startsWith("#=")) {
      // Nested block comment.
      let depth = 0, j = i;
      while (j < n) {
        if (code.startsWith("#=", j)) { depth++; j += 2; }
        else if (code.startsWith("=#", j)) { depth--; j += 2; if (depth === 0) break; }
        else j++;
      }
      push("comment", code.slice(i, j));
      i = j;
    } else if (rest[0] === "#") {
      const end = code.indexOf("\n", i);
      const stop = end < 0 ? n : end;
      push("comment", code.slice(i, stop));
      i = stop;
    } else if (rest.startsWith('"""') || rest[0] === '"') {
      const fence = rest.startsWith('"""') ? '"""' : '"';
      let j = i + fence.length;
      while (j < n && !code.startsWith(fence, j)) j += code[j] === "\\" ? 2 : 1;
      j = Math.min(n, j + fence.length);
      push("string", code.slice(i, j));
      i = j;
    } else if (rest[0] === "'" && !transposable() &&
               (m = rest.match(/^'(\\.[^']*|[^'\\])'/))) {
      push("string", m[0]);
      i += m[0].length;
    } else if (rest.startsWith("???")) {
      push("todo", "???");
      i += 3;
    } else if ((m = rest.match(/^@[A-Za-z_][\w!]*(\.[A-Za-z_][\w!]*)*/))) {
      push("macro", m[0]);
      i += m[0].length;
    } else if ((m = rest.match(/^(0x[\da-fA-F_]+|\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?)/))) {
      // Digits inside a name never get here: the name rule takes the whole word.
      push("num", m[0]);
      i += m[0].length;
    } else if ((m = rest.match(/^[A-Za-z_\u00a1-\uffff][\w!\u00a1-\uffff]*/))) {
      const word = m[0];
      const after = code.slice(i + word.length);
      if (JULIA_KEYWORDS.has(word)) push("kw", word);
      else if (JULIA_CONSTANTS.has(word)) push("const", word);
      else if (/^(\.)?\(/.test(after) || /^\{[^}]*\}\(/.test(after)) push("call", word);
      else if (/^[A-Z]/.test(word)) push("type", word);
      else push("name", word);
      i += word.length;
    } else if (rest[0] === ":" && (m = rest.match(/^:[A-Za-z_][\w!]*/)) &&
               !transposable() && !/[\w)\]]\s*$/.test(code.slice(Math.max(0, i - 2), i))) {
      push("symbol", m[0]);
      i += m[0].length;
    } else {
      push("", rest[0]);
      i += 1;
    }
  }
  return tokens;
}

function highlightJulia(code) {
  return juliaTokens(code).map(([kind, text]) =>
    kind && kind !== "name" ? `<span class="tok-${kind}">${escapeCode(text)}</span>` : escapeCode(text)
  ).join("");
}

// Lay a highlighted copy behind a textarea. Every way the value changes --
// typing, or code setting .value -- redraws it.
function enhanceEditor(textarea) {
  if (!textarea || textarea._highlight) return textarea;
  const wrapper = document.createElement("div");
  wrapper.className = "code-editor";
  if (textarea.classList.contains("exercise-setup")) wrapper.classList.add("tight");
  const pre = document.createElement("pre");
  pre.className = "code-highlight";
  pre.setAttribute("aria-hidden", "true");
  const code = document.createElement("code");
  pre.append(code);
  textarea.replaceWith(wrapper);
  wrapper.append(pre, textarea);
  textarea.setAttribute("wrap", "off");
  textarea._highlight = pre;
  const render = () => {
    // A trailing newline needs a character after it to occupy a line.
    code.innerHTML = highlightJulia(textarea.value) + "\n ";
    pre.scrollTop = textarea.scrollTop;
    pre.scrollLeft = textarea.scrollLeft;
  };
  const native = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
  Object.defineProperty(textarea, "value", {
    configurable: true,
    get() { return native.get.call(this); },
    set(text) { native.set.call(this, text); render(); },
  });
  textarea.addEventListener("input", render);
  textarea.addEventListener("scroll", () => {
    pre.scrollTop = textarea.scrollTop;
    pre.scrollLeft = textarea.scrollLeft;
  });
  render();
  return textarea;
}

if (typeof module !== "undefined") module.exports = {juliaTokens, highlightJulia};
