// Agent VCR website: copy buttons and the docs sidebar. No dependencies.
(function () {
    "use strict";

    // Copy buttons on every code block that has a title bar.
    document.querySelectorAll(".code:not([data-nocopy]) .code-bar").forEach(function (bar) {
        var pre = bar.parentElement.querySelector("pre");
        if (!pre || !navigator.clipboard) return;
        var button = document.createElement("button");
        button.type = "button";
        button.className = "copy-button";
        button.textContent = "Copy";
        button.setAttribute("aria-label", "Copy code to clipboard");
        button.addEventListener("click", function () {
            navigator.clipboard.writeText(pre.innerText.replace(/\n$/, "")).then(function () {
                button.textContent = "Copied";
                setTimeout(function () { button.textContent = "Copy"; }, 1400);
            });
        });
        bar.appendChild(button);
    });

    var nav = document.querySelector(".docs-nav");
    if (!nav) return;

    // The sidebar is a <details>: always open on desktop, a collapsed menu on phones.
    var narrow = window.matchMedia("(max-width: 960px)");
    function syncOpen() { nav.open = !narrow.matches; }
    syncOpen();
    if (narrow.addEventListener) narrow.addEventListener("change", syncOpen);
    nav.querySelector("summary").addEventListener("click", function (event) {
        if (!narrow.matches) event.preventDefault();
    });
    nav.addEventListener("click", function (event) {
        if (narrow.matches && event.target.closest("a")) nav.open = false;
    });

    // Highlight the section being read.
    var links = {};
    nav.querySelectorAll('a[href^="#"]').forEach(function (a) {
        links[a.getAttribute("href").slice(1)] = a;
    });
    var sections = Array.prototype.filter.call(
        document.querySelectorAll(".doc section[id]"),
        function (s) { return links[s.id]; }
    );
    if (!("IntersectionObserver" in window) || !sections.length) return;
    var current = null;
    function mark(id) {
        if (id === current) return;
        if (current && links[current]) links[current].removeAttribute("aria-current");
        current = id;
        if (links[id]) links[id].setAttribute("aria-current", "true");
    }
    var visible = {};
    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting; });
        for (var i = 0; i < sections.length; i++) {
            if (visible[sections[i].id]) { mark(sections[i].id); return; }
        }
    }, { rootMargin: "-80px 0px -60% 0px" });
    sections.forEach(function (s) { observer.observe(s); });
})();
