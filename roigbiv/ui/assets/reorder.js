/* Drag-and-drop reordering for the Track page's session list.
 *
 * Native HTML5 drag events, no dependency. Dash owns the list's contents, so
 * this never mutates state itself — on drop it reorders the DOM and writes the
 * resulting stem order into a hidden input, then dispatches an `input` event
 * so Dash's own listener picks the value up as a normal callback Input.
 *
 * Rows are re-rendered by Dash whenever the workspace is rescanned, so
 * listeners are attached by delegation on a container that persists, rather
 * than bound per-row at render time.
 */
(function () {
  "use strict";

  var LIST_ID = "roigbiv-track-list";
  var SINK_ID = "roigbiv-track-order-sink";
  var ITEM_SELECTOR = "[data-track-stem]";

  var dragged = null;

  function itemFromEvent(e) {
    return e.target && e.target.closest ? e.target.closest(ITEM_SELECTOR) : null;
  }

  function currentOrder(list) {
    return Array.prototype.map.call(
      list.querySelectorAll(ITEM_SELECTOR),
      function (el) { return el.getAttribute("data-track-stem"); }
    );
  }

  /* Dash listens for React's synthetic `input`, whose value it reads off the
   * node — setting `.value` directly bypasses React's setter, so go through
   * the prototype descriptor the way React expects. */
  function publish(order) {
    var sink = document.getElementById(SINK_ID);
    if (!sink) return;
    var setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value"
    ).set;
    setter.call(sink, JSON.stringify(order));
    sink.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function onDragStart(e) {
    var item = itemFromEvent(e);
    if (!item) return;
    dragged = item;
    item.classList.add("roigbiv-track-dragging");
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = "move";
      // Firefox refuses to start a drag without data set.
      e.dataTransfer.setData("text/plain", item.getAttribute("data-track-stem"));
    }
  }

  function onDragOver(e) {
    if (!dragged) return;
    var over = itemFromEvent(e);
    if (!over || over === dragged) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "move";

    // Insert before or after the hovered row depending on which half of it the
    // pointer is in, so a drag reads the way the cursor looks.
    var box = over.getBoundingClientRect();
    var after = (e.clientY - box.top) > (box.height / 2);
    over.parentNode.insertBefore(dragged, after ? over.nextSibling : over);
  }

  function onDrop(e) {
    if (!dragged) return;
    e.preventDefault();
    finish();
  }

  function finish() {
    if (!dragged) return;
    dragged.classList.remove("roigbiv-track-dragging");
    var list = document.getElementById(LIST_ID);
    dragged = null;
    if (list) publish(currentOrder(list));
  }

  document.addEventListener("dragstart", onDragStart, true);
  document.addEventListener("dragover", onDragOver, true);
  document.addEventListener("drop", onDrop, true);
  document.addEventListener("dragend", finish, true);
})();
