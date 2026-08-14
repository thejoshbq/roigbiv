"""Walking a Dash component tree, for tests that assert on rendered layout.

The pages are built from nested ``html``/``dbc`` components, so almost every
layout assertion is "is this id present", "is this control the right type", or
"does this text appear". Those three questions used to be re-implemented at the
top of each page's test module.
"""
from __future__ import annotations


def walk(component):
    """Yield ``component`` and every descendant in its children tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        # Skip raw strings / numbers — only Dash components have children.
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from walk(child)


def ids(root) -> set:
    return {getattr(c, "id", None) for c in walk(root)}


def find_by_id(root, target_id):
    for comp in walk(root):
        if getattr(comp, "id", None) == target_id:
            return comp
    raise AssertionError(f"component id={target_id!r} not found in layout")


def h6_texts(root) -> list:
    out = []
    for c in walk(root):
        if type(c).__name__ == "H6":
            kids = c.children
            out.append(kids if isinstance(kids, str) else str(kids))
    return out


def text(root) -> str:
    """All string content in the tree, space-joined."""
    parts = []
    for c in walk(root):
        kids = getattr(c, "children", None)
        if isinstance(kids, str):
            parts.append(kids)
        elif isinstance(kids, (list, tuple)):
            parts.extend(k for k in kids if isinstance(k, str))
    return " ".join(parts)
