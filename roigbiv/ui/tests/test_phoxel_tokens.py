"""Executable contracts for the Phoxel design-system migration.

The app's palette lives in ``phoxel_tokens``, not in this repository. These
tests are what keeps it that way: they fail if a colour literal, an
under-floor type size, over-wide tracking, or a full-page scanline layer
reappears in ROIGBIV's own stylesheet.

Each one was written to fail against the pre-migration ``roigbiv.css`` first.
"""
from __future__ import annotations

import re
from pathlib import Path

import phoxel_tokens as pt
import pytest

CSS = Path(__file__).resolve().parents[1] / "assets" / "roigbiv.css"


@pytest.fixture(scope="module")
def css() -> str:
    return CSS.read_text()


@pytest.fixture(scope="module")
def app_client():
    from roigbiv.ui.app import build_app
    return build_app().server.test_client()


# ── the stylesheets are actually served ───────────────────────────────────
@pytest.mark.parametrize(
    "name", ["phoxel-tokens.css", "phoxel-bootstrap-bridge.css"])
def test_token_stylesheet_is_served(app_client, name):
    resp = app_client.get(f"/phoxel-assets/{name}")
    assert resp.status_code == 200, f"/phoxel-assets/{name} is not reachable"
    assert b":root" in resp.data


def test_token_stylesheets_load_before_the_app_sheet(app_client):
    """Order matters: roigbiv.css aliases these, so they have to exist first."""
    html = app_client.get("/").get_data(as_text=True)
    positions = [
        html.index("/phoxel-assets/phoxel-tokens.css"),
        html.index("/phoxel-assets/phoxel-bootstrap-bridge.css"),
        html.index("roigbiv.css"),
    ]
    assert positions == sorted(positions), (
        "roigbiv.css must be linked after both token stylesheets")


# ── ROIGBIV owns no colour ────────────────────────────────────────────────
def test_stylesheet_has_no_hex_literals(css):
    found = re.findall(r"#[0-9A-Fa-f]{3,8}\b", css)
    assert not found, (
        f"colour literals in roigbiv.css: {sorted(set(found))} — "
        "every colour must come from a phoxel-tokens custom property")


def test_stylesheet_has_no_tinted_rgba(css):
    """Black scrims are legibility devices over image data, not palette.

    Anything with a colour in it is a palette value and belongs in the package.
    """
    tinted = [
        m.group(0)
        for m in re.finditer(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", css)
        if any(float(g) > 0 for g in m.groups())
    ]
    assert not tinted, f"tinted rgba() literals in roigbiv.css: {tinted}"


def test_every_roigbiv_alias_points_at_a_real_token(css):
    """`--roigbiv-*` survives only as an alias — never as a value of its own."""
    aliases = re.findall(r"(--roigbiv-[a-z-]+):\s*([^;]+);", css)
    # --roigbiv-cells-chrome is a layout measurement, not a design token.
    aliases = [(k, v) for k, v in aliases if k != "--roigbiv-cells-chrome"]
    assert aliases, "the alias block disappeared"

    known = set(pt.COLOR) | set(pt.FONT) | {"radius"}
    for name, value in aliases:
        ref = re.fullmatch(r"var\(\s*--([a-z0-9-]+)\s*\)", value.strip())
        assert ref, f"{name} is set to {value!r} rather than a token reference"
        token = ref.group(1)
        assert token in known or token.startswith("font-"), (
            f"{name} aliases --{token}, which phoxel-tokens does not define")


def test_every_var_reference_resolves(css):
    """A typo'd token name renders as nothing at all, silently.

    Every custom property roigbiv.css reads must be defined by the token
    stylesheet, by the bootstrap bridge, or by roigbiv.css itself.
    """
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    for sheet in (pt.css_path(), pt.bootstrap_bridge_path()):
        defined |= set(re.findall(r"(--[a-z0-9-]+)\s*:", sheet.read_text()))

    used = set(re.findall(r"var\(\s*(--[a-z0-9-]+)", css))
    # --bs-* that CYBORG defines and this sheet only reads back.
    unresolved = {u for u in used - defined if not u.startswith("--bs-")}
    assert not unresolved, (
        f"roigbiv.css reads undefined custom properties: {sorted(unresolved)}")


def test_stylesheet_braces_balance(css):
    assert css.count("{") == css.count("}"), "unbalanced braces in roigbiv.css"
    assert css.count("/*") == css.count("*/"), "unclosed comment in roigbiv.css"


# ── the type floor and the tracking cap ───────────────────────────────────
_SCALE_PX = {f"--fs-{k}": int(v["size"].removesuffix("px"))
             for k, v in pt.FONT_SIZE.items()}


def _resolve_px(value: str) -> float | None:
    """Font size in px, or None when it is not a fixed length."""
    value = value.strip()
    if value in _SCALE_PX:
        return _SCALE_PX[value]
    ref = re.fullmatch(r"var\(\s*(--[a-z0-9-]+)\s*\)", value)
    if ref:
        return _SCALE_PX.get(ref.group(1))
    if m := re.fullmatch(r"([\d.]+)px", value):
        return float(m.group(1))
    if m := re.fullmatch(r"([\d.]+)rem", value):
        return float(m.group(1)) * 16
    return None


def test_no_type_below_the_floor(css):
    floor = pt.FONT_SIZE["micro"]["size"]
    floor_px = int(floor.removesuffix("px"))
    offenders = []
    for raw in re.findall(r"font-size:\s*([^;!]+)", css):
        px = _resolve_px(raw)
        if px is not None and px < floor_px:
            offenders.append(f"{raw.strip()} ({px:g}px)")
    assert not offenders, (
        f"font sizes below the {floor} floor: {offenders}")


def test_tracking_never_exceeds_the_cap(css):
    cap = 0.06
    offenders = []
    for raw in re.findall(r"letter-spacing:\s*([^;!]+)", css):
        raw = raw.strip()
        if m := re.fullmatch(r"([\d.]+)em", raw):
            if float(m.group(1)) > cap:
                offenders.append(raw)
        elif raw not in ("0", "normal") and not raw.startswith("var(--tracking-"):
            offenders.append(raw)
    assert not offenders, (
        f"tracking above the {cap}em cap, or off-token: {offenders}")


# ── the CRT layer is chrome-only ──────────────────────────────────────────
def test_no_full_page_scanline_layer(css):
    for banned in ("body::before", "body::after"):
        assert banned not in css, (
            f"{banned} is back — the CRT texture must not cover content")


def test_crt_class_is_applied_to_the_navbar_and_nothing_else():
    """`.crt` is opt-in chrome. The navbar is the only surface that qualifies."""
    from roigbiv.ui import app as app_module

    ui_root = Path(app_module.__file__).resolve().parent
    users = {}
    for path in ui_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for match in re.finditer(r'className="([^"]*)"', path.read_text()):
            if "crt" in match.group(1).split():
                users[path.name] = match.group(1)

    assert users == {"app.py": "roigbiv-navbar crt mb-2"}, (
        f".crt appears on unexpected surfaces: {users}")


def _luminance(hex_colour: str) -> float:
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(fg: str, bg: str) -> float:
    a, b = _luminance(fg), _luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def test_navbar_text_clears_aa_on_a_scanline_row(css):
    """The rule the CRT layer's token floor is a proxy for, measured directly.

    Every colour the navbar puts on text, composited against the darkest row
    the scanline gradient produces on --surface-2.
    """
    alpha = float(pt.CRT["scanlineColor"].rsplit(",", 1)[1].strip(" )"))
    surface = pt.color("surface-2").lstrip("#")
    darkest = "#%02X%02X%02X" % tuple(
        round(int(surface[i:i + 2], 16) * (1 - alpha)) for i in (0, 2, 4))

    navbar_block = css[css.index("── Navbar"):css.index("── Cards")]
    branding = css[css.index("── Branding"):css.index("── Navbar")]
    # Negative lookbehind so `border-color:` and `background-color:` — which
    # are not text — do not get swept in.
    tokens = sorted(set(re.findall(r"(?<![-\w])color:\s*var\(\s*(--[a-z-]+)\s*\)",
                                   navbar_block + branding)))
    assert tokens, "no text colours found on the navbar"

    for token in tokens:
        name = token.removeprefix("--")
        if name not in pt.COLOR:
            continue
        ratio = _contrast(pt.color(name), darkest)
        assert ratio >= 4.5, (
            f"{token} on a scanline row measures {ratio:.2f}:1 — "
            "navbar text must clear AA under the CRT overlay")


# ── the figure canvas keeps its own ground ────────────────────────────────
def test_figure_canvas_is_the_black_token():
    """The one documented departure from the shared Plotly template."""
    import plotly.io as pio

    from roigbiv.ui.services.theme import figure_paper_bg

    tmpl = pio.templates["roigbiv-reacher"]
    assert figure_paper_bg() == pt.color("surface-black")
    assert tmpl.layout.paper_bgcolor == pt.color("surface-black")
    assert tmpl.layout.plot_bgcolor == pt.color("surface-black")
    # ...and it really is a departure, not the shared value by coincidence.
    assert pt.PLOTLY_TEMPLATE["layout"]["paper_bgcolor"] != pt.color("surface-black")
