import base64
import io
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw


# =========================================================
# CONFIG
# =========================================================
DB_PATH = "floorplans.db"
st.set_page_config(page_title="FM Floorplan Editor", layout="wide")


# =========================================================
# DB HELPERS
# =========================================================
def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS floorplans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            source_filename TEXT,
            uploaded_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            floorplan_id INTEGER,
            room_identifier TEXT,
            source_name TEXT,
            display_name TEXT,
            department TEXT,
            geometry TEXT,
            attributes TEXT,
            FOREIGN KEY (floorplan_id) REFERENCES floorplans(id)
        )
        """
    )
    conn.commit()
    conn.close()


def save_floorplan_to_db(
    name: str,
    filename: str,
    rooms_df: pd.DataFrame,
    lookup: Dict[str, Dict[str, Any]],
) -> int:
    """
    Persist current floorplan + all rooms to SQLite.
    """
    conn = get_connection()
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat(timespec="seconds")

    cur.execute(
        "INSERT INTO floorplans (name, source_filename, uploaded_at) VALUES (?, ?, ?)",
        (name, filename, ts),
    )
    floorplan_id = cur.lastrowid

    for _, row in rooms_df.iterrows():
        rid = row.get("room_id")
        source_name = row.get("room_name")
        display_name = row.get("room_name")
        department = row.get("department") if pd.notna(row.get("department")) else None

        raw = lookup.get(rid, {})
        # prefer polygon, else bbox
        geom = raw.get("polygon") or raw.get("bbox") or raw.get("geometry")
        attrs = {k: v for k, v in raw.items() if k not in ("polygon", "bbox", "geometry")}

        cur.execute(
            """
            INSERT INTO rooms (
                floorplan_id,
                room_identifier,
                source_name,
                display_name,
                department,
                geometry,
                attributes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                floorplan_id,
                rid,
                source_name,
                display_name,
                department,
                json.dumps(geom) if geom else None,
                json.dumps(attrs) if attrs else None,
            ),
        )

    conn.commit()
    conn.close()
    return floorplan_id


def load_saved_floorplans() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM floorplans ORDER BY uploaded_at DESC", conn)
    conn.close()
    return df


def load_rooms_for_floorplan(fid: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM rooms WHERE floorplan_id = ? ORDER BY id", conn, params=(fid,))
    conn.close()
    return df


# =========================================================
# IMAGE / JSON HELPERS
# =========================================================
def decode_base64_image(b64_str: str) -> Image.Image:
    # strip data:image/...;base64, if present
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")


def json_to_rooms(json_data: Dict[str, Any]):
    """
    Parse our merged floorplan JSON into:
      - base image (PIL)
      - rooms dataframe
      - room_lookup (raw per-room dict)
      - page meta
    Expected JSON:
    {
      "pages": {
        "0": {
          "image_b64": "...",
          "image_size": {...},
          "rooms": [...]
        }
      }
    }
    """
    page = json_data["pages"]["0"]
    img = decode_base64_image(page["image_b64"])
    rooms = page.get("rooms", [])
    lookup = {}
    rows = []
    for r in rooms:
        rid = r.get("room_id") or r.get("id") or r.get("name")
        name = r.get("room_name") or r.get("name") or rid
        # we still store bbox in df to keep df view nice
        bbox = r.get("bbox") or {}
        rows.append(
            {
                "room_id": rid,
                "room_name": name,
                "department": "",
                "bbox": json.dumps(bbox),
            }
        )
        lookup[rid] = r

    df = pd.DataFrame(rows)
    return img, df, lookup, page


# =========================================================
# DRAW OVERLAY (polygon-aware)
# =========================================================
def bbox_to_polygon(bbox: Dict[str, Any]) -> List[List[float]]:
    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    w = float(bbox.get("width", 0))
    h = float(bbox.get("height", 0))
    return [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ]


def polygon_to_bbox(polygon: List[List[float]]) -> Dict[str, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "x": float(min_x),
        "y": float(min_y),
        "width": float(max_x - min_x),
        "height": float(max_y - min_y),
    }


def sync_page_rooms():
    if not st.session_state.get("page_meta") or st.session_state.get("rooms_df") is None:
        return

    rooms = []
    df = st.session_state.rooms_df
    for _, row in df.iterrows():
        rid = row["room_id"]
        raw = dict(st.session_state.room_lookup.get(rid, {}))
        raw.setdefault("room_id", rid)
        raw.setdefault("room_name", row.get("room_name", rid))
        dept_val = row.get("department")
        if isinstance(dept_val, str) and dept_val.strip():
            raw["department"] = dept_val.strip()
        polygon = raw.get("polygon")
        if polygon:
            raw["polygon"] = [[float(p[0]), float(p[1])] for p in polygon]
            if not raw.get("bbox"):
                raw["bbox"] = polygon_to_bbox(raw["polygon"])
        rooms.append(raw)

    st.session_state.page_meta["rooms"] = rooms


def draw_overlay(
    base_img: Image.Image,
    rooms_df: pd.DataFrame,
    highlight_ids: List[str] = None,
    room_lookup: Dict[str, Dict[str, Any]] = None,
) -> Image.Image:
    """
    Draws polygons if present, else falls back to rectangles (bbox).
    """
    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    highlight_ids = highlight_ids or []
    room_lookup = room_lookup or {}

    for _, row in rooms_df.iterrows():
        rid = row["room_id"]
        room_name = row.get("room_name", rid)
        raw = room_lookup.get(rid, {})

        # 1) polygon (high fidelity)
        polygon = raw.get("polygon")
        if polygon:
            pts = [(int(p[0]), int(p[1])) for p in polygon]
            is_selected = rid in highlight_ids
            outline = (255, 0, 0, 255) if is_selected else (0, 128, 255, 200)
            draw.line(pts + [pts[0]], fill=outline, width=4 if is_selected else 2)
            # label near first point
            if pts:
                draw.text((pts[0][0] + 4, pts[0][1] + 4), room_name[:22], fill=(0, 0, 0, 255))
            continue

        # 2) else try bbox
        bbox = raw.get("bbox")
        if not bbox:
            # maybe df has it as string
            df_bbox = row.get("bbox")
            if isinstance(df_bbox, str) and df_bbox.strip():
                try:
                    bbox = json.loads(df_bbox)
                except Exception:
                    bbox = None

        if bbox:
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("width", 50))
            h = int(bbox.get("height", 50))
            is_selected = rid in highlight_ids
            outline = (255, 0, 0, 255) if is_selected else (0, 128, 255, 160)
            draw.rectangle([x, y, x + w, y + h], outline=outline, width=4 if is_selected else 2)
            draw.text((x + 4, y + 4), room_name[:22], fill=(0, 0, 0, 255))

    return img


def render_polygon_canvas(
    image_b64: str,
    polygon: List[List[float]],
    canvas_key: str,
    width: int,
    height: int,
):
    image_src = f"data:image/png;base64,{image_b64}"
    polygon = polygon or []
    html = f"""
    <div class=\"polygon-editor\">
      <style>
        .polygon-editor {{
            width: 100%;
        }}
        .polygon-toolbar {{
            margin-bottom: 0.5rem;
        }}
        .polygon-toolbar button {{
            margin-right: 0.5rem;
        }}
      </style>
      <div class=\"polygon-toolbar\">
        <button id=\"add-corner-{canvas_key}\">Add corner</button>
        <button id=\"remove-corner-{canvas_key}\">Remove corner</button>
        <span>Drag the red handles to move room corners.</span>
      </div>
      <canvas id=\"editor-canvas-{canvas_key}\" width=\"{width}\" height=\"{height}\"></canvas>
    </div>
    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.2.4/fabric.min.js\"></script>
    <script>
    const Streamlit = (window.parent && window.parent.Streamlit) ? window.parent.Streamlit : window.Streamlit;
    if (Streamlit && Streamlit.setComponentReady) {{
        Streamlit.setComponentReady();
    }}
    const imageSrc = {json.dumps(image_src)};
    const initialPoints = {json.dumps(polygon)};
    const handleRadius = 8;
    let canvas, polygonObject, handles = [];

    function sendUpdate() {{
        if (!Streamlit || !polygonObject) return;
        const pts = polygonObject.points.map(pt => [Math.round(pt.x), Math.round(pt.y)]);
        if (Streamlit && Streamlit.setComponentValue) {{
            Streamlit.setComponentValue(pts);
        }}
        if (Streamlit && Streamlit.setFrameHeight) {{
            Streamlit.setFrameHeight({height} + 120);
        }}
    }}

    function clearHandles() {{
        handles.forEach(h => canvas.remove(h));
        handles = [];
    }}

    function addHandle(idx) {{
        const pt = polygonObject.points[idx];
        const circle = new fabric.Circle({{
            left: pt.x - handleRadius,
            top: pt.y - handleRadius,
            radius: handleRadius,
            fill: '#ff0000',
            stroke: '#ffffff',
            strokeWidth: 2,
            hasBorders: false,
            hasControls: false,
            hoverCursor: 'move',
            name: 'handle_' + idx
        }});
        circle.on('moving', () => {{
            const cx = circle.left + handleRadius;
            const cy = circle.top + handleRadius;
            polygonObject.points[idx].x = cx;
            polygonObject.points[idx].y = cy;
            polygonObject.dirty = true;
            canvas.renderAll();
            sendUpdate();
        }});
        handles.push(circle);
        canvas.add(circle);
    }}

    function rebuildHandles() {{
        clearHandles();
        polygonObject.points.forEach((_, idx) => addHandle(idx));
        canvas.renderAll();
        sendUpdate();
    }}

    function buildPolygon(points) {{
        let pts = points && points.length ? points.map(p => [p[0], p[1]]) : [];
        if (pts.length === 0) {{
            pts = [
                [Math.round({width} * 0.3), Math.round({height} * 0.3)],
                [Math.round({width} * 0.5), Math.round({height} * 0.3)],
                [Math.round({width} * 0.5), Math.round({height} * 0.5)],
                [Math.round({width} * 0.3), Math.round({height} * 0.5)]
            ];
        }}
        polygonObject = new fabric.Polygon(pts.map(p => {{ return {{ x: p[0], y: p[1] }}; }}), {{
            fill: '',
            stroke: '#ff0000',
            strokeWidth: 2,
            selectable: false,
            evented: false,
            objectCaching: false
        }});
        canvas.add(polygonObject);
        rebuildHandles();
    }}

    function initEditor() {{
        canvas = new fabric.Canvas('editor-canvas-{canvas_key}', {{ selection: false }});
        fabric.Image.fromURL(imageSrc, function(img) {{
            canvas.setWidth(img.width);
            canvas.setHeight(img.height);
            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
            buildPolygon(initialPoints);
            sendUpdate();
        }});
    }}

    document.getElementById('add-corner-{canvas_key}').addEventListener('click', () => {{
        if (!polygonObject) return;
        const cx = canvas.getWidth() / 2;
        const cy = canvas.getHeight() / 2;
        polygonObject.points.push({{ x: cx, y: cy }});
        polygonObject.dirty = true;
        rebuildHandles();
    }});

    document.getElementById('remove-corner-{canvas_key}').addEventListener('click', () => {{
        if (!polygonObject) return;
        if (polygonObject.points.length <= 3) return;
        polygonObject.points.pop();
        polygonObject.dirty = true;
        rebuildHandles();
    }});

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initEditor);
    }} else {{
        initEditor();
    }}
    </script>
    """
    return components.html(html, height=height + 160, key=f"canvas_{canvas_key}")


# =========================================================
# STREAMLIT APP
# =========================================================
def main():
    init_db()
    st.title("🏥 Floorplan / Facilities Editor")

    # ---- session init ----
    if "rooms_df" not in st.session_state:
        st.session_state.rooms_df = None
    if "room_lookup" not in st.session_state:
        st.session_state.room_lookup = {}
    if "base_image" not in st.session_state:
        st.session_state.base_image = None
    if "page_meta" not in st.session_state:
        st.session_state.page_meta = None
    if "selected_rooms" not in st.session_state:
        st.session_state.selected_rooms = []
    if "current_filename" not in st.session_state:
        st.session_state.current_filename = ""
    if "image_b64" not in st.session_state:
        st.session_state.image_b64 = None
    if "latest_canvas_points" not in st.session_state:
        st.session_state.latest_canvas_points = {}

    # ---- sidebar upload ----
    st.sidebar.header("1. Load floorplan JSON")
    uploaded = st.sidebar.file_uploader("Upload merged JSON", type=["json"])

    if uploaded:
        data = json.load(uploaded)
        base_img, df, lookup, page_meta = json_to_rooms(data)
        st.session_state.base_image = base_img
        st.session_state.rooms_df = df
        st.session_state.room_lookup = lookup
        st.session_state.page_meta = page_meta
        st.session_state.current_filename = uploaded.name
        st.session_state.image_b64 = page_meta.get("image_b64")
        st.session_state.latest_canvas_points = {}
        sync_page_rooms()
        st.success(f"Loaded {uploaded.name} with {len(df)} rooms")

    col_canvas, col_controls = st.columns([2, 1])

    # =====================================================
    # LEFT: CANVAS / OVERLAY
    # =====================================================
    with col_canvas:
        st.subheader("Floorplan preview")
        if st.session_state.base_image is not None and st.session_state.rooms_df is not None:
            overlay_img = draw_overlay(
                st.session_state.base_image,
                st.session_state.rooms_df,
                highlight_ids=st.session_state.selected_rooms,
                room_lookup=st.session_state.room_lookup,
            )
            # compatible with older streamlit
            try:
                st.image(overlay_img, caption="Floorplan with overlay", use_container_width=True)
            except TypeError:
                st.image(overlay_img, caption="Floorplan with overlay", use_column_width=True)

            st.markdown("#### Interactive room boundary editor")
            room_ids = st.session_state.rooms_df["room_id"].tolist()
            if room_ids:
                edit_choice = st.selectbox("Choose a room to adjust", ["--"] + room_ids, key="interactive_room_select")
                if edit_choice != "--":
                    raw_shape = st.session_state.room_lookup.get(edit_choice, {}).copy()
                    polygon = raw_shape.get("polygon")
                    if not polygon:
                        bbox = raw_shape.get("bbox")
                        if isinstance(bbox, str):
                            try:
                                bbox = json.loads(bbox)
                            except Exception:
                                bbox = None
                        if bbox:
                            polygon = bbox_to_polygon(bbox)
                    if polygon is None:
                        polygon = []

                    if st.session_state.image_b64:
                        canvas_key = f"room_{abs(hash(edit_choice))}"
                        result = render_polygon_canvas(
                            st.session_state.image_b64,
                            polygon,
                            canvas_key=canvas_key,
                            width=st.session_state.base_image.width,
                            height=st.session_state.base_image.height,
                        )
                        if result is not None:
                            parsed = result
                            if isinstance(result, str):
                                try:
                                    parsed = json.loads(result)
                                except Exception:
                                    parsed = None
                            if parsed:
                                st.session_state.latest_canvas_points[edit_choice] = parsed

                        if st.button("Apply shape changes", key=f"apply_{edit_choice}"):
                            updated_pts = st.session_state.latest_canvas_points.get(edit_choice)
                            if updated_pts:
                                polygon_pts = [[float(p[0]), float(p[1])] for p in updated_pts]
                                raw_shape["polygon"] = polygon_pts
                                raw_shape["bbox"] = polygon_to_bbox(polygon_pts)
                                raw_shape["room_id"] = edit_choice
                                st.session_state.room_lookup[edit_choice] = raw_shape
                                mask = st.session_state.rooms_df["room_id"] == edit_choice
                                st.session_state.rooms_df.loc[mask, "bbox"] = json.dumps(raw_shape["bbox"])
                                sync_page_rooms()
                                st.success("Updated room geometry.")
                            else:
                                st.warning("Move the handles before applying to capture new points.")
        else:
            st.info("Upload a JSON to see the floorplan.")

    # =====================================================
    # RIGHT: CONTROLS
    # =====================================================
    with col_controls:
        st.subheader("Rooms")

        if st.session_state.rooms_df is not None:
            df = st.session_state.rooms_df
            room_labels = [f"{r.room_name} ({r.room_id})" for _, r in df.iterrows()]
            label_to_id = {f"{r.room_name} ({r.room_id})": r.room_id for _, r in df.iterrows()}

            # single-select highlight
            sel_label = st.selectbox("Select a room to highlight", ["-- none --"] + room_labels)
            if sel_label != "-- none --":
                rid = label_to_id[sel_label]
                st.session_state.selected_rooms = [rid]
            else:
                st.session_state.selected_rooms = []

            st.markdown("**Multi-select rooms (for department):**")
            ms_labels = st.multiselect("Select rooms", room_labels, [])
            ms_ids = [label_to_id[l] for l in ms_labels]

            dept_name = st.text_input("Department name")
            if st.button("Assign selected rooms to department"):
                if not dept_name.strip():
                    st.warning("Enter a department name.")
                else:
                    updated = st.session_state.rooms_df.copy()
                    mask = updated["room_id"].isin(ms_ids)
                    updated.loc[mask, "department"] = dept_name.strip()
                    st.session_state.rooms_df = updated
                    sync_page_rooms()
                    st.success(f"Assigned {len(ms_ids)} rooms to '{dept_name.strip()}'")

            st.markdown("---")
            st.markdown("**Current rooms**")
            st.dataframe(st.session_state.rooms_df)

            # save to db
            if st.button("💾 Save to database"):
                plan_name = st.session_state.page_meta.get("title", "Floorplan") if st.session_state.page_meta else "Floorplan"
                fid = save_floorplan_to_db(
                    name=plan_name,
                    filename=st.session_state.current_filename,
                    rooms_df=st.session_state.rooms_df,
                    lookup=st.session_state.room_lookup,
                )
                st.success(f"Saved floorplan to DB with id={fid}")

            st.markdown("---")
            st.markdown("### Add a new room")
            with st.form("add_room_form"):
                new_room_id = st.text_input("Room ID", key="new_room_id")
                new_room_name = st.text_input("Room name", key="new_room_name")
                submitted_new = st.form_submit_button("Create room for editing")
                if submitted_new:
                    if not new_room_id.strip():
                        st.warning("Room ID is required.")
                    elif new_room_id in st.session_state.room_lookup:
                        st.warning("Room ID already exists.")
                    else:
                        name_value = new_room_name.strip() if new_room_name else new_room_id.strip()
                        width, height = st.session_state.base_image.size
                        default_poly = [
                            [float(width * 0.35), float(height * 0.35)],
                            [float(width * 0.55), float(height * 0.35)],
                            [float(width * 0.55), float(height * 0.55)],
                            [float(width * 0.35), float(height * 0.55)],
                        ]
                        bbox = polygon_to_bbox(default_poly)
                        st.session_state.room_lookup[new_room_id] = {
                            "room_id": new_room_id,
                            "room_name": name_value,
                            "polygon": default_poly,
                            "bbox": bbox,
                        }
                        new_row = pd.DataFrame(
                            [
                                {
                                    "room_id": new_room_id,
                                    "room_name": name_value,
                                    "department": "",
                                    "bbox": json.dumps(bbox),
                                }
                            ]
                        )
                        st.session_state.rooms_df = pd.concat(
                            [st.session_state.rooms_df, new_row], ignore_index=True
                        )
                        sync_page_rooms()
                        st.success("New room created. Use the editor to adjust its shape.")

            st.markdown("---")
            if st.session_state.page_meta and st.session_state.image_b64:
                sync_page_rooms()
                json_payload = {
                    "pages": {
                        "0": {**st.session_state.page_meta, "rooms": st.session_state.page_meta.get("rooms", [])}
                    }
                }
                filename = (st.session_state.current_filename or "floorplan.json").replace(".json", "")
                st.download_button(
                    "💾 Save JSON",
                    data=json.dumps(json_payload, indent=2),
                    file_name=f"{filename}_edited.json",
                    mime="application/json",
                )

    # =====================================================
    # POLYGON EDITOR (bottom)
    # =====================================================
    st.markdown("---")
    st.subheader("🟦 Polygon / shape editor (manual)")

    if st.session_state.rooms_df is not None:
        room_ids = st.session_state.rooms_df["room_id"].tolist()
        edit_room_id = st.selectbox("Pick a room to edit its shape", ["--"] + room_ids)
        if edit_room_id != "--":
            current_raw = st.session_state.room_lookup.get(edit_room_id, {})
            st.write("Current shape data:", current_raw)

            current_poly = current_raw.get("polygon", [])
            poly_str_default = json.dumps(current_poly, indent=2)

            new_poly_str = st.text_area(
                "Paste / edit polygon [[x1,y1],[x2,y2],...] (pixel coords)",
                value=poly_str_default,
                height=160,
            )

            if st.button("Update polygon for this room"):
                try:
                    new_poly = json.loads(new_poly_str)
                    current_raw["polygon"] = new_poly
                    # leave other fields intact (bbox etc)
                    st.session_state.room_lookup[edit_room_id] = current_raw
                    if new_poly:
                        bbox = polygon_to_bbox(new_poly)
                        st.session_state.room_lookup[edit_room_id]["bbox"] = bbox
                        mask = st.session_state.rooms_df["room_id"] == edit_room_id
                        st.session_state.rooms_df.loc[mask, "bbox"] = json.dumps(bbox)
                    sync_page_rooms()
                    st.success("Polygon updated. Scroll up to see overlay.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
    else:
        st.info("Load a floorplan to edit shapes.")

    # =====================================================
    # DB VIEWER
    # =====================================================
    st.markdown("---")
    st.subheader("📦 Database browser")

    fl_df = load_saved_floorplans()
    st.dataframe(fl_df)

    if not fl_df.empty:
        pick_id = st.selectbox("View rooms for floorplan id", [None] + fl_df["id"].tolist())
        if pick_id:
            r_df = load_rooms_for_floorplan(int(pick_id))
            st.dataframe(r_df)


if __name__ == "__main__":
    main()
