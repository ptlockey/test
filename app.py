import base64
import io
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
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
            fill = (255, 0, 0, 64) if is_selected else (0, 128, 255, 40)
            outline = (255, 0, 0, 255) if is_selected else (0, 128, 255, 200)
            draw.polygon(pts, fill=fill, outline=outline)
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
            fill = (255, 0, 0, 64) if is_selected else (0, 128, 255, 30)
            outline = (255, 0, 0, 255) if is_selected else (0, 128, 255, 160)
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=3 if is_selected else 1)
            draw.text((x + 4, y + 4), room_name[:22], fill=(0, 0, 0, 255))

    return img


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
