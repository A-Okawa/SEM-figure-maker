"""
SEM/EDS 論文処理アプリ
Streamlit-based GUI for processing SEM images and EDS data for publication.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
import zipfile
import math
import os
from pathlib import Path

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SEM/EDS 論文処理アプリ",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 SEM/EDS 論文処理アプリ")
st.caption("SEM像・EDSデータを論文出版用に処理するツール")

# ─────────────────────────────────────────────
#  Fonts
# ─────────────────────────────────────────────
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

def load_font(size: int, bold: bool = False):
    candidates = FONT_CANDIDATES if not bold else [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ] + FONT_CANDIDATES
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
if "images" not in st.session_state:
    st.session_state.images = {}   # name -> PIL Image
if "panel_result" not in st.session_state:
    st.session_state.panel_result = None

# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt="PNG", dpi=300) -> bytes:
    buf = io.BytesIO()
    if fmt == "TIFF":
        img.save(buf, format="TIFF", compression="lzw", dpi=(dpi, dpi))
    else:
        img.save(buf, format=fmt, dpi=(dpi, dpi))
    buf.seek(0)
    return buf.read()


def detect_sem_databar(pil_img: Image.Image) -> int:
    """
    Detect the black SEM data bar at the bottom of the image.
    Uses row MEDIAN (not mean) so white text inside the black strip
    doesn't fool the detector.
    Returns the y-coordinate (from top) where the data bar starts.
    Returns image height if no bar is detected.
    """
    arr = np.array(pil_img.convert("L"))
    h, w = arr.shape
    row_medians = np.median(arr, axis=1)  # shape (h,)

    # Scan from bottom upward.
    # Data-bar rows: mostly black → median ≈ 0.
    # Actual image rows: median clearly > near-zero.
    # Use a 3-row window to avoid noise from isolated dark/bright rows.
    WINDOW = 3
    MEDIAN_THRESH = 12  # median > this → not a data-bar row

    for y in range(h - 1, max(h - 300, 0), -1):
        y0 = max(0, y - WINDOW + 1)
        window_med = float(np.median(row_medians[y0 : y + 1]))
        if window_med > MEDIAN_THRESH:
            return y + 1   # data bar starts just below this actual-image row
    return h  # no databar found


def detect_scale_bar_in_region(pil_img: Image.Image, roi_y_start: int) -> tuple | None:
    """
    Detect the white scale bar within the SEM data strip (rows roi_y_start..bottom).
    Uses row-by-row longest-run analysis: the scale bar is a solid continuous white
    horizontal line, so its row will have the longest uninterrupted white pixel run.
    This avoids the bounding-rect merge problem where the bar + text label above it
    are grouped into one tall blob (ch≈15) that fails a thin-line filter.
    Returns (x_start, y_start, x_end, y_end) in full-image coordinates, or None.
    """
    arr = np.array(pil_img.convert("L"))
    h, w = arr.shape

    if roi_y_start >= h - 2:
        return None

    roi = arr[roi_y_start:h, :]
    _, binary = cv2.threshold(roi.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)

    best_run_len = 0
    best_row = -1
    best_x = 0
    rh = binary.shape[0]

    for y in range(rh):
        row = binary[y]
        padded = np.concatenate([[0], row, [0]])
        is_white = (padded == 255).astype(np.int8)
        transitions = np.diff(is_white)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        if len(starts) == 0:
            continue
        lengths = ends - starts
        max_idx = int(np.argmax(lengths))
        max_run = int(lengths[max_idx])
        max_x = int(starts[max_idx])
        if max_run > best_run_len and max_run > 15:
            best_run_len = max_run
            best_row = y
            best_x = max_x

    if best_run_len > 15 and best_row >= 0:
        return (best_x, roi_y_start + best_row,
                best_x + best_run_len, roi_y_start + best_row + 1)
    return None


def ocr_scale_label_from_databar(pil_img: Image.Image, roi_y_start: int,
                                  bar_region: tuple | None = None) -> str | None:
    """
    Run OCR on the region near the detected scale bar (not the full data bar)
    to extract a label such as '1 µm', '500 nm', '10 µm', etc.
    bar_region: (x1, y1, x2, y2) of the detected white scale bar.
    If provided, OCR is limited to a horizontal slice around the bar to avoid
    reading unrelated values like WD (working distance) on the right side.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]
    if roi_y_start >= h:
        return None

    # Limit horizontal range to near the scale bar to avoid WD text etc.
    if bar_region is not None:
        bx1, _, bx2, _ = bar_region
        bar_w = bx2 - bx1
        x_lo = max(0, bx1 - bar_w * 2)
        x_hi = min(w, bx2 + bar_w * 2)
        roi = arr[roi_y_start:h, x_lo:x_hi]
    else:
        roi = arr[roi_y_start:h, :]

    # Upscale for better OCR accuracy
    scale = 3
    roi_up = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale),
                        interpolation=cv2.INTER_CUBIC)

    # Binarize: white text on black → invert → dark text on white (tesseract prefers this)
    gray = cv2.cvtColor(roi_up, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    # Invert so text is black on white
    inv = cv2.bitwise_not(binary)
    roi_pil = Image.fromarray(inv)

    cfg = "--psm 6 -c tessedit_char_whitelist=0123456789.µumnMkK "
    raw = pytesseract.image_to_string(roi_pil, config=cfg).strip()

    # Normalize common OCR mistakes
    raw = raw.replace("um", "µm").replace("Um", "µm").replace("UM", "µm")
    raw = raw.replace("nm", "nm").replace("Nm", "nm").replace("NM", "nm")
    raw = raw.replace("mm", "mm")

    # Find scale patterns like "1 µm", "500 nm", "1.5 µm", "10nm" ...
    import re
    # Match: number (optional decimal) + optional space + unit
    pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*(µm|um|nm|mm|km)',
        re.IGNORECASE
    )
    matches = pattern.findall(raw)
    if not matches:
        return None

    # Prefer µm/nm over mm — "mm" in the data bar is usually WD (working distance),
    # not the scale label. Scale labels are typically in µm or nm.
    scale_units = {"µm", "um", "nm"}
    preferred = [(v, u) for v, u in matches if u.lower() in scale_units]
    val, unit = preferred[0] if preferred else matches[0]

    unit = unit.lower().replace("um", "µm")
    val_str = val.rstrip("0").rstrip(".") if "." in val else val
    return f"{val_str} {unit}"


def draw_material_name(draw, x, y, text, font_main, font_sub, use_subscript, color="white"):
    """Draw material name with optional subscript for digits and 'x'."""
    curr_x = x
    if not use_subscript:
        draw.text((x, y), text, font=font_main, fill=color)
        return
    for char in text:
        if char.isdigit() or char == "x":
            draw.text((curr_x, y + font_main.size // 3), char, font=font_sub, fill=color)
            curr_x += int(draw.textlength(char, font=font_sub)) + 1
        else:
            draw.text((curr_x, y), char, font=font_main, fill=color)
            curr_x += int(draw.textlength(char, font=font_main))


def draw_clean_scalebar(
    img: Image.Image,
    bar_length_px: int,
    label: str,
    position: str = "右下",
    color: str = "white",
    bar_thickness: int = 8,
    margin: int = 20,
    font_size: int = 30,
) -> Image.Image:
    """Add a clean scale bar and label to a PIL image. Returns new image."""
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = load_font(font_size)

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if position == "右下":
        x_end = w - margin
        x_start = x_end - bar_length_px
        bar_y = h - margin - bar_thickness
    elif position == "左下":
        x_start = margin
        x_end = x_start + bar_length_px
        bar_y = h - margin - bar_thickness
    elif position == "右上":
        x_end = w - margin
        x_start = x_end - bar_length_px
        bar_y = margin
    else:  # 左上
        x_start = margin
        x_end = x_start + bar_length_px
        bar_y = margin

    draw.rectangle([x_start, bar_y, x_end, bar_y + bar_thickness], fill=color)

    bar_cx = (x_start + x_end) // 2
    text_x = bar_cx - text_w // 2
    if position in ("右下", "左下"):
        text_y = bar_y - text_h - 4
    else:
        text_y = bar_y + bar_thickness + 4
    draw.text((text_x, text_y), label, fill=color, font=font)
    return img


def make_label_list(style: str, n: int) -> list[str]:
    if style == "(a), (b), (c)...":
        return [f"({chr(97+i)})" for i in range(n)]
    elif style == "(A), (B), (C)...":
        return [f"({chr(65+i)})" for i in range(n)]
    elif style == "a), b), c)...":
        return [f"{chr(97+i)})" for i in range(n)]
    elif style == "A), B), C)...":
        return [f"{chr(65+i)})" for i in range(n)]
    elif style == "1, 2, 3...":
        return [str(i+1) for i in range(n)]
    else:
        return [""] * n



def parse_jeol_txt(txt_content: str) -> dict:
    """
    Parse JEOL SEM metadata .txt file.
    Returns dict with 'bar_px' (int) and 'bar_label' (str, e.g. '10 µm').
    Reads $$SM_MICRON_BAR and $$SM_MICRON_MARKER lines.
    """
    import re as _re
    result = {}
    for line in txt_content.splitlines():
        line = line.strip()
        if line.startswith("$$SM_MICRON_BAR"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    result["bar_px"] = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith("$$SM_MICRON_MARKER"):
            parts = line.split()
            if len(parts) >= 2:
                raw = parts[1]  # e.g. "10um", "1um", "500nm"
                m = _re.match(r"([\d.]+)\s*(um|µm|nm|mm)", raw, _re.IGNORECASE)
                if m:
                    val_str = m.group(1)
                    unit = m.group(2).lower()
                    if unit == "um":
                        unit = "µm"
                    if "." in val_str:
                        val_str = val_str.rstrip("0").rstrip(".")
                    result["bar_label"] = f"{val_str} {unit}"
    return result


def extract_eds_maps_from_docx(docx_bytes: bytes) -> dict:
    """
    Extract images from an Oxford EDS docx.
    Excludes tiny images (logos etc.) by minimum size threshold.
    Returns all images with both dimensions >= 150px.
    key -> PIL Image (RGBA preserved)
    """
    import zipfile as _zf
    all_imgs = {}
    with _zf.ZipFile(io.BytesIO(docx_bytes)) as z:
        media = sorted([n for n in z.namelist()
                        if n.startswith("word/media/")
                        and n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))])
        for mname in media:
            data = z.read(mname)
            img = Image.open(io.BytesIO(data)).copy()  # force-load into memory
            w, h = img.size
            if w >= 150 and h >= 150:   # exclude logos and tiny icons
                all_imgs[mname] = img
    return all_imgs


def crop_eds_map(img: Image.Image) -> Image.Image:
    """
    Remove the Oxford EDS top element-name bar and transparent padding.
    Returns the EDS map area composited on a black background (RGB).
    The embedded scale bar remains intact inside the returned image.
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb   = arr[:, :, :3]

    row_max_alpha = alpha.max(axis=1)
    row_max_rgb   = rgb.max(axis=(1, 2))
    col_max_alpha = alpha.max(axis=0)

    # Map starts at the first row with alpha AND colored content (rgb > 30)
    map_start = 0
    for y in range(h):
        if row_max_alpha[y] > 0 and row_max_rgb[y] > 30:
            map_start = y
            break

    rows_ok = np.where(row_max_alpha > 0)[0]
    cols_ok = np.where(col_max_alpha > 0)[0]
    if len(rows_ok) == 0 or len(cols_ok) == 0:
        return img.convert("RGB")

    bottom = int(rows_ok[-1]) + 1
    left   = int(cols_ok[0])
    right  = int(cols_ok[-1]) + 1

    cropped = rgba.crop((left, map_start, right, bottom))
    bg = Image.new("RGB", cropped.size, (0, 0, 0))
    bg.paste(cropped, mask=cropped.split()[3])
    return bg


def ocr_eds_element_name(img_rgba: Image.Image) -> str:
    """
    Try to read the element name (e.g. 'Al Kα1') from the top bar
    of an Oxford EDS map.  Returns '' on failure.
    """
    try:
        import pytesseract
    except ImportError:
        return ""

    arr = np.array(img_rgba.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb   = arr[:, :, :3]

    # Top-bar rows: alpha > 0 but nearly black RGB (text rows)
    top_bar_rows = [y for y in range(min(60, h))
                    if alpha[y].max() > 0 and rgb[y].max() < 30]
    if not top_bar_rows:
        return ""

    y0, y1 = min(top_bar_rows), max(top_bar_rows) + 1
    # 元素記号は左端にあるため左半分だけ使用（Kα1などの殻表記を除外）
    bar_w = max(1, w // 2)
    bar = img_rgba.convert("RGBA").crop((0, y0, bar_w, y1))
    bg = Image.new("RGB", bar.size, "white")
    bg.paste(bar.convert("RGB"), mask=bar.split()[3])
    up = bg.resize((bg.width * 6, bg.height * 6), Image.LANCZOS)
    try:
        import re as _re
        cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        raw = pytesseract.image_to_string(up, config=cfg).strip()
        # 先頭が小文字の場合（OCR誤認識）を大文字に補正
        if raw and raw[0].islower():
            raw = raw[0].upper() + raw[1:]

        # 既知の元素記号リスト
        _ELEMENTS = {
            'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
            'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
            'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru',
            'Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce',
            'Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta',
            'W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Th','U',
        }

        m = _re.match(r'([A-Z][a-z]?)', raw)
        if m:
            sym = m.group(1)
            # V/W は M の誤認識が多い → 後続の小文字と組み合わせて確認
            if sym[0] in ('V', 'W'):
                for ch in raw[1:6]:
                    if ch.islower():
                        candidate = 'M' + ch
                        if candidate in _ELEMENTS:
                            return candidate
                # M単体の元素記号も候補（例: Mn, Mg は小文字が取れない場合）
                sym = sym[0]  # V/W → 補正失敗なので V or W として扱う
            if sym in _ELEMENTS:
                return sym
            # 2文字マッチが元素にない場合、1文字だけ試す（例: 'Ok'→'O'）
            if len(sym) == 2 and sym[0] in _ELEMENTS:
                return sym[0]
        return "SEI"
    except Exception:
        return "SEI"


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab2, tab_eds, tab3, tab1 = st.tabs([
    "📏 スケールバー処理",
    "📊 EDSグラフ",
    "🗂️ パネル配置",
    "📁 画像管理",
])


# ══════════════════════════════════════════════
#  TAB EDS — EDS Map Processing
# ══════════════════════════════════════════════
if "eds_extracted" not in st.session_state:
    st.session_state.eds_extracted = {}   # key -> PIL Image (RGBA)
if "eds_ocr_cache" not in st.session_state:
    st.session_state.eds_ocr_cache = {}   # key -> str

with tab_eds:
    st.header("EDSマップ処理")
    st.caption("OxfordのEDSレポート（.docx）をアップロード → 元素マップを自動抽出・即パネル配置に追加")

    docx_files = st.file_uploader(
        "EDSレポート (.docx) をアップロード（複数可）",
        type=["docx"],
        accept_multiple_files=True,
        key="docx_upload",
    )

    if docx_files:
        # 新しいdocxのみ処理（既に抽出済みはスキップ）
        for df in docx_files:
            stem = Path(df.name).stem
            df_bytes = df.read()
            try:
                maps = extract_eds_maps_from_docx(df_bytes)
                for i, (mname, img) in enumerate(maps.items()):
                    ext = Path(mname).suffix
                    key = f"{stem}_map{i+1}{ext}"
                    if key not in st.session_state.eds_extracted:
                        st.session_state.eds_extracted[key] = img
                        # OCR を即時実行してキャッシュ
                        st.session_state.eds_ocr_cache[key] = ocr_eds_element_name(img)
                        # クロップ済み画像をパネル配置に即追加
                        cropped = crop_eds_map(img)
                        st.session_state.images[key] = cropped
            except Exception as e:
                st.warning(f"{df.name}: 読み込みエラー ({e})")

    if st.session_state.eds_extracted:
        n_maps = len(st.session_state.eds_extracted)
        st.success(f"{n_maps} 枚を抽出済み（パネル配置に自動追加済み）")

        if st.button("EDS画像をクリア", key="eds_clear"):
            for k in list(st.session_state.eds_extracted.keys()):
                st.session_state.images.pop(k, None)
            st.session_state.eds_extracted = {}
            st.session_state.eds_ocr_cache = {}
            st.rerun()

        st.divider()

        # ── 共通設定 ─────────────────────────────
        with st.expander("ラベル設定"):
            eds_label_fs    = st.slider("元素ラベル フォントサイズ", 8, 120, 80, key="eds_lfs")
            eds_label_color = st.selectbox("元素ラベル 色", ["white", "black", "yellow"], key="eds_lcol")

        st.subheader("各マップの元素名設定")
        st.caption("元素名を確認・編集 → 「ラベルを適用」で図内の左上テキストを更新します")

        hc = st.columns([2, 2, 1])
        hc[0].caption("プレビュー")
        hc[1].caption("元素名（左上に表示）")
        hc[2].caption("除外")

        eds_entries = []
        for i, (key, img_rgba) in enumerate(st.session_state.eds_extracted.items()):
            auto_name = st.session_state.eds_ocr_cache.get(key, "")
            wkey = f"eds_elem_{key}"
            # OCR結果をセッションに先書き（初回のみ）
            if wkey not in st.session_state and auto_name:
                st.session_state[wkey] = auto_name
            c0, c1, c2 = st.columns([2, 2, 1])
            c0.image(st.session_state.images.get(key, crop_eds_map(img_rgba)), use_column_width=True)
            elem = c1.text_input("元素名", key=wkey,
                                  label_visibility="collapsed", placeholder="例: Al Kα1")
            remove = c2.checkbox("除外", value=False, key=f"eds_rm_{key}")
            eds_entries.append((key, img_rgba, elem, remove))

        if st.button("🔤 ラベルを適用・更新", type="primary", use_container_width=True):
            for key, img_rgba, elem, remove in eds_entries:
                if remove:
                    st.session_state.images.pop(key, None)
                    continue
                result = crop_eds_map(img_rgba)
                if elem.strip():
                    _draw = ImageDraw.Draw(result)
                    _font = load_font(eds_label_fs, bold=True)
                    _draw.text((8, 4), elem.strip(), fill=eds_label_color, font=_font)
                st.session_state.images[key] = result
            st.success("更新しました → 「パネル配置」タブで確認できます")
            st.rerun()

# ══════════════════════════════════════════════
#  TAB 1 — Image Management
# ══════════════════════════════════════════════
with tab1:
    st.header("画像管理")

    uploaded = st.file_uploader(
        "SEM画像をアップロード（複数可） — TIFF / PNG / JPEG / BMP",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded:
        for f in uploaded:
            img = Image.open(f)
            st.session_state.images[f.name] = img
        st.success(f"{len(st.session_state.images)} 枚の画像を保持中")

    if st.session_state.images:
        if st.button("全画像をクリア"):
            st.session_state.images = {}
            st.rerun()

        n = len(st.session_state.images)
        ncols = min(n, 4)
        cols = st.columns(ncols)
        for i, (name, img) in enumerate(st.session_state.images.items()):
            with cols[i % ncols]:
                st.image(img, caption=name, use_column_width=True)
                w, h = img.size
                st.caption(f"{w} × {h} px | {img.mode}")
    else:
        st.info("上のアップローダーからSEM画像を追加してください。")


# ══════════════════════════════════════════════
#  TAB 2 — Scale Bar
# ══════════════════════════════════════════════
with tab2:
    st.header("スケールバー処理")

    # ── 画像アップロード ───────────────────────────
    fs2 = st.file_uploader(
        "SEM画像をアップロード（複数可）",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        key="sb_upload", accept_multiple_files=True,
    )
    work_imgs = [(f.name, Image.open(f).copy()) for f in fs2] if fs2 else []

    # ── JEOL .txt（複数可）→ ファイル名でマッチング ──
    txt_files = st.file_uploader(
        "JEOL メタデータ (.txt) — 画像と同名のものをまとめてアップロード（任意）",
        type=["txt"], key="jeol_txts", accept_multiple_files=True,
    )
    # stem → {bar_px, bar_label}
    jeol_metas: dict = {}
    for tf in (txt_files or []):
        meta = parse_jeol_txt(tf.read().decode("utf-8", errors="ignore"))
        if meta:
            jeol_metas[Path(tf.name).stem] = meta

    if work_imgs:
        st.divider()

        # ── ファイルごとのスケール入力 ────────────────
        st.subheader("スケールラベル設定")
        st.caption("各画像のスケール数値を入力してください")

        # ヘッダー行
        h1, h2, h3 = st.columns([3, 1, 1])
        h1.markdown("**ファイル名**")
        h2.markdown("**数値**")
        h3.markdown("**単位**")

        scale_entries = []  # (img_name, img, label_val, label_unit, bar_px_from_txt)
        import re as _re_tbl
        for i, (img_name, img) in enumerate(work_imgs):
            stem = Path(img_name).stem
            meta = jeol_metas.get(stem, {})
            # txt からデフォルト値を取得
            _def_val, _def_unit_idx, _bar_px = "", 0, None
            if meta.get("bar_label"):
                _m = _re_tbl.match(r"([\d.]+)\s*(µm|um|nm)", meta["bar_label"], _re_tbl.IGNORECASE)
                if _m:
                    _def_val = _m.group(1)
                    _def_unit_idx = 1 if _m.group(2).lower() == "nm" else 0
            if meta.get("bar_px"):
                _bar_px = meta["bar_px"]

            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            c1.text(stem)
            val  = c2.text_input("数値", key=f"lval_{i}",  label_visibility="collapsed",
                                  value=_def_val, placeholder="10")
            unit = c3.selectbox("単位", ["µm", "nm"], key=f"lunit_{i}",
                                 label_visibility="collapsed", index=_def_unit_idx)
            if _bar_px:
                c4.caption(f"bar={_bar_px}px")
            scale_entries.append((img_name, img, val, unit, _bar_px))

        st.divider()

        # ── 共通設定（詳細） ──────────────────────────
        with st.expander("詳細設定"):
            databar_mode = st.radio(
                "データバー（下部黒帯）",
                ["手動でクロップ高さ指定", "自動検出してクロップ", "クロップしない"],
                index=0,
            )
            manual_crop_h = 86
            if databar_mode == "手動でクロップ高さ指定":
                manual_crop_h = st.slider("クロップする高さ (px)", 10, 300, 86)
            extract_orig = st.checkbox("装置のスケールバーを抽出して再配置する", value=True)
            bar_thickness = st.slider("スケールバーの太さ (px)", 1, 20, 12)
            bar_fsize = st.slider("ラベルフォントサイズ", 12, 120, 70)
            add_matname = st.checkbox("材料名ラベルを追加", value=False)
            mat_name, use_subscript, mat_fsize = "", True, 40
            if add_matname:
                mat_name = st.text_input("材料名 (例: Ti3C2Tx)", value="Ti3C2Tx")
                use_subscript = st.checkbox("数字・x を下付き文字に", value=True)
                mat_fsize = st.slider("材料名フォントサイズ", 12, 100, 40)

        st.caption(f"{len(work_imgs)} 枚を一括処理します")
        do_process = st.button("✅ 一括処理を実行", type="primary", use_container_width=True)

        if do_process:
            results = []
            for img_name, img, lval, lunit, txt_bar_px in scale_entries:
                scale_label = f"{lval} {lunit}" if lval.strip() else ""
                iw, ih = img.size

                # 1. Detect databar
                if databar_mode == "自動検出してクロップ":
                    dy = detect_sem_databar(img)
                elif databar_mode == "手動でクロップ高さ指定":
                    dy = ih - manual_crop_h
                else:
                    dy = ih
                result = img.convert("RGB").crop((0, 0, iw, dy))

                # 2. Determine bar width (txt優先 → 画像検出フォールバック)
                search_from = dy if dy < ih else int(ih * 0.85)
                if extract_orig:
                    bw = None
                    if txt_bar_px:
                        bw = txt_bar_px   # JEOL txt から正確な値を使用
                    else:
                        br = detect_scale_bar_in_region(img, search_from)
                        if br:
                            bw = br[2] - br[0]
                    if bw:
                        rw, rh = result.size
                        # 参照コード準拠: 右から60px・下から45px
                        px_ = rw - bw - 60
                        py_ = rh - 45
                        # 元バーはペーストせず固定厚さの白矩形を描画（全画像で統一）
                        _d = ImageDraw.Draw(result)
                        _d.rectangle([px_, py_, px_ + bw, py_ + bar_thickness], fill="white")
                        if scale_label:
                            _f = load_font(bar_fsize)
                            _bb = _d.textbbox((0, 0), scale_label, font=_f)
                            _lw = _bb[2] - _bb[0]
                            _lh = _bb[3] - _bb[1]
                            # ラベルはバー上端から15px上、中央揃え
                            _d.text((px_ + bw // 2 - _lw // 2, py_ - _lh - 15),
                                    scale_label, fill="white", font=_f)

                # 3. Material name
                if add_matname and mat_name:
                    _d2 = ImageDraw.Draw(result)
                    draw_material_name(_d2, 20, 20, mat_name,
                                       load_font(mat_fsize, bold=True),
                                       load_font(int(mat_fsize * 0.65)),
                                       use_subscript)

                save_name = f"{Path(img_name).stem}_processed.png"
                st.session_state.images[save_name] = result.copy()
                results.append((save_name, result))

            st.session_state.sb_results = results

        if "sb_results" in st.session_state and st.session_state.sb_results:
            results = st.session_state.sb_results
            st.divider()
            st.subheader("処理結果")
            n = len(results)
            grid = st.columns(min(n, 3))
            for i, (name, img) in enumerate(results):
                with grid[i % min(n, 3)]:
                    st.image(img, caption=name, use_column_width=True)

            st.info(f"📂 {n} 枚を「画像管理」に保存しました。「パネル配置」タブで選択できます。")

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, img in results:
                    zf.writestr(name, pil_to_bytes(img, "PNG", 300))
            zip_buf.seek(0)
            st.download_button(
                "💾 全画像を ZIP でダウンロード",
                data=zip_buf.read(),
                file_name="sem_processed.zip",
                mime="application/zip",
            )


# ══════════════════════════════════════════════
#  TAB 3 — Panel Arrangement
# ══════════════════════════════════════════════
with tab3:
    st.header("パネル配置（マルチパネルFigure作成）")

    col_ctrl3, col_prev3 = st.columns([1, 1.4], gap="large")

    with col_ctrl3:
        st.subheader("画像の選択と順序")

        panel_images = []
        panel_names = []

        if st.session_state.images:
            avail = list(st.session_state.images.keys())
            processed = [n for n in avail if "_processed" in n]
            others    = [n for n in avail if "_processed" not in n]

            # ── チェックボックスで選択 ────────────────
            st.caption("使用する画像にチェック")
            checked = {}
            if processed:
                st.markdown("**加工済み**")
                for name in processed:
                    checked[name] = st.checkbox(Path(name).stem, value=True, key=f"chk_{name}")
            if others:
                st.markdown("**その他**")
                for name in others:
                    checked[name] = st.checkbox(Path(name).stem, value=True, key=f"chk_{name}")

            selected = [n for n in avail if checked.get(n)]

            # ── 順序・サンプル名テーブル ──────────────
            if selected:
                if ("panel_order" not in st.session_state
                        or set(st.session_state.panel_order) != set(selected)):
                    st.session_state.panel_order = selected.copy()

                order = [n for n in st.session_state.panel_order if n in selected]
                for n in selected:
                    if n not in order:
                        order.append(n)

                st.divider()
                st.caption("順序を変更し、サンプル名を入力してください")

                # ヘッダー
                hc = st.columns([0.5, 1.8, 2.5, 0.4, 0.4])
                hc[0].caption("順位")
                hc[1].caption("プレビュー")
                hc[2].caption("サンプル名")

                reordered = False
                for i, name in enumerate(order):
                    thumb = st.session_state.images[name].copy()
                    thumb.thumbnail((100, 100))
                    # EDS元素名があれば優先、なければファイル名から生成
                    eds_elem = st.session_state.get(f"eds_elem_{name}", "").strip()
                    default_sname = eds_elem if eds_elem else Path(name).stem.replace("_processed", "")
                    # サンプル名をセッションに先書き（初回のみ）
                    if f"sname_{name}" not in st.session_state:
                        st.session_state[f"sname_{name}"] = default_sname

                    c0, c1, c2, c3, c4 = st.columns([0.5, 1.8, 2.5, 0.4, 0.4])
                    c0.markdown(f"**{i+1}**")
                    c1.image(thumb, use_column_width=True)
                    c2.text_input(
                        "サンプル名",
                        key=f"sname_{name}",
                        value=st.session_state.get(f"sname_{name}", default_sname),
                        label_visibility="collapsed",
                    )
                    if i > 0 and c3.button("↑", key=f"up_{i}"):
                        order[i], order[i-1] = order[i-1], order[i]
                        reordered = True
                    if i < len(order)-1 and c4.button("↓", key=f"dn_{i}"):
                        order[i], order[i+1] = order[i+1], order[i]
                        reordered = True

                st.session_state.panel_order = order
                if reordered:
                    st.rerun()

                for n in order:
                    panel_images.append(st.session_state.images[n].convert("RGB"))
                    panel_names.append(n)
        else:
            st.info("先に「スケールバー処理」タブで画像を処理してください。")

        if panel_images:
            n_imgs = len(panel_images)
            st.caption(f"{n_imgs} 枚選択中")

            st.divider()
            st.subheader("レイアウト設定")

            layout_mode = st.radio("モード", ["グリッド", "混在 (1枚 + グリッド)"], horizontal=True)

            if layout_mode == "グリッド":
                LAYOUTS = {
                    "1×1": (1,1), "1×2": (1,2), "2×1": (2,1),
                    "1×3": (1,3), "3×1": (3,1), "2×2": (2,2),
                    "2×3": (2,3), "3×2": (3,2), "3×3": (3,3),
                    "1×4": (1,4), "4×1": (4,1), "2×4": (2,4), "4×2": (4,2),
                }
                layout_key = st.selectbox("グリッド (行 × 列)", list(LAYOUTS.keys()), index=5)
                rows, cols = LAYOUTS[layout_key]
            else:
                st.caption("1枚目 = 左の大きい画像 / 残り = 右のグリッド")
                mix_right_rows = st.selectbox("右グリッド 行数", [1, 2, 3, 4], index=1)
                mix_right_cols = st.selectbox("右グリッド 列数", [1, 2, 3, 4], index=1)
                mix_left_pct  = st.slider("左の幅 (%)", 20, 80, 50)

            st.subheader("パネルサイズ")
            uniform_size = st.checkbox("全画像を正方形にリサイズ", value=False)
            panel_px = st.slider("各パネルの辺 (px)", 200, 1200, 500, step=50)

            st.subheader("余白・背景")
            spacing = st.slider("パネル間隔 (px)", 0, 80, 8)
            bg_choice = st.selectbox("背景色", ["白 (white)", "黒 (black)", "グレー (gray)"])
            bg_val = bg_choice.split("(")[1].rstrip(")")

            st.subheader("パネルラベル")
            lbl_content = st.radio(
                "ラベル内容",
                ["サンプル名のみ", "文字のみ (a)(b)...", "両方"],
                index=0, horizontal=True,
            )
            label_style = "なし"
            if lbl_content in ("文字のみ (a)(b)...", "両方"):
                label_style = st.selectbox(
                    "文字スタイル",
                    ["(a), (b), (c)...", "(A), (B), (C)...", "a), b), c)...", "A), B), C)...", "1, 2, 3..."],
                    index=0,
                )
            c1, c2, c3 = st.columns(3)
            lbl_color = c1.selectbox("ラベル色", ["白 (white)", "黒 (black)"], key="lbl_col")
            lbl_color_val = lbl_color.split("(")[1].rstrip(")")
            lbl_pos = c2.selectbox("ラベル位置", ["左上", "右上", "左下", "右下"])
            lbl_fs = c3.slider("フォントサイズ", 12, 150, 80, key="panel_lfs")

            # ── ラベルテキスト生成 ──────────────────────
            def build_labels(panel_names_list):
                base_labels = make_label_list(label_style, len(panel_names_list)) if label_style != "なし" else [""] * len(panel_names_list)
                result = []
                for idx, pname in enumerate(panel_names_list):
                    base = base_labels[idx] if idx < len(base_labels) else ""
                    sname = st.session_state.get(f"sname_{pname}", "").strip()
                    if lbl_content == "サンプル名のみ":
                        result.append(sname)
                    elif lbl_content == "文字のみ (a)(b)...":
                        result.append(base)
                    else:  # 両方
                        if base and sname:
                            result.append(f"{base} {sname}")
                        else:
                            result.append(base or sname)
                return result

            def paste_with_label(canvas, draw, img, pw, ph, x0, y0, lbl, font_lbl):
                resized = img.resize((pw, ph), Image.LANCZOS)
                canvas.paste(resized, (x0, y0))
                if lbl:
                    lm = 8
                    bbox = draw.textbbox((0, 0), lbl, font=font_lbl)
                    lw, lh = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    if lbl_pos == "左上":
                        lx, ly = x0 + lm, y0 + lm
                    elif lbl_pos == "右上":
                        lx, ly = x0 + pw - lw - lm, y0 + lm
                    elif lbl_pos == "左下":
                        lx, ly = x0 + lm, y0 + ph - lh - lm
                    else:
                        lx, ly = x0 + pw - lw - lm, y0 + ph - lh - lm
                    draw.text((lx, ly), lbl, fill=lbl_color_val, font=font_lbl)

            st.divider()
            if st.button("📐 パネルを作成", type="primary", use_container_width=True):
                font_lbl = load_font(lbl_fs, bold=True)

                def img_size(img):
                    ow, oh = img.size
                    if uniform_size:
                        return panel_px, panel_px
                    aspect = ow / oh
                    if aspect >= 1:
                        return panel_px, int(panel_px / aspect)
                    return int(panel_px * aspect), panel_px

                if layout_mode == "グリッド":
                    labels = build_labels(panel_names)
                    panel_sizes = [img_size(img) for img in panel_images]
                    col_w = max(s[0] for s in panel_sizes)
                    row_h = max(s[1] for s in panel_sizes)
                    canvas_w = cols * col_w + (cols - 1) * spacing
                    canvas_h = rows * row_h + (rows - 1) * spacing
                    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_val)
                    draw = ImageDraw.Draw(canvas)
                    for idx, (img, (pw, ph)) in enumerate(zip(panel_images, panel_sizes)):
                        if idx >= rows * cols:
                            break
                        r, c = idx // cols, idx % cols
                        x0 = c * (col_w + spacing) + (col_w - pw) // 2
                        y0 = r * (row_h + spacing) + (row_h - ph) // 2
                        paste_with_label(canvas, draw, img, pw, ph, x0, y0,
                                         labels[idx] if idx < len(labels) else "", font_lbl)

                else:  # 混在レイアウト
                    labels = build_labels(panel_names)
                    main_img = panel_images[0]
                    small_imgs = panel_images[1:]
                    small_names = panel_names[1:]
                    small_labels = labels[1:]

                    total_w = panel_px * 3  # 全体幅の基準
                    left_w  = int(total_w * mix_left_pct / 100)
                    right_w = total_w - left_w - spacing

                    # 左: メイン画像をleft_w幅に収める
                    mw, mh = main_img.size
                    main_ph = int(left_w * mh / mw)
                    total_h = main_ph

                    # 右: グリッド
                    n_small = mix_right_rows * mix_right_cols
                    cell_w = (right_w - (mix_right_cols - 1) * spacing) // mix_right_cols
                    cell_h = (total_h - (mix_right_rows - 1) * spacing) // mix_right_rows

                    canvas = Image.new("RGB", (total_w, total_h), bg_val)
                    draw = ImageDraw.Draw(canvas)

                    # 左パネル
                    paste_with_label(canvas, draw, main_img, left_w, main_ph,
                                     0, 0, labels[0] if labels else "", font_lbl)

                    # 右グリッド
                    for si, (simg, slbl) in enumerate(zip(small_imgs[:n_small], small_labels[:n_small])):
                        sr, sc = si // mix_right_cols, si % mix_right_cols
                        sx = left_w + spacing + sc * (cell_w + spacing)
                        sy = sr * (cell_h + spacing)
                        # セルに収まるよう縮小
                        sw_orig, sh_orig = simg.size
                        scale = min(cell_w / sw_orig, cell_h / sh_orig)
                        sw = int(sw_orig * scale)
                        sh = int(sh_orig * scale)
                        ox = (cell_w - sw) // 2
                        oy = (cell_h - sh) // 2
                        paste_with_label(canvas, draw, simg, sw, sh,
                                         sx + ox, sy + oy, slbl, font_lbl)

                st.session_state.panel_result = canvas

    with col_prev3:
        st.subheader("プレビュー")
        if st.session_state.panel_result:
            pres = st.session_state.panel_result
            st.image(pres, caption="パネル配置結果", use_column_width=True)
            pw, ph = pres.size
            st.caption(f"出力サイズ: {pw} × {ph} px")

            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "💾 PNG (300 dpi)",
                data=pil_to_bytes(pres, "PNG", 300),
                file_name="panel_figure.png",
                mime="image/png",
            )
            c2.download_button(
                "💾 TIFF (300 dpi)",
                data=pil_to_bytes(pres, "TIFF", 300),
                file_name="panel_figure_300dpi.tif",
                mime="image/tiff",
            )
            c3.download_button(
                "💾 TIFF (600 dpi)",
                data=pil_to_bytes(pres, "TIFF", 600),
                file_name="panel_figure_600dpi.tif",
                mime="image/tiff",
            )

