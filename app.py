from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="ROI Studio", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# ----------------------------
# Professional UI
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#070a0f;--panel:#0f141c;--panel2:#141b24;--line:#25303d;--text:#f5f7fb;--muted:#8e9aaa;--blue:#64a9ff;--mint:#65e4c0;}
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:transparent}#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden;height:0}
.block-container{max-width:1520px;padding-top:.8rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:#0c1118;border-right:1px solid var(--line)}
.brandbar{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:4px 0 14px;margin-bottom:16px}
.brand-logo{width:310px;max-width:62vw;height:auto;display:block}.pro-badge{border:1px solid #30445d;background:#101b29;color:#acd2ff;padding:7px 11px;border-radius:999px;font-size:10px;font-weight:800;letter-spacing:.7px}
.hero{background:linear-gradient(135deg,#101821,#0d131b);border:1px solid var(--line);border-radius:18px;padding:17px 20px;margin-bottom:14px}.hero h1{margin:0;font-size:26px;letter-spacing:-.8px}.hero p{margin:5px 0 0;color:var(--muted);font-size:12px}
.section{color:#94a0af;text-transform:uppercase;letter-spacing:1.3px;font-size:10px;font-weight:800;margin:6px 0 9px}.hint{color:var(--muted);font-size:11px;line-height:1.55}
.roi-card{background:var(--panel2);border:1px solid #293645;border-radius:14px;padding:11px 12px;margin-bottom:8px}
.roi-num{width:28px;height:28px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800}
div.stButton>button{border-radius:10px;border:1px solid #2d3948;background:#151d27;color:#fff;font-weight:700;min-height:40px}div.stButton>button:hover{border-color:#5b7899;background:#182331}
div.stDownloadButton>button{width:100%;border-radius:11px;border:0;background:linear-gradient(135deg,#5faaff,#7b6cff);color:#fff;font-weight:800;min-height:44px}
[data-testid="stFileUploaderDropzone"]{background:#0e151e;border:1px dashed #3a4858;border-radius:13px}
[data-baseweb="tab-list"]{gap:7px}button[data-baseweb="tab"]{color:#8e9aaa!important;border-radius:9px}button[data-baseweb="tab"][aria-selected="true"]{color:#fff!important;background:#171f29!important}
[data-testid="stImage"] img{border-radius:12px}.stMetric{background:#111820;border:1px solid var(--line);border-radius:12px;padding:10px}
.footer-clean{border-top:1px solid var(--line);margin-top:24px;padding-top:12px;color:#697586;font-size:10px;display:flex;justify-content:space-between}
</style>
""", unsafe_allow_html=True)

FILTERS = ["Original", "Gaussian Blur", "Median Filter", "Sharpening", "Edge Detection", "Grayscale", "Detail Boost", "Soft Glow"]
COLORS = ["#65e4c0", "#64a9ff", "#ff718a", "#b58cff", "#ffd166", "#ff9f68"]


def logo_path() -> str:
    return os.path.join(os.path.dirname(__file__), "assets", "roi_studio_logo.png")


def rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def preview(img: Image.Image, max_width: int = 1080) -> Image.Image:
    out = img.copy()
    if out.width > max_width:
        out.thumbnail((max_width, 10000), Image.Resampling.LANCZOS)
    return out


def roi_filter(crop: np.ndarray, name: str) -> np.ndarray:
    """Filter a crop only. Never receives the full image."""
    if crop.size == 0 or name == "Original":
        return crop.copy()
    if name == "Gaussian Blur":
        return cv2.GaussianBlur(crop, (11, 11), 0)
    if name == "Median Filter":
        m = min(crop.shape[:2])
        k = min(7, m if m % 2 else m - 1)
        return crop.copy() if k < 3 else cv2.medianBlur(crop, k)
    if name == "Sharpening":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        return cv2.filter2D(crop, -1, kernel)
    if name == "Edge Detection":
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if name == "Grayscale":
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if name == "Detail Boost":
        blur = cv2.GaussianBlur(crop, (0,0), 2)
        return cv2.addWeighted(crop, 1.45, blur, -0.45, 0)
    if name == "Soft Glow":
        blur = cv2.GaussianBlur(crop, (0,0), 7)
        return cv2.addWeighted(crop, .72, blur, .28, 0)
    return crop.copy()


def clamp_box(x1: float,y1: float,x2: float,y2: float,w: int,h: int):
    xa,xb=sorted((int(round(x1)),int(round(x2)))); ya,yb=sorted((int(round(y1)),int(round(y2))))
    return max(0,min(w,xa)),max(0,min(h,ya)),max(0,min(w,xb)),max(0,min(h,yb))


def rect_box(obj: dict[str,Any]):
    left=float(obj.get("left",0)); top=float(obj.get("top",0))
    width=float(obj.get("width",0))*abs(float(obj.get("scaleX",1)))
    height=float(obj.get("height",0))*abs(float(obj.get("scaleY",1)))
    return left,top,left+width,top+height


def circle_box(obj: dict[str,Any]):
    left=float(obj.get("left",0)); top=float(obj.get("top",0))
    if "radius" in obj:
        rx=float(obj.get("radius",0))*abs(float(obj.get("scaleX",1)))
        ry=float(obj.get("radius",0))*abs(float(obj.get("scaleY",1)))
    else:
        rx=float(obj.get("width",0))*abs(float(obj.get("scaleX",1)))/2
        ry=float(obj.get("height",0))*abs(float(obj.get("scaleY",1)))/2
    return left,top,left+2*rx,top+2*ry


def path_points(obj: dict[str,Any]) -> list[tuple[float,float]]:
    path=obj.get("path") or []
    left=float(obj.get("left",0)); top=float(obj.get("top",0))
    pts=[]
    for cmd in path:
        if not isinstance(cmd,(list,tuple)) or len(cmd)<3: continue
        vals=[v for v in cmd[1:] if isinstance(v,(int,float))]
        c=str(cmd[0]).upper()
        if c in {"M","L"} and len(vals)>=2: pts.append((left+vals[0],top+vals[1]))
        elif c in {"Q","C"} and len(vals)>=2: pts.append((left+vals[-2],top+vals[-1]))
    return pts


def extract_shapes(json_data: dict[str,Any]|None, display_w:int, display_h:int, image_w:int, image_h:int):
    if not json_data: return []
    sx=image_w/float(display_w); sy=image_h/float(display_h)
    result=[]
    for obj in json_data.get("objects",[]):
        typ=str(obj.get("type","")).lower()
        if typ=="rect":
            a,b,c,d=rect_box(obj)
            x1,y1,x2,y2=a*sx,b*sy,c*sx,d*sy
            if x2-x1>=4 and y2-y1>=4: result.append({"shape":"rectangle","x1":x1,"y1":y1,"x2":x2,"y2":y2,"filter":"Original"})
        elif typ=="circle":
            a,b,c,d=circle_box(obj)
            x1,y1,x2,y2=a*sx,b*sy,c*sx,d*sy
            if x2-x1>=4 and y2-y1>=4: result.append({"shape":"circle","x1":x1,"y1":y1,"x2":x2,"y2":y2,"filter":"Original"})
        elif typ=="path":
            pts=path_points(obj)
            if len(pts)>=3:
                result.append({"shape":"freehand","points":[(x*sx,y*sy) for x,y in pts],"filter":"Original"})
    return result


def shape_mask(shape:dict[str,Any], w:int,h:int)->np.ndarray:
    mask=np.zeros((h,w),np.uint8); kind=shape.get("shape","rectangle")
    if kind in {"rectangle","circle"}:
        x1,y1,x2,y2=clamp_box(shape["x1"],shape["y1"],shape["x2"],shape["y2"],w,h)
        if x2<=x1 or y2<=y1: return mask
        if kind=="rectangle": cv2.rectangle(mask,(x1,y1),(x2-1,y2-1),255,-1)
        else:
            cx=(x1+x2)//2; cy=(y1+y2)//2; rx=max(1,(x2-x1)//2); ry=max(1,(y2-y1)//2)
            cv2.ellipse(mask,(cx,cy),(rx,ry),0,0,360,255,-1)
    elif kind=="freehand":
        pts=np.array(shape.get("points",[]),dtype=np.int32)
        if len(pts)>=3:
            pts[:,0]=np.clip(pts[:,0],0,w-1); pts[:,1]=np.clip(pts[:,1],0,h-1)
            cv2.fillPoly(mask,[pts],255)
    return mask


def apply_roi(base:np.ndarray, roi:dict[str,Any])->np.ndarray:
    """Apply one filter only where the ROI mask is non-zero."""
    name=roi.get("filter","Original")
    if name in (None,"Original"): return base
    h,w=base.shape[:2]; mask=shape_mask(roi,w,h)
    ys,xs=np.where(mask>0)
    if len(xs)==0: return base
    x1,x2=int(xs.min()),int(xs.max())+1; y1,y2=int(ys.min()),int(ys.max())+1
    crop=base[y1:y2,x1:x2].copy(); filtered=roi_filter(crop,name)
    local=mask[y1:y2,x1:x2]>0; out=base.copy(); out_crop=out[y1:y2,x1:x2]
    out_crop[local]=filtered[local]; out[y1:y2,x1:x2]=out_crop
    return out


def render_result(original:np.ndarray,rois:list[dict[str,Any]])->np.ndarray:
    out=original.copy()
    for roi in rois: out=apply_roi(out,roi)
    return out


def bbox(roi):
    if roi.get("shape")=="freehand":
        pts=roi.get("points",[])
        if not pts:return 0,0,0,0
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        return int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
    return int(min(roi["x1"],roi["x2"])),int(min(roi["y1"],roi["y2"])),int(max(roi["x1"],roi["x2"])),int(max(roi["y1"],roi["y2"]))


def overlay(image:np.ndarray,rois:list[dict[str,Any]])->Image.Image:
    p=preview(pil_rgb(image),1100); sx=p.width/image.shape[1]; sy=p.height/image.shape[0]; d=ImageDraw.Draw(p)
    for i,r in enumerate(rois):
        color=COLORS[i%len(COLORS)]; kind=r.get("shape","rectangle"); label=f"ROI {i+1} · {r.get('filter','Original')}"
        if kind=="freehand":
            pts=[(int(x*sx),int(y*sy)) for x,y in r.get("points",[])]
            if len(pts)>=2:d.line(pts+[pts[0]],fill=color,width=3); d.text((pts[0][0]+5,pts[0][1]+5),label,fill=color)
        else:
            x1,y1,x2,y2=bbox(r); box=(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy))
            if kind=="circle":d.ellipse(box,outline=color,width=3)
            else:d.rectangle(box,outline=color,width=3)
            d.text((box[0]+6,box[1]+5),label,fill=color)
    return p


def png_bytes(image:np.ndarray)->bytes:
    b=io.BytesIO(); pil_rgb(image).save(b,format="PNG",optimize=True); return b.getvalue()

# ----------------------------
# State
# ----------------------------
st.session_state.setdefault("rois",[])
st.session_state.setdefault("canvas_version",0)
st.session_state.setdefault("image_key",None)

# ----------------------------
# Header with generated logo
# ----------------------------
if os.path.exists(logo_path()):
    logo_b64 = __import__("base64").b64encode(open(logo_path(),"rb").read()).decode("ascii")
    logo_html=f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='ROI Studio'>"
else:
    logo_html="<div style='font-size:25px;font-weight:800'>ROI Studio</div>"

st.markdown(f"<div class='brandbar'><div>{logo_html}</div><div class='pro-badge'>PRO EDITOR</div></div>",unsafe_allow_html=True)

upload=st.file_uploader("Import image",type=["jpg","jpeg","png","webp","bmp","tif","tiff"],label_visibility="collapsed")
if upload is None:
    st.markdown("<div class='hero'><h1>Precision editing, region by region.</h1><p>Upload an image, draw a shape, assign a filter, and only that selected region changes.</p></div>",unsafe_allow_html=True)
    st.info("Start by uploading an image above.")
    st.stop()

# Reset ROI state for a newly uploaded image.
raw=upload.getvalue(); image_key=f"{upload.name}:{len(raw)}"
if st.session_state.image_key!=image_key:
    st.session_state.image_key=image_key; st.session_state.rois=[]; st.session_state.canvas_version+=1

original=rgb_array(Image.open(io.BytesIO(raw)))
img_h,img_w=original.shape[:2]

with st.sidebar:
    st.markdown("### ROI Editor")
    st.markdown("<div class='hint'>Filters in this app are ROI-only. The original image is never filtered outside a selected region.</div>",unsafe_allow_html=True)
    st.divider()
    st.markdown("### Quick actions")
    if st.button("Clear all regions",use_container_width=True):
        st.session_state.rois=[]; st.session_state.canvas_version+=1; st.rerun()
    st.caption("Tip: draw a region first, then press Add selected regions.")

result=render_result(original,st.session_state.rois)

st.markdown(f"<div class='hero'><h1>{upload.name}</h1><p>{img_w:,} × {img_h:,} px · {len(st.session_state.rois)} ROI region(s) · edits stay inside masks</p></div>",unsafe_allow_html=True)

m1,m2,m3=st.columns(3); m1.metric("Resolution",f"{img_w:,} × {img_h:,}"); m2.metric("ROI regions",len(st.session_state.rois)); m3.metric("Filtered regions",sum(r.get("filter") not in (None,"Original") for r in st.session_state.rois))

edit,compare,manager,export=st.tabs(["✦ EDIT","◐ BEFORE / AFTER","⌖ ROI MANAGER","↓ EXPORT"])

with edit:
    left,right=st.columns([2.15,1],gap="large")
    with left:
        st.markdown("<div class='section'>1 · Choose ROI shape</div>",unsafe_allow_html=True)
        shape_mode=st.selectbox("ROI shape",["Rectangle","Circle / Ellipse","Freehand"],key="shape_mode",label_visibility="collapsed")
        mode={"Rectangle":"rect","Circle / Ellipse":"circle","Freehand":"freedraw"}[shape_mode]
        bg=preview(pil_rgb(original),1080)
        canvas=st_canvas(fill_color="rgba(100,169,255,.18)",stroke_width=2,stroke_color="#65e4c0",background_image=bg,update_streamlit=True,height=bg.height,width=bg.width,drawing_mode=mode,key=f"roi_canvas_{st.session_state.canvas_version}")
        objects=(canvas.json_data or {}).get("objects",[]) if canvas else []
        detected=extract_shapes({"objects":objects},bg.width,bg.height,img_w,img_h)
        st.caption(f"Detected on canvas: **{len(detected)}** shape(s). Draw → Add selected regions.")
        a,b,c=st.columns([1.45,1,1])
        if a.button("＋ Add selected regions",type="primary",use_container_width=True):
            existing=set()
            for r in st.session_state.rois:
                if r.get("shape")=="freehand":
                    pts=r.get("points",[]); existing.add(("freehand",len(pts),round(min([p[0] for p in pts],default=0)),round(min([p[1] for p in pts],default=0)),round(max([p[0] for p in pts],default=0)),round(max([p[1] for p in pts],default=0))))
                else: existing.add((r.get("shape"),round(r.get("x1",0)),round(r.get("y1",0)),round(r.get("x2",0)),round(r.get("y2",0))))
            added=0
            for r in detected:
                if r.get("shape")=="freehand":
                    pts=r.get("points",[]); sig=("freehand",len(pts),round(min([p[0] for p in pts],default=0)),round(min([p[1] for p in pts],default=0)),round(max([p[0] for p in pts],default=0)),round(max([p[1] for p in pts],default=0)))
                else: sig=(r.get("shape"),round(r["x1"]),round(r["y1"]),round(r["x2"]),round(r["y2"]))
                if sig not in existing: st.session_state.rois.append(r); existing.add(sig); added+=1
            if added:
                st.session_state.canvas_version+=1; st.rerun()
            else: st.warning("No new ROI detected. Draw a visible shape inside the image first.")
        if b.button("Remove last ROI",use_container_width=True):
            if st.session_state.rois: st.session_state.rois.pop(); st.session_state.canvas_version+=1; st.rerun()
            else: st.info("No ROI to remove.")
        if c.button("Clear drawing",use_container_width=True):
            st.session_state.rois=[]; st.session_state.canvas_version+=1; st.rerun()

        if st.session_state.rois:
            st.markdown("<div class='section'>2 · Edited preview</div>",unsafe_allow_html=True)
            st.image(overlay(result,st.session_state.rois),use_container_width=True)

    with right:
        st.markdown("<div class='section'>3 · ROI filters</div>",unsafe_allow_html=True)
        st.markdown("<div class='hint'>Choose a filter for each region. The filter is mathematically masked to that ROI only.</div>",unsafe_allow_html=True)
        if not st.session_state.rois: st.info("No ROI yet. Draw a shape and click Add selected regions.")
        for i,r in enumerate(st.session_state.rois):
            color=COLORS[i%len(COLORS)]
            with st.container(border=True):
                c1,c2=st.columns([.22,1])
                c1.markdown(f"<div class='roi-num' style='background:{color}22;color:{color}'>{i+1}</div>",unsafe_allow_html=True)
                c2.markdown(f"**ROI {i+1} · {r.get('shape','rectangle').title()}**")
                cur=r.get("filter") or "Original"; selected=st.selectbox("Filter",FILTERS,index=FILTERS.index(cur),key=f"filter_{image_key}_{i}",label_visibility="collapsed"); r["filter"]=selected
                x1,y1,x2,y2=bbox(r); st.caption(f"Bounds: x={x1}, y={y1}, w={max(0,x2-x1)}, h={max(0,y2-y1)} px")
                if st.button("Delete region",key=f"delete_{image_key}_{i}",use_container_width=True): st.session_state.rois.pop(i); st.rerun()

with compare:
    st.markdown("<div class='section'>Visual comparison</div>",unsafe_allow_html=True)
    view=st.radio("View",["Side by side","Original","Edited"],horizontal=True,label_visibility="collapsed")
    if view=="Side by side":
        a,b=st.columns(2); a.caption("ORIGINAL"); a.image(preview(pil_rgb(original)),use_container_width=True); b.caption("EDITED — ROI ONLY"); b.image(preview(pil_rgb(result)),use_container_width=True)
    elif view=="Original": st.image(preview(pil_rgb(original)),use_container_width=True)
    else: st.image(preview(pil_rgb(result)),use_container_width=True)

with manager:
    st.markdown("<div class='section'>ROI manager</div>",unsafe_allow_html=True)
    if not st.session_state.rois: st.info("No ROI regions created yet.")
    else:
        st.image(overlay(result,st.session_state.rois),use_container_width=True)
        for i,r in enumerate(st.session_state.rois,1):
            x1,y1,x2,y2=bbox(r); st.write(f"**ROI {i}** · {r.get('shape','rectangle').title()} · **{r.get('filter','Original')}** · x={x1}, y={y1}, w={max(0,x2-x1)}, h={max(0,y2-y1)}")
        st.caption("If ROIs overlap, later regions are applied after earlier regions.")

with export:
    st.markdown("<div class='section'>Final export</div>",unsafe_allow_html=True)
    a,b=st.columns([1.6,1],gap="large")
    with a: st.image(preview(pil_rgb(result),1150),use_container_width=True)
    with b:
        st.markdown("### Ready to export")
        st.write("Only the selected ROI edits are exported. ROI outlines are not included.")
        st.metric("Output",f"{img_w:,} × {img_h:,} px")
        st.metric("ROI regions",len(st.session_state.rois))
        st.download_button("↓ Download PNG",data=png_bytes(result),file_name=f"roi_studio_{datetime.now():%Y%m%d_%H%M%S}.png",mime="image/png")

st.markdown("<div class='footer-clean'><span>ROI Studio</span><span>Select · Filter · Enhance</span></div>",unsafe_allow_html=True)
