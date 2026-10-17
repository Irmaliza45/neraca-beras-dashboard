"""
Dashboard Neraca Produksi dan Konsumsi Beras
Badan Pusat Statistik Kabupaten Bojonegoro

Jalankan:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import base64
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf

# =====================================================================
# KONSTANTA
# =====================================================================
ROOT = Path(__file__).parent
LOGO = ROOT / "assets" / "logo_bps.png"
DATA = ROOT / "data" / "data_neraca_beras_bersih.csv"

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]
SINGKAT = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

TEAL = "#0E9F8E"
TEAL_TUA = "#0B6B63"
BPS = "#0B3C78"
LANGIT = "#3B9EE0"
PADI = "#F2B01E"
PADI_TUA = "#C98A0B"
HIJAU = "#22A06B"
MERAH = "#E5484D"
ORANYE = "#F28C28"
TEKS = "#16323A"
ABU = "#5F7480"
GARIS = "#E3ECEF"
PALET_TAHUN = [LANGIT, TEAL, BPS, PADI, ORANYE]
WARNA_KATEGORI = {"Aman": HIJAU, "Waspada": ORANYE, "Rawan": MERAH}
IKON_KATEGORI = {"Aman": "check_circle", "Waspada": "error", "Rawan": "warning"}
FONT = "Plus Jakarta Sans, Segoe UI, system-ui, sans-serif"

# =====================================================================
# HALAMAN & GAYA
# =====================================================================
st.set_page_config(
    page_title="Neraca Beras Bojonegoro | BPS",
    page_icon=str(LOGO) if LOGO.exists() else ":material/grass:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400..600,0..1,0&display=block');

/* Font: hanya elemen teks, jangan menyentuh span ikon */
html, body, .stApp, [data-testid="stMarkdownContainer"], p, li, label, input, textarea,
button, td, th, h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif;
}
.msr { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
       line-height: 1; letter-spacing: normal; text-transform: none; white-space: nowrap;
       font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 24; display: inline-block; }

/* Latar aplikasi */
.stApp { background: linear-gradient(140deg, #E4F4F0 0%, #F1F8FA 45%, #EAF1FA 100%); }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1360px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(185deg, #0B6B63 0%, #0C5A73 55%, #0B3C78 100%); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 12px; margin: 2px 4px; padding: 8px 12px; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span { color: rgba(255,255,255,.82); font-weight: 600; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,.10); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(255,255,255,.18); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span { color: #FFFFFF; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] img,
section[data-testid="stSidebar"] [data-testid="stLogo"] { background: #FFFFFF; border-radius: 12px; padding: 4px; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] { border-color: rgba(255,255,255,.18); }
section[data-testid="stSidebar"] button svg, section[data-testid="stSidebar"] [data-testid="stIconMaterial"] { color: rgba(255,255,255,.9); }
.sidebar-info { color: rgba(255,255,255,.78); font-size: .82rem; line-height: 1.65; margin: 18px 6px 0;
                padding-top: 14px; border-top: 1px solid rgba(255,255,255,.18); }
.sidebar-info b { color: #FFFFFF; }

/* Kartu (container dengan key berawalan kartu) */
[class*="st-key-kartu"] { background: #FFFFFF; border-radius: 22px; padding: 20px 22px 12px;
                          box-shadow: 0 12px 32px rgba(11, 60, 120, .07); border: 1px solid rgba(14,159,142,.10); }
.kartu-head { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.kartu-ikon { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex: none; }
.kartu-ikon .msr { font-size: 22px; }
.kartu-judul { font-weight: 800; font-size: 1.06rem; color: #16323A; line-height: 1.3; }
.kartu-ket { color: #5F7480; font-size: .86rem; line-height: 1.55; margin: 2px 0 6px; max-width: 80ch; }

/* Hero */
.hero { position: relative; overflow: hidden; border-radius: 28px; padding: 30px 34px 30px;
        background: linear-gradient(118deg, #0B6B63 0%, #0E9F8E 42%, #0C5A73 70%, #0B3C78 100%);
        color: #FFFFFF; display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 18px;
        box-shadow: 0 22px 48px rgba(11, 107, 99, .25); margin-bottom: 18px; }
.hero::before { content: ""; position: absolute; width: 420px; height: 420px; right: -120px; top: -180px;
                background: radial-gradient(circle, rgba(255,255,255,.16) 0%, rgba(255,255,255,0) 65%); }
.hero::after { content: ""; position: absolute; width: 300px; height: 300px; left: -110px; bottom: -170px;
               background: radial-gradient(circle, rgba(242,176,30,.28) 0%, rgba(242,176,30,0) 65%); }
.hero-isi { position: relative; z-index: 1; }
.hero-instansi { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
.hero-logo { width: 52px; height: 52px; border-radius: 14px; background: #FFFFFF; display: flex;
             align-items: center; justify-content: center; padding: 6px; flex: none; }
.hero-logo img { max-width: 100%; max-height: 100%; }
.hero-logo-teks { color: #0B3C78; font-weight: 800; font-size: .95rem; }
.hero-nama { font-weight: 800; font-size: 1.02rem; line-height: 1.25; }
.hero-wilayah { font-weight: 500; font-size: .9rem; opacity: .85; }
.hero-judul { font-size: 2.3rem; font-weight: 800; line-height: 1.16; letter-spacing: -0.02em; max-width: 24ch; }
.hero-pengantar { font-size: 1rem; line-height: 1.65; opacity: .9; margin-top: 12px; max-width: 60ch; }
.hero-tag { display: inline-flex; align-items: center; gap: 6px; margin-top: 18px; padding: 7px 14px;
            border-radius: 999px; background: rgba(255,255,255,.16); font-size: .82rem; font-weight: 600; }
.hero-tag .msr { font-size: 18px; color: #FFD25C; }
.hero-gambar { position: relative; z-index: 1; display: flex; align-items: flex-end; justify-content: center; }
.hero-gambar img { width: 100%; max-width: 560px; }
.hero.ringkas { padding: 24px 30px; }
.hero.ringkas .hero-judul { font-size: 1.85rem; }
@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; }
  .hero-gambar { display: none; }
  .hero-judul { font-size: 1.7rem; }
}

/* Kartu angka utama */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin: 4px 0 18px; }
.kpi { background: #FFFFFF; border-radius: 22px; padding: 18px 20px 14px; position: relative; overflow: hidden;
       box-shadow: 0 12px 32px rgba(11, 60, 120, .07); border: 1px solid rgba(14,159,142,.10); }
.kpi-atas { display: flex; align-items: center; gap: 10px; }
.kpi-label { font-size: .86rem; color: #5F7480; font-weight: 600; }
.kpi-nilai { font-size: 1.75rem; font-weight: 800; color: #16323A; margin-top: 10px; line-height: 1.2;
             font-variant-numeric: tabular-nums; }
.kpi-nilai small { font-size: .85rem; font-weight: 600; color: #5F7480; margin-left: 4px; }
.kpi-catatan { font-size: .8rem; color: #5F7480; margin-top: 4px; min-height: 1.2em; }
.kpi-grafik { margin-top: 8px; height: 44px; }
.kpi-grafik img { width: 100%; height: 44px; }
.kpi-ring { display: flex; align-items: center; gap: 14px; margin-top: 8px; }
.ring { position: relative; width: 74px; height: 74px; flex: none; }
.ring img { width: 74px; height: 74px; }
.ring span { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
             font-weight: 800; font-size: 1rem; color: #16323A; }
.naik { color: #22A06B; font-weight: 700; }
.turun { color: #E5484D; font-weight: 700; }

/* Kalender kecil */
.kal-kecil { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 8px; }
.kal-sel { border-radius: 14px; padding: 10px 6px 8px; text-align: center; }
.kal-sel b { display: block; font-size: .9rem; }
.kal-sel span { font-size: .72rem; font-weight: 600; }
.legenda { display: flex; gap: 14px; flex-wrap: wrap; font-size: .78rem; color: #5F7480; margin-top: 12px; }
.legenda i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }

/* Temuan */
.temuan-item { display: flex; gap: 12px; padding: 12px 0; border-bottom: 1px dashed #E3ECEF; }
.temuan-item:last-child { border-bottom: none; }
.temuan-item b { display: block; color: #16323A; font-size: .95rem; margin-bottom: 2px; }
.temuan-item span.isi { color: #5F7480; font-size: .86rem; line-height: 1.55; }

/* Kalender risiko proyeksi */
.kalender { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 6px 0 6px; }
@media (max-width: 1100px) { .kalender { grid-template-columns: repeat(4, 1fr); } }
@media (max-width: 640px) { .kalender { grid-template-columns: repeat(2, 1fr); } }
.bln { border-radius: 18px; padding: 14px 14px 12px; position: relative; overflow: hidden; }
.bln-atas { display: flex; justify-content: space-between; align-items: center; }
.bln-nama { font-weight: 800; font-size: .95rem; }
.bln .msr { font-size: 22px; }
.bln-persen { font-size: 1.75rem; font-weight: 800; font-variant-numeric: tabular-nums; margin-top: 6px; line-height: 1.15; }
.bln-meter { height: 7px; background: rgba(255,255,255,.65); border-radius: 4px; overflow: hidden; margin: 8px 0 6px; }
.bln-meter span { display: block; height: 100%; border-radius: 4px; }
.bln-ket { font-size: .76rem; font-weight: 600; opacity: .85; }

/* Panel model */
.panel-model { border-radius: 20px; padding: 0; overflow: hidden; background: #FFFFFF; height: 100%;
               box-shadow: 0 12px 32px rgba(11, 60, 120, .07); border: 1px solid rgba(14,159,142,.10); }
.panel-kepala { padding: 16px 20px; color: #FFFFFF; display: flex; justify-content: space-between; align-items: center; }
.panel-kepala b { font-size: 1.05rem; }
.panel-kepala small { display: block; opacity: .85; font-weight: 500; }
.lencana { background: #FFD25C; color: #5B3F00; font-weight: 800; font-size: .75rem; padding: 4px 12px; border-radius: 999px; }
.panel-badan { padding: 14px 20px 16px; }
.panel-angka { display: flex; gap: 26px; margin-bottom: 8px; }
.panel-angka div div:first-child { font-size: .78rem; color: #5F7480; font-weight: 600; }
.panel-angka div div:last-child { font-size: 1.35rem; font-weight: 800; color: #16323A; }
.tabel-uji { width: 100%; border-collapse: collapse; font-size: .88rem; }
.tabel-uji td { padding: 8px 2px; border-bottom: 1px solid #EEF3F5; color: #16323A; }
.tabel-uji td:last-child { text-align: right; }
.tabel-uji small { color: #5F7480; }
.chip { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .76rem; font-weight: 700; }
.chip-ya { background: #E1F5EC; color: #1C8A5B; }
.chip-tidak { background: #FDE8E8; color: #C93C40; }

/* Langkah metodologi */
.langkah { display: grid; grid-template-columns: 46px 1fr; gap: 14px; padding: 12px 0; }
.langkah-no { width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
              font-weight: 800; color: #FFFFFF; background: linear-gradient(135deg, #0E9F8E, #0B3C78); }
.langkah b { color: #16323A; display: block; margin-bottom: 2px; }
.langkah span { color: #5F7480; font-size: .9rem; line-height: 1.6; }

/* Tab model pil */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #FFFFFF; padding: 6px; border-radius: 16px;
                                     box-shadow: 0 8px 24px rgba(11, 60, 120, .06); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 11px; padding: 8px 16px; height: auto; }
.stTabs [data-baseweb="tab"] p { font-weight: 700; color: #5F7480; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0E9F8E, #0B6B63); }
.stTabs [aria-selected="true"] p { color: #FFFFFF; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# FUNGSI BANTU UMUM
# =====================================================================
def fmt(angka, desimal=0):
    teks = f"{angka:,.{desimal}f}"
    return teks.replace(",", "#").replace(".", ",").replace("#", ".")


def persen(angka, desimal=2):
    return f"{'+' if angka > 0 else ''}{fmt(angka, desimal)}%"


def gabung(daftar):
    daftar = list(daftar)
    if not daftar:
        return ""
    if len(daftar) == 1:
        return daftar[0]
    if len(daftar) == 2:
        return f"{daftar[0]} dan {daftar[1]}"
    return ", ".join(daftar[:-1]) + ", dan " + daftar[-1]


def kategori(peluang):
    if peluang <= 20:
        return "Aman"
    if peluang <= 50:
        return "Waspada"
    return "Rawan"


def warna_tahun(tahun, semua):
    return PALET_TAHUN[sorted(semua).index(tahun) % len(PALET_TAHUN)]


def tint(hex_warna, alpha):
    h = hex_warna.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def rapat(html):
    return "\n".join(baris.strip() for baris in html.splitlines())


def md(html, wadah=None):
    (wadah or st).markdown(rapat(html), unsafe_allow_html=True)


def svg_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()


def ikon(nama, warna, ukuran=22):
    return f'<span class="msr" style="font-size:{ukuran}px;color:{warna}">{nama}</span>'


def logo_html():
    if LOGO.exists():
        b64 = base64.b64encode(LOGO.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="Logo BPS">'
    return '<span class="hero-logo-teks">BPS</span>'


_hitung_kartu = {"n": 0}


def kartu():
    _hitung_kartu["n"] += 1
    return st.container(key=f"kartu_{_hitung_kartu['n']}")


def kepala_kartu(judul, nama_ikon, warna, ket=None):
    html = f"""
    <div class="kartu-head">
        <div class="kartu-ikon" style="background:{tint(warna, .14)}">{ikon(nama_ikon, warna)}</div>
        <div class="kartu-judul">{judul}</div>
    </div>"""
    if ket:
        html += f'<div class="kartu-ket">{ket}</div>'
    md(html)


def pilih(label, opsi, default, key):
    if hasattr(st, "segmented_control"):
        nilai = st.segmented_control(label, opsi, default=default, key=key, label_visibility="collapsed")
        return nilai if nilai is not None else default
    return st.radio(label, opsi, index=opsi.index(default), horizontal=True, key=key,
                    label_visibility="collapsed")


def gaya(fig, tinggi=400, legend=True):
    fig.update_layout(
        template="plotly_white", height=tinggi,
        font=dict(family=FONT, size=13, color=TEKS),
        margin=dict(l=6, r=6, t=34 if legend else 10, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GARIS, font_color=TEKS),
        separators=",.", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
    )
    try:
        fig.update_layout(barcornerradius=7)
    except Exception:
        pass
    fig.update_xaxes(showgrid=False, linecolor="#D3DEE3", tickcolor="#D3DEE3")
    fig.update_yaxes(gridcolor="#EEF3F5", zeroline=False, linecolor="#D3DEE3")
    return fig


def tampil(fig):
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
           "toImageButtonOptions": {"format": "png", "scale": 2}}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def tabel(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def sumbu_bulan(fig, label, setiap=3):
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=label,
                     tickmode="array", tickvals=label[::setiap], tickangle=0)
    return fig


# =====================================================================
# ILUSTRASI & GRAFIK MINI (SVG BUATAN SENDIRI)
# =====================================================================
def svg_sawah():
    """Ilustrasi sawah terasering dan tanaman padi untuk hero (gambar asli, dibuat dengan kode)."""
    def daun(x0, y0, x1, y1, lebar, warna):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        pj = math.hypot(dx, dy) or 1
        nx, ny = -dy / pj * lebar, dx / pj * lebar
        return (f'<path d="M{x0:.1f},{y0:.1f} Q{mx + nx:.1f},{my + ny:.1f} {x1:.1f},{y1:.1f} '
                f'Q{mx - nx * 0.35:.1f},{my - ny * 0.35:.1f} {x0:.1f},{y0:.1f} Z" fill="{warna}"/>')

    def batang(bx, tinggi, condong, arah, sk=1.0):
        dasar = 272
        tx, ty = bx + condong, dasar - tinggi
        s = daun(bx, dasar - 4, bx - 58 * arah * sk, dasar - tinggi * 0.55, 7 * sk * arah, "url(#gDaun1)")
        s += daun(bx, dasar - 10, bx + 44 * arah * sk, dasar - tinggi * 0.62, -6 * sk * arah, "url(#gDaun2)")
        s += (f'<path d="M{bx},{dasar + 2} Q{bx + condong * 0.15:.1f},{dasar - tinggi * 0.6:.1f} {tx:.1f},{ty:.1f}" '
              f'stroke="url(#gBatang)" stroke-width="{4.2 * sk:.1f}" fill="none" stroke-linecap="round"/>')
        titik = []
        for k in range(13):
            t = k / 12
            titik.append((tx + arah * 40 * sk * t, ty + 52 * sk * t ** 1.6))
        s += ('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in titik) +
              f'" fill="none" stroke="#C99A1E" stroke-width="{1.6 * sk:.1f}" stroke-linecap="round"/>')
        for k in range(1, 13):
            t = k / 12
            x, y = titik[k]
            dxt = arah * 40 * sk
            dyt = 52 * sk * 1.6 * max(t, 0.05) ** 0.6
            sudut = math.degrees(math.atan2(dyt, dxt))
            nx, ny = -math.sin(math.radians(sudut)), math.cos(math.radians(sudut))
            off = 5.2 * sk
            for sisi in (-1, 1):
                gx, gy = x + nx * off * sisi, y + ny * off * sisi
                s += (f'<ellipse cx="{gx:.1f}" cy="{gy:.1f}" rx="{3.6 * sk:.1f}" ry="{7.2 * sk:.1f}" fill="url(#gGabah)" '
                      f'transform="rotate({sudut - 90 + sisi * 28:.0f} {gx:.1f} {gy:.1f})"/>')
        s += (f'<ellipse cx="{titik[-1][0] + arah * 2:.1f}" cy="{titik[-1][1] + 6 * sk:.1f}" rx="{3.4 * sk:.1f}" ry="{7 * sk:.1f}" '
              f'fill="url(#gGabah)" transform="rotate({arah * 10:.0f} {titik[-1][0]:.1f} {titik[-1][1]:.1f})"/>')
        return s

    belakang = "".join(batang(x, h, c, a, 0.66) for x, h, c, a in
                       [(250, 118, 10, 1), (292, 132, 12, 1), (566, 124, -10, -1)])
    depan1 = "".join(batang(x, h, c, a) for x, h, c, a in [(352, 172, 18, 1), (398, 206, 22, 1), (440, 186, 18, 1)])
    depan2 = "".join(batang(x, h, c, a) for x, h, c, a in [(486, 198, -20, -1), (526, 168, -16, -1)])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 280">
<defs>
<radialGradient id="gMatahari" cx="50%" cy="50%" r="50%">
<stop offset="0%" stop-color="#FFF1B8"/><stop offset="55%" stop-color="#FFD25C" stop-opacity=".5"/>
<stop offset="100%" stop-color="#FFD25C" stop-opacity="0"/></radialGradient>
<linearGradient id="gBukit1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#9BE6CD"/><stop offset="1" stop-color="#46B894"/></linearGradient>
<linearGradient id="gBukit2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F1CF63"/><stop offset="1" stop-color="#D19A1C"/></linearGradient>
<linearGradient id="gBukit3" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5ECF9F"/><stop offset="1" stop-color="#1A8A64"/></linearGradient>
<linearGradient id="gBatang" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#1B7A42"/><stop offset="1" stop-color="#7AD48F"/></linearGradient>
<linearGradient id="gDaun1" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#1F8A4C"/><stop offset="1" stop-color="#5CCB7E"/></linearGradient>
<linearGradient id="gDaun2" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#2AA05A"/><stop offset="1" stop-color="#8BE0A0"/></linearGradient>
<radialGradient id="gGabah" cx="35%" cy="30%" r="75%"><stop offset="0" stop-color="#FFF0B3"/><stop offset=".55" stop-color="#F6C443"/><stop offset="1" stop-color="#D4950F"/></radialGradient>
<linearGradient id="gPudar" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".28" stop-color="#fff" stop-opacity="1"/></linearGradient>
<mask id="mPudar"><rect width="600" height="280" fill="url(#gPudar)"/></mask>
</defs>
<style>
.goyang1 {{ transform-origin: 400px 272px; animation: goyang 4.8s ease-in-out infinite alternate; }}
.goyang2 {{ transform-origin: 505px 272px; animation: goyang 6.2s ease-in-out infinite alternate-reverse; }}
@keyframes goyang {{ from {{ transform: rotate(-2deg); }} to {{ transform: rotate(2.4deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .goyang1, .goyang2 {{ animation: none; }} }}
</style>
<circle cx="452" cy="78" r="82" fill="url(#gMatahari)"/>
<circle cx="452" cy="78" r="32" fill="#FFD25C"/>
<g fill="#FFFFFF" opacity=".92"><ellipse cx="250" cy="62" rx="38" ry="11"/><ellipse cx="272" cy="52" rx="23" ry="14"/><ellipse cx="234" cy="56" rx="17" ry="10"/></g>
<g fill="#FFFFFF" opacity=".6"><ellipse cx="560" cy="126" rx="26" ry="8"/><ellipse cx="572" cy="119" rx="15" ry="9"/></g>
<g mask="url(#mPudar)">
<path d="M0 176 C 110 140, 250 128, 360 152 S 520 132, 600 126 L600 280 L0 280 Z" fill="url(#gBukit1)"/>
<path d="M0 206 C 130 176, 260 172, 380 194 S 530 184, 600 176 L600 280 L0 280 Z" fill="url(#gBukit2)"/>
<g fill="none" stroke="#FFF4C8" stroke-opacity=".6" stroke-width="2">
<path d="M0 222 C 130 194, 260 190, 380 210 S 530 200, 600 194"/>
<path d="M0 238 C 140 212, 270 208, 390 226 S 535 218, 600 212"/></g>
<path d="M0 246 C 150 222, 290 226, 410 242 S 545 236, 600 232 L600 280 L0 280 Z" fill="url(#gBukit3)"/>
<g fill="none" stroke="#BDF2D9" stroke-opacity=".5" stroke-width="2"><path d="M0 262 C 150 242, 290 246, 410 258 S 545 254, 600 250"/></g>
</g>
<g opacity=".85">{belakang}</g>
<g class="goyang1">{depan1}</g>
<g class="goyang2">{depan2}</g>
</svg>"""
    return svg_uri(svg)


def svg_sparkline(nilai, warna, isi=True):
    nilai = list(nilai)
    w, h, p = 220, 44, 3
    lo, hi = min(nilai), max(nilai)
    rentang = hi - lo or 1
    titik = [(p + i * (w - 2 * p) / (len(nilai) - 1), h - p - (v - lo) / rentang * (h - 2 * p))
             for i, v in enumerate(nilai)]
    garis = " ".join(f"{x:.1f},{y:.1f}" for x, y in titik)
    area = f"M{titik[0][0]:.1f},{h} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in titik) + f" L{titik[-1][0]:.1f},{h} Z"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{warna}" stop-opacity=".35"/>
<stop offset="1" stop-color="{warna}" stop-opacity="0"/></linearGradient></defs>
{f'<path d="{area}" fill="url(#g)"/>' if isi else ''}
<polyline points="{garis}" fill="none" stroke="{warna}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
</svg>"""
    return svg_uri(svg)


def svg_batang_mini(nilai, warna_pos, warna_neg):
    nilai = list(nilai)
    w, h = 220, 44
    maks = max(abs(v) for v in nilai) or 1
    lebar = w / len(nilai)
    tengah = h * 0.72
    isi = ""
    for i, v in enumerate(nilai):
        tinggi = abs(v) / maks * (tengah - 2) if v > 0 else abs(v) / maks * (h - tengah - 2) * 3
        tinggi = max(tinggi, 1.5)
        y = tengah - tinggi if v > 0 else tengah
        isi += f'<rect x="{i * lebar + lebar * 0.18:.1f}" y="{y:.1f}" width="{lebar * 0.64:.1f}" height="{tinggi:.1f}" rx="1.5" fill="{warna_pos if v > 0 else warna_neg}"/>'
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="none">{isi}</svg>'
    return svg_uri(svg)


def svg_cincin(persen_nilai, warna, latar="#E9F2F1"):
    r = 30
    keliling = 2 * math.pi * r
    isi = keliling * max(min(persen_nilai, 100), 0) / 100
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 74 74">
<circle cx="37" cy="37" r="{r}" fill="none" stroke="{latar}" stroke-width="8"/>
<circle cx="37" cy="37" r="{r}" fill="none" stroke="{warna}" stroke-width="8" stroke-linecap="round"
stroke-dasharray="{isi:.1f} {keliling:.1f}" transform="rotate(-90 37 37)"/></svg>"""
    return svg_uri(svg)


def svg_siklus(rekap):
    """Infografik jalan berliku 12 bulan: penanda panen dan bulan rawan."""
    w, h = 1000, 262
    xs = [60 + i * (w - 120) / 11 for i in range(12)]
    ys = [150 + 34 * math.sin(i * 0.95 + 0.4) for i in range(12)]
    jalur = f"M{xs[0]:.1f},{ys[0]:.1f}"
    for i in range(1, 12):
        cx = (xs[i - 1] + xs[i]) / 2
        jalur += f" C{cx:.1f},{ys[i - 1]:.1f} {cx:.1f},{ys[i]:.1f} {xs[i]:.1f},{ys[i]:.1f}"
    rata_kons = rekap["rata_kons"].mean()
    urut = rekap["rata_prod"].sort_values(ascending=False).index.tolist()
    puncak1 = urut[0]
    puncak2 = next((b for b in urut[1:] if min(abs(b - puncak1), 12 - abs(b - puncak1)) >= 3), None)
    penanda = ""
    for i in range(12):
        b = i + 1
        r = rekap.loc[b]
        x, y = xs[i], ys[i]
        if b in (puncak1, puncak2):
            warna, judul, ket = PADI_TUA, ("Panen raya" if b == puncak1 else "Panen kedua"), f"{fmt(r['rata_prod'] / 1000, 0)} rb ton"
        elif r["n_defisit"] >= 2:
            warna, judul, ket = MERAH, "Rawan defisit", f"{int(r['n_defisit'])}x defisit"
        else:
            warna, judul, ket = None, None, None
        if warna:
            penanda += f'<path d="M{x:.1f},{y - 10:.1f} c-13,-15 -13,-32 0,-32 c13,0 13,17 0,32 z" fill="{warna}"/>'
            penanda += f'<circle cx="{x:.1f}" cy="{y - 31:.1f}" r="5.5" fill="#FFFFFF"/>'
            penanda += f'<text x="{x:.1f}" y="{y - 70:.1f}" text-anchor="middle" font-size="15" font-weight="800" fill="{warna}">{judul}</text>'
            penanda += f'<text x="{x:.1f}" y="{y - 52:.1f}" text-anchor="middle" font-size="13" fill="{ABU}">{ket}</text>'
        warna_titik = HIJAU if r["rata_prod"] > rata_kons else MERAH
        penanda += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#FFFFFF" stroke="{warna_titik}" stroke-width="3.5"/>'
        penanda += f'<text x="{x:.1f}" y="{h - 6}" text-anchor="middle" font-size="13" font-weight="700" fill="{TEKS}">{SINGKAT[i]}</text>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Plus Jakarta Sans, Segoe UI, Arial, sans-serif">
<path d="{jalur}" fill="none" stroke="#D7E4E8" stroke-width="22" stroke-linecap="round"/>
<path d="{jalur}" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-dasharray="10 9" stroke-linecap="round"/>
{penanda}</svg>"""
    return svg_uri(svg)


def hero(judul_teks, pengantar, tag=None, ringkas=False, wadah=None):
    tag_html = f'<div class="hero-tag">{ikon("grass", "#FFD25C", 18)} {tag}</div>' if tag else ""
    kelas = "hero ringkas" if ringkas else "hero"
    md(f"""
    <div class="{kelas}">
        <div class="hero-isi">
            <div class="hero-instansi">
                <div class="hero-logo">{logo_html()}</div>
                <div><div class="hero-nama">Badan Pusat Statistik</div><div class="hero-wilayah">Kabupaten Bojonegoro</div></div>
            </div>
            <div class="hero-judul">{judul_teks}</div>
            <div class="hero-pengantar">{pengantar}</div>
            {tag_html}
        </div>
        <div class="hero-gambar"><img src="{svg_sawah()}" alt="Ilustrasi sawah dan tanaman padi"></div>
    </div>""", wadah)


def kartu_kpi(item):
    """item: dict(label, nilai, satuan, ikon, warna, catatan, grafik_html)"""
    isi = ""
    for k in item:
        isi += f"""
        <div class="kpi">
            <div class="kpi-atas">
                <div class="kartu-ikon" style="background:{tint(k['warna'], .14)}">{ikon(k['ikon'], k['warna'])}</div>
                <div class="kpi-label">{k['label']}</div>
            </div>
            {k.get('badan') or f'<div class="kpi-nilai">{k["nilai"]}<small>{k.get("satuan", "")}</small></div>'}
            <div class="kpi-catatan">{k.get('catatan') or ''}</div>
            {k.get('grafik', '')}
        </div>"""
    md(f'<div class="kpi-grid">{isi}</div>')


def delta_html(nilai, pembanding, kebalik=False):
    baik = nilai >= 0 if not kebalik else nilai <= 0
    return f'<span class="{"naik" if baik else "turun"}">{persen(nilai)}</span> {pembanding}'


# =====================================================================
# DATA & ANALISIS
# =====================================================================
@st.cache_data
def muat_data():
    df = pd.read_csv(DATA)
    df = df.sort_values(["Tahun", "Bulan_No"]).reset_index(drop=True)
    df["Bulan"] = df["Bulan_No"].map(lambda b: BULAN[b - 1])
    df["Bulan_Singkat"] = df["Bulan_No"].map(lambda b: SINGKAT[b - 1])
    df["Label"] = df["Bulan_Singkat"] + " " + df["Tahun"].astype(str)
    df["Surplus_Defisit_Ton"] = df["Produksi_Beras_Ton"] - df["Konsumsi_Beras_Ton"]
    df["Status"] = np.where(df["Surplus_Defisit_Ton"] > 0, "Surplus", "Defisit")
    df["t"] = np.arange(1, len(df) + 1)
    return df


def matriks_x(t, bulan_no):
    t = np.asarray(t, dtype=float)
    bulan_no = np.asarray(bulan_no)
    X = pd.DataFrame({"const": np.ones(len(t)), "t": t})
    for i in range(2, 13):
        X[f"D_{SINGKAT[i - 1]}"] = (bulan_no == i).astype(float)
    return X


def uji_asumsi(model):
    resid = model.resid
    sw_stat, sw_p = stats.shapiro(resid)
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
    dw = durbin_watson(resid)
    bg = acorr_breusch_godfrey(model, nlags=1)
    if isinstance(bg, tuple):
        bg_stat, bg_p = bg[0], bg[1]
    else:
        bg_stat, bg_p = getattr(bg, "lm", np.nan), getattr(bg, "lmpval", np.nan)
    exog = model.model.exog
    vif_maks = max(variance_inflation_factor(exog, i) for i in range(1, exog.shape[1]))
    return pd.DataFrame([
        ["Normalitas residual", "Shapiro-Wilk", sw_stat, sw_p, sw_p > 0.05],
        ["Normalitas residual", "Jarque-Bera", jb_stat, jb_p, jb_p > 0.05],
        ["Homoskedastisitas", "Breusch-Pagan", bp_stat, bp_p, bp_p > 0.05],
        ["Non-autokorelasi", "Breusch-Godfrey (lag 1)", bg_stat, bg_p, bg_p > 0.05],
        ["Non-autokorelasi", "Durbin-Watson", dw, np.nan, 1.5 < dw < 2.5],
        ["Non-multikolinearitas", "VIF maksimum", vif_maks, np.nan, vif_maks < 10],
    ], columns=["Asumsi", "Uji", "Statistik", "p_value", "Terpenuhi"])


@st.cache_resource
def analisis():
    df = muat_data()
    X = matriks_x(df["t"], df["Bulan_No"])
    y = df["Produksi_Beras_Ton"].astype(float)
    model1 = sm.OLS(y, X).fit()
    model2 = sm.OLS(np.log(y), X).fit()
    smearing = float(np.mean(np.exp(model2.resid)))
    model_kons = sm.OLS(df["Konsumsi_Beras_Ton"].astype(float), sm.add_constant(df["t"].astype(float))).fit()

    ci = model2.conf_int()
    koef = pd.DataFrame({
        "Variabel": model2.params.index, "Koefisien": model2.params.values,
        "Std_Error": model2.bse.values, "t_hitung": model2.tvalues.values, "p_value": model2.pvalues.values,
        "Persen": (np.exp(model2.params.values) - 1) * 100,
        "Persen_Bawah": (np.exp(ci[0].values) - 1) * 100,
        "Persen_Atas": (np.exp(ci[1].values) - 1) * 100,
    })

    tahun_proy = int(df["Tahun"].max()) + 1
    t_proy = np.arange(len(df) + 1, len(df) + 13)
    pred = model2.get_prediction(matriks_x(t_proy, np.arange(1, 13))).summary_frame(alpha=0.05)
    proy = pd.DataFrame({
        "Bulan_No": np.arange(1, 13), "Bulan": BULAN, "Bulan_Singkat": SINGKAT,
        "Label": [f"{s} {tahun_proy}" for s in SINGKAT], "t": t_proy,
        "mu": pred["mean"].values, "sd": np.sqrt(pred["mean_se"].values ** 2 + model2.scale),
        "lo": pred["obs_ci_lower"].values, "hi": pred["obs_ci_upper"].values,
        "kons": model_kons.params["const"] + model_kons.params["t"] * t_proy,
    })
    return {"df": df, "model1": model1, "model2": model2, "model_kons": model_kons, "smearing": smearing,
            "uji1": uji_asumsi(model1), "uji2": uji_asumsi(model2), "koef": koef,
            "proy": proy, "tahun_proy": tahun_proy}


def hitung_proyeksi(A, naik_konsumsi=0.0, ubah_produksi=0.0):
    P = A["proy"].copy()
    fp, fk = 1 + ubah_produksi / 100, 1 + naik_konsumsi / 100
    P["Produksi"] = np.exp(P["mu"]) * A["smearing"] * fp
    P["Bawah"] = np.exp(P["lo"]) * fp
    P["Atas"] = np.exp(P["hi"]) * fp
    P["Konsumsi"] = P["kons"] * fk
    P["Surplus"] = P["Produksi"] - P["Konsumsi"]
    P["Peluang"] = stats.norm.cdf((np.log(P["Konsumsi"]) - (P["mu"] + np.log(fp))) / P["sd"]) * 100
    P["Kategori"] = P["Peluang"].apply(kategori)
    return P


A = analisis()
DF = A["df"]
TAHUN = sorted(DF["Tahun"].unique().tolist())
RENTANG = f"{TAHUN[0]}–{TAHUN[-1]}"
REKAP_BULAN = DF.groupby("Bulan_No").agg(
    n_defisit=("Status", lambda s: (s == "Defisit").sum()),
    rata_defisit=("Surplus_Defisit_Ton", "mean"),
    rata_prod=("Produksi_Beras_Ton", "mean"),
    rata_kons=("Konsumsi_Beras_Ton", "mean"),
)


# =====================================================================
# HALAMAN 1: RINGKASAN
# =====================================================================
def halaman_ringkasan():
    rawan = [BULAN[b - 1] for b, r in REKAP_BULAN.iterrows() if r["n_defisit"] == len(TAHUN)]
    total_surplus = DF["Surplus_Defisit_Ton"].sum()
    judul_hero = (f"Surplus {fmt(total_surplus)} ton dalam {len(TAHUN)} tahun, tetapi {gabung(rawan)} selalu defisit"
                  if rawan else f"Surplus {fmt(total_surplus)} ton dalam {len(TAHUN)} tahun")
    hero(judul_hero,
         "Bojonegoro memproduksi beras jauh di atas kebutuhannya. Tantangannya ada pada waktu: panen "
         "terkumpul di beberapa bulan, sementara kebutuhan tetap sama sepanjang tahun.",
         tag=f"Neraca beras bulanan {RENTANG}")

    pilihan = pilih("Periode", ["Semua tahun"] + [str(t) for t in TAHUN], "Semua tahun", "ringkasan_tahun")
    if pilihan == "Semua tahun":
        data, sebelum = DF, None
    else:
        th = int(pilihan)
        data = DF[DF["Tahun"] == th]
        sebelum = DF[DF["Tahun"] == th - 1] if th - 1 in TAHUN else None

    prod, kons = data["Produksi_Beras_Ton"].sum(), data["Konsumsi_Beras_Ton"].sum()
    surp = data["Surplus_Defisit_Ton"].sum()
    n_def = int((data["Status"] == "Defisit").sum())
    pct_aman = (len(data) - n_def) / len(data) * 100
    c_prod = c_kons = c_surp = None
    if sebelum is not None:
        teks = f"dari {int(pilihan) - 1}"
        c_prod = delta_html((prod / sebelum["Produksi_Beras_Ton"].sum() - 1) * 100, teks)
        c_kons = delta_html((kons / sebelum["Konsumsi_Beras_Ton"].sum() - 1) * 100, teks, kebalik=True)
        c_surp = delta_html((surp / sebelum["Surplus_Defisit_Ton"].sum() - 1) * 100, teks)

    kartu_kpi([
        dict(label="Produksi beras", nilai=fmt(prod), satuan="ton", ikon="agriculture", warna=PADI_TUA,
             catatan=c_prod or "total seluruh bulan",
             grafik=f'<div class="kpi-grafik"><img src="{svg_sparkline(data["Produksi_Beras_Ton"], PADI)}"></div>'),
        dict(label="Konsumsi beras", nilai=fmt(kons), satuan="ton", ikon="rice_bowl", warna=BPS,
             catatan=c_kons or "kebutuhan penduduk",
             grafik=f'<div class="kpi-grafik"><img src="{svg_sparkline(data["Konsumsi_Beras_Ton"], BPS)}"></div>'),
        dict(label="Surplus bersih", nilai=fmt(surp), satuan="ton", ikon="inventory_2", warna=HIJAU,
             catatan=c_surp or "produksi dikurangi konsumsi",
             grafik=f'<div class="kpi-grafik"><img src="{svg_batang_mini(data["Surplus_Defisit_Ton"], HIJAU, MERAH)}"></div>'),
        dict(label="Rasio swasembada", nilai=fmt(prod / kons * 100, 1), satuan="%", ikon="donut_large", warna=TEAL,
             catatan="produksi dibanding kebutuhan",
             grafik=f'<div class="kpi-grafik"><img src="{svg_sparkline((data["Produksi_Beras_Ton"] / data["Konsumsi_Beras_Ton"]).values, TEAL)}"></div>'),
        dict(label="Bulan aman", ikon="event_available", warna=LANGIT,
             badan=f'<div class="kpi-ring"><div class="ring"><img src="{svg_cincin(pct_aman, LANGIT)}"><span>{fmt(pct_aman, 0)}%</span></div>'
                   f'<div><div class="kpi-nilai" style="margin-top:0">{len(data) - n_def}<small>dari {len(data)}</small></div>'
                   f'<div class="kpi-catatan"><span class="turun">{n_def} bulan</span> defisit</div></div></div>'),
    ])

    kiri, kanan = st.columns([2.15, 1], gap="medium")
    with kiri:
        with kartu():
            kepala_kartu("Pasokan naik turun, kebutuhan nyaris datar", "show_chart", PADI_TUA,
                         "Area kuning adalah produksi, garis biru adalah kebutuhan. Kolom merah menandai bulan defisit.")
            label = data["Label"].tolist()
            fig = go.Figure()
            for i, (_, r) in enumerate(data.iterrows()):
                if r["Status"] == "Defisit":
                    fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor=MERAH, opacity=0.09, line_width=0, layer="below")
            fig.add_trace(go.Scatter(
                x=label, y=data["Produksi_Beras_Ton"], name="Produksi beras", mode="lines+markers",
                line=dict(color=PADI, width=3, shape="spline", smoothing=0.6),
                fill="tozeroy", fillcolor="rgba(242,176,30,0.20)",
                marker=dict(size=8, color=np.where(data["Status"] == "Defisit", MERAH, PADI), line=dict(color="white", width=1.5)),
                hovertemplate="%{x}<br>Produksi %{y:,.0f} ton<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=label, y=data["Konsumsi_Beras_Ton"], name="Kebutuhan konsumsi", mode="lines",
                line=dict(color=BPS, width=2.4, dash="dash"), hovertemplate="%{x}<br>Kebutuhan %{y:,.0f} ton<extra></extra>"))
            gaya(fig, 400)
            sumbu_bulan(fig, label, 1 if len(label) <= 12 else 3)
            tampil(fig)
    with kanan:
        with kartu():
            kepala_kartu("Kalender rawan", "calendar_month", MERAH,
                         f"Berapa kali setiap bulan mengalami defisit selama {len(TAHUN)} tahun.")
            sel = ""
            for b, r in REKAP_BULAN.iterrows():
                n = int(r["n_defisit"])
                warna = HIJAU if n == 0 else (ORANYE if n == 1 else MERAH)
                sel += (f'<div class="kal-sel" style="background:{tint(warna, .13)};color:{warna}">'
                        f'<b>{SINGKAT[b - 1]}</b><span>{"aman" if n == 0 else f"{n}x defisit"}</span></div>')
            md(f'<div class="kal-kecil">{sel}</div>'
               f'<div class="legenda"><span><i style="background:{HIJAU}"></i>selalu surplus</span>'
               f'<span><i style="background:{ORANYE}"></i>sekali defisit</span>'
               f'<span><i style="background:{MERAH}"></i>berulang</span></div>')
        with kartu():
            kepala_kartu("Temuan utama", "lightbulb", TEAL)
            urut = REKAP_BULAN["rata_prod"].sort_values(ascending=False).index.tolist()
            puncak = urut[0]
            kali = REKAP_BULAN.loc[puncak, "rata_prod"] / DF["Konsumsi_Beras_Ton"].mean()
            item = [("agriculture", PADI_TUA, f"{BULAN[puncak - 1]} puncak panen raya",
                     f"Rata-rata {fmt(REKAP_BULAN.loc[puncak, 'rata_prod'])} ton, sekitar {fmt(kali, 0)} kali kebutuhan sebulan.")]
            if rawan:
                rata_def = REKAP_BULAN.loc[[BULAN.index(b) + 1 for b in rawan], "rata_defisit"].mean()
                item.append(("warning", MERAH, f"{gabung(rawan)} selalu defisit",
                             f"Kekurangan rata-rata {fmt(abs(rata_def))} ton per bulan, di jeda antar musim panen."))
            if len(TAHUN) >= 2:
                a, b = TAHUN[-2], TAHUN[-1]
                ka, kb = (DF.loc[DF["Tahun"] == x, "Konsumsi_Beras_Ton"].sum() for x in (a, b))
                pa, pb = (DF.loc[DF["Tahun"] == x, "Produksi_Beras_Ton"].sum() for x in (a, b))
                item.append(("trending_up", BPS, f"Kebutuhan {b} tumbuh {persen((kb / ka - 1) * 100)}",
                             f"Produksi hanya {persen((pb / pa - 1) * 100)}, sehingga ruang surplus menyempit."))
            md("".join(f'<div class="temuan-item"><div class="kartu-ikon" style="background:{tint(w, .14)}">{ikon(i, w)}</div>'
                       f'<div><b>{j}</b><span class="isi">{k}</span></div></div>' for i, w, j, k in item))

    with kartu():
        kepala_kartu("Siklus musim dalam setahun", "route", TEAL,
                     "Perjalanan pasokan beras dari Januari sampai Desember. Titik hijau berarti rata-rata produksi "
                     "di atas kebutuhan, titik merah berarti di bawahnya.")
        md(f'<img src="{svg_siklus(REKAP_BULAN)}" style="width:100%;margin:4px 0 6px" alt="Siklus musim tanam dan panen">')

    kiri, kanan = st.columns([1.6, 1], gap="medium")
    with kiri:
        with kartu():
            kepala_kartu("Surplus dan defisit per bulan", "bar_chart", HIJAU)
            label = data["Label"].tolist()
            fig = go.Figure(go.Bar(x=label, y=data["Surplus_Defisit_Ton"],
                                   marker_color=np.where(data["Surplus_Defisit_Ton"] > 0, HIJAU, MERAH),
                                   hovertemplate="%{x}<br>%{y:,.0f} ton<extra></extra>"))
            fig.add_hline(y=0, line_color="#9FB3BC", line_width=1)
            gaya(fig, 330, legend=False)
            sumbu_bulan(fig, label, 1 if len(label) <= 12 else 3)
            tampil(fig)
    with kanan:
        with kartu():
            kepala_kartu("Daftar bulan defisit", "format_list_bulleted", MERAH)
            defisit = data[data["Status"] == "Defisit"]
            if len(defisit):
                tabel(pd.DataFrame({
                    "Bulan": defisit["Bulan"] + " " + defisit["Tahun"].astype(str),
                    "Produksi": defisit["Produksi_Beras_Ton"].map(fmt),
                    "Kekurangan (ton)": defisit["Surplus_Defisit_Ton"].abs().map(fmt),
                }), height=290)
            else:
                st.markdown("Tidak ada bulan defisit pada periode ini.")


# =====================================================================
# HALAMAN 2: EKSPLORASI DATA
# =====================================================================
def halaman_eda():
    hero("Mengenali pola sebelum membuat model",
         "Setiap tab menjawab satu pertanyaan tentang data, dan jawabannya menjadi dasar bentuk model regresi.",
         tag="Eksplorasi data", ringkas=True)
    tab = st.tabs(["Statistik deskriptif", "Pola musiman", "Antar tahun", "Sebaran data", "Luas panen", "Nilai ekstrem"])

    with tab[0]:
        kolom = {"Luas_Panen_Ha": "Luas panen (ha)", "Produksi_Gabah_Ton": "Produksi gabah (ton)",
                 "Produksi_Beras_Ton": "Produksi beras (ton)", "Konsumsi_Beras_Ton": "Konsumsi beras (ton)",
                 "Surplus_Defisit_Ton": "Surplus/defisit (ton)"}
        with kartu():
            kepala_kartu(f"Ringkasan statistik {len(DF)} bulan", "table_chart", BPS)
            rows = []
            for k, nama in kolom.items():
                s = DF[k]
                cv = "" if k == "Surplus_Defisit_Ton" else fmt(s.std() / s.mean() * 100, 1) + "%"
                rows.append([nama, fmt(s.mean()), fmt(s.median()), fmt(s.std()), fmt(s.min()), fmt(s.max()), cv, fmt(s.skew(), 2)])
            tabel(pd.DataFrame(rows, columns=["Variabel", "Rata-rata", "Median", "Simpangan baku", "Minimum",
                                              "Maksimum", "Koef. variasi", "Kemencengan"]))
        with kartu():
            kepala_kartu("Seberapa besar fluktuasinya", "equalizer", PADI_TUA,
                         "Kebutuhan konsumsi hampir tidak bergerak, sedangkan panen dan produksi berfluktuasi lebih dari 100%.")
            cv = pd.Series({kolom[k]: DF[k].std() / DF[k].mean() * 100
                            for k in ["Konsumsi_Beras_Ton", "Luas_Panen_Ha", "Produksi_Gabah_Ton", "Produksi_Beras_Ton"]})
            fig = go.Figure(go.Bar(x=cv.values, y=cv.index, orientation="h",
                                   marker_color=[BPS, TEAL, PADI_TUA, PADI], text=[fmt(v, 1) + "%" for v in cv.values],
                                   textposition="outside", hovertemplate="%{y}: %{x:,.1f}%<extra></extra>"))
            gaya(fig, 270, legend=False)
            fig.update_xaxes(range=[0, cv.max() * 1.18], title_text="koefisien variasi (%)")
            tampil(fig)
            st.caption("Produksi gabah dan beras berfluktuasi sama persis karena beras dihitung dari gabah dengan angka "
                       "konversi tetap. Model regresi cukup memakai produksi beras.")

    with tab[1]:
        kons_rata = DF["Konsumsi_Beras_Ton"].mean()
        c1, c2 = st.columns([1, 1.3], gap="medium")
        with c1:
            with kartu():
                kepala_kartu("Jam panen Bojonegoro", "schedule", PADI_TUA,
                             "Panjang jari-jari adalah rata-rata produksi tiap bulan. Lingkaran biru adalah kebutuhan.")
                rata = REKAP_BULAN["rata_prod"]
                warna = [MERAH if v < kons_rata else (PADI if v > 40000 else "#F6D27A") for v in rata.values]
                fig = go.Figure()
                fig.add_trace(go.Barpolar(r=rata.values, theta=SINGKAT, marker_color=warna, marker_line_color="white",
                                          marker_line_width=2, hovertemplate="%{theta}: %{r:,.0f} ton<extra></extra>"))
                fig.add_trace(go.Scatterpolar(r=[kons_rata] * 13, theta=SINGKAT + [SINGKAT[0]], mode="lines",
                                              line=dict(color=BPS, width=3), hovertemplate="Kebutuhan %{r:,.0f} ton<extra></extra>"))
                gaya(fig, 430, legend=False)
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                             angularaxis=dict(direction="clockwise", rotation=90, gridcolor=GARIS, linecolor=GARIS),
                                             radialaxis=dict(showticklabels=False, gridcolor=GARIS, linecolor="rgba(0,0,0,0)")))
                tampil(fig)
        with c2:
            with kartu():
                kepala_kartu("Sebaran produksi per bulan", "candlestick_chart", TEAL,
                             "Kotak menunjukkan rentang tiga tahun, titik berwarna adalah nilai tiap tahun.")
                fig = go.Figure()
                fig.add_trace(go.Box(x=DF["Bulan_Singkat"], y=DF["Produksi_Beras_Ton"], boxpoints=False,
                                     fillcolor="rgba(242,176,30,0.22)", line=dict(color=PADI_TUA, width=1.5),
                                     showlegend=False, hoverinfo="skip"))
                for th in TAHUN:
                    d = DF[DF["Tahun"] == th]
                    fig.add_trace(go.Scatter(x=d["Bulan_Singkat"], y=d["Produksi_Beras_Ton"], mode="markers", name=str(th),
                                             marker=dict(size=10, color=warna_tahun(th, TAHUN), line=dict(color="white", width=1.5)),
                                             hovertemplate=f"%{{x}} {th}<br>%{{y:,.0f}} ton<extra></extra>"))
                fig.add_hline(y=kons_rata, line_dash="dash", line_color=BPS, line_width=1.5,
                              annotation_text="kebutuhan", annotation_font_color=BPS)
                gaya(fig, 430)
                fig.update_xaxes(type="category", categoryorder="array", categoryarray=SINGKAT)
                tampil(fig)
        with kartu():
            kepala_kartu("Rata-rata per bulan", "calendar_view_month", BPS)
            tabel(pd.DataFrame({
                "Bulan": BULAN,
                "Rata-rata produksi (ton)": REKAP_BULAN["rata_prod"].map(fmt).values,
                "Rata-rata konsumsi (ton)": REKAP_BULAN["rata_kons"].map(fmt).values,
                "Rata-rata surplus (ton)": REKAP_BULAN["rata_defisit"].map(fmt).values,
                "Kali defisit": [f"{int(v)} dari {len(TAHUN)}" for v in REKAP_BULAN["n_defisit"].values],
            }))

    with tab[2]:
        c1, c2 = st.columns([1.5, 1], gap="medium")
        with c1:
            with kartu():
                kepala_kartu("Bentuk musim yang berulang", "stacked_line_chart", TEAL,
                             "Ketiga tahun memiliki dua puncak panen. Pada 2023 puncak pertama bergeser ke Februari.")
                fig = go.Figure()
                for th in TAHUN:
                    d = DF[DF["Tahun"] == th]
                    fig.add_trace(go.Scatter(x=d["Bulan_Singkat"], y=d["Produksi_Beras_Ton"], name=str(th), mode="lines+markers",
                                             line=dict(color=warna_tahun(th, TAHUN), width=3, shape="spline", smoothing=0.5),
                                             marker=dict(size=7), hovertemplate=f"%{{x}} {th}<br>%{{y:,.0f}} ton<extra></extra>"))
                fig.add_trace(go.Scatter(x=SINGKAT, y=REKAP_BULAN["rata_kons"], name="Kebutuhan", mode="lines",
                                         line=dict(color=MERAH, dash="dot", width=2)))
                gaya(fig, 390)
                fig.update_xaxes(type="category", categoryorder="array", categoryarray=SINGKAT)
                tampil(fig)
        with c2:
            with kartu():
                kepala_kartu("Total per tahun", "leaderboard", PADI_TUA)
                th_df = DF.groupby("Tahun")[["Produksi_Beras_Ton", "Konsumsi_Beras_Ton"]].sum()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=th_df.index.astype(str), y=th_df["Produksi_Beras_Ton"], name="Produksi", marker_color=PADI,
                                     text=[fmt(v / 1000) + " rb" for v in th_df["Produksi_Beras_Ton"]], textposition="outside"))
                fig.add_trace(go.Bar(x=th_df.index.astype(str), y=th_df["Konsumsi_Beras_Ton"], name="Konsumsi", marker_color=BPS,
                                     text=[fmt(v / 1000) + " rb" for v in th_df["Konsumsi_Beras_Ton"]], textposition="outside"))
                gaya(fig, 390)
                fig.update_layout(barmode="group", bargap=0.3)
                fig.update_yaxes(range=[0, th_df["Produksi_Beras_Ton"].max() * 1.2])
                tampil(fig)
        rek = DF.groupby("Tahun").agg(Luas=("Luas_Panen_Ha", "sum"), Gabah=("Produksi_Gabah_Ton", "sum"),
                                      Produksi=("Produksi_Beras_Ton", "sum"), Konsumsi=("Konsumsi_Beras_Ton", "sum"),
                                      Defisit=("Status", lambda s: (s == "Defisit").sum()))
        rek["Produktivitas"] = rek["Gabah"] * 10 / rek["Luas"]
        ubah = rek[["Produksi", "Konsumsi"]].pct_change() * 100
        with kartu():
            kepala_kartu("Perbandingan tahunan", "compare_arrows", BPS,
                         "Luas panen menyusut tipis, tetapi produktivitas per hektar naik sehingga produksi tetap bertambah.")
            tabel(pd.DataFrame({
                "Tahun": rek.index.astype(str), "Luas panen (ha)": rek["Luas"].map(fmt).values,
                "Produktivitas (ku/ha)": rek["Produktivitas"].map(lambda v: fmt(v, 2)).values,
                "Produksi (ton)": rek["Produksi"].map(fmt).values,
                "Perubahan produksi": ["" if np.isnan(v) else persen(v) for v in ubah["Produksi"]],
                "Konsumsi (ton)": rek["Konsumsi"].map(fmt).values,
                "Perubahan konsumsi": ["" if np.isnan(v) else persen(v) for v in ubah["Konsumsi"]],
                "Rasio swasembada": (rek["Produksi"] / rek["Konsumsi"] * 100).map(lambda v: fmt(v, 1) + "%").values,
                "Bulan defisit": rek["Defisit"].astype(int).values,
            }))

    with tab[3]:
        prod = DF["Produksi_Beras_Ton"]
        lg = np.log(prod)
        sw_a, sw_l = stats.shapiro(prod), stats.shapiro(lg)
        status = lambda p: '<span class="naik">normal</span>' if p > 0.05 else '<span class="turun">tidak normal</span>'
        kartu_kpi([
            dict(label="Kemencengan data asli", nilai=fmt(prod.skew(), 2), ikon="ssid_chart", warna=PADI_TUA, catatan="menceng ke kanan"),
            dict(label="Kemencengan setelah log", nilai=fmt(lg.skew(), 2), ikon="balance", warna=TEAL, catatan="lebih simetris"),
            dict(label="Shapiro-Wilk data asli", nilai="p = " + fmt(sw_a.pvalue, 4), ikon="science", warna=MERAH, catatan=status(sw_a.pvalue)),
            dict(label="Shapiro-Wilk setelah log", nilai="p = " + fmt(sw_l.pvalue, 4), ikon="verified", warna=HIJAU, catatan=status(sw_l.pvalue)),
        ])
        with kartu():
            kepala_kartu("Mengapa memakai logaritma", "functions", BPS,
                         "Bulan panen raya membuat ekor panjang ke kanan. Transformasi logaritma natural meratakan sebarannya.")
            fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.17,
                                subplot_titles=["Histogram produksi beras", "Q-Q plot produksi beras",
                                                "Histogram ln(produksi beras)", "Q-Q plot ln(produksi beras)"])
            for baris, (seri, warna) in enumerate([(prod, PADI), (lg, TEAL)], start=1):
                fig.add_trace(go.Histogram(x=seri, nbinsx=12, marker_color=warna, marker_line_color="white", marker_line_width=1.5), row=baris, col=1)
                (osm, osr), (slope, intercept, _) = stats.probplot(seri, dist="norm")
                fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=warna, size=8, line=dict(color="white", width=1))), row=baris, col=2)
                fig.add_trace(go.Scatter(x=osm, y=slope * np.asarray(osm) + intercept, mode="lines", line=dict(color=MERAH, width=2)), row=baris, col=2)
            gaya(fig, 600, legend=False)
            fig.update_annotations(font=dict(family=FONT, size=13, color=TEKS))
            tampil(fig)

    with tab[4]:
        r_p, _ = stats.pearsonr(DF["Luas_Panen_Ha"], DF["Produksi_Beras_Ton"])
        r_s, _ = stats.spearmanr(DF["Luas_Panen_Ha"], DF["Produksi_Beras_Ton"])
        prodv = DF.groupby("Tahun")["Produksi_Gabah_Ton"].sum() * 10 / DF.groupby("Tahun")["Luas_Panen_Ha"].sum()
        kartu_kpi([
            dict(label="Korelasi Pearson", nilai=fmt(r_p, 4), ikon="link", warna=TEAL, catatan="hubungan linear sangat kuat"),
            dict(label="Korelasi Spearman", nilai=fmt(r_s, 4), ikon="sort", warna=BPS, catatan="berdasarkan peringkat"),
            dict(label=f"Produktivitas {TAHUN[-1]}", nilai=fmt(prodv.iloc[-1], 2), satuan="ku/ha", ikon="eco", warna=HIJAU,
                 catatan=f"{persen((prodv.iloc[-1] / prodv.iloc[0] - 1) * 100)} sejak {TAHUN[0]}"),
        ])
        c1, c2 = st.columns([1.6, 1], gap="medium")
        with c1:
            with kartu():
                kepala_kartu("Produksi mengikuti luas panen", "scatter_plot", TEAL,
                             "Pola musiman produksi pada dasarnya adalah pola kapan petani memanen.")
                fig = go.Figure()
                for th in TAHUN:
                    d = DF[DF["Tahun"] == th]
                    fig.add_trace(go.Scatter(x=d["Luas_Panen_Ha"], y=d["Produksi_Beras_Ton"], mode="markers", name=str(th), text=d["Label"],
                                             marker=dict(size=12, color=warna_tahun(th, TAHUN), line=dict(color="white", width=1.5)),
                                             hovertemplate="%{text}<br>%{x:,.0f} ha<br>%{y:,.0f} ton<extra></extra>"))
                k = np.polyfit(DF["Luas_Panen_Ha"], DF["Produksi_Beras_Ton"], 1)
                xs = np.linspace(0, DF["Luas_Panen_Ha"].max() * 1.04, 50)
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(k, xs), mode="lines", name="Garis tren", line=dict(color="#9FB3BC", dash="dash")))
                gaya(fig, 410)
                fig.update_xaxes(title_text="luas panen (ha)")
                fig.update_yaxes(title_text="produksi beras (ton)")
                tampil(fig)
        with c2:
            with kartu():
                kepala_kartu("Produktivitas padi per tahun", "eco", HIJAU)
                fig = go.Figure(go.Bar(x=prodv.index.astype(str), y=prodv.values, marker_color=[warna_tahun(t, TAHUN) for t in prodv.index],
                                       text=[fmt(v, 2) for v in prodv.values], textposition="outside"))
                gaya(fig, 410, legend=False)
                fig.update_yaxes(range=[prodv.min() * 0.92, prodv.max() * 1.04], title_text="kuintal gabah per ha")
                tampil(fig)

    with tab[5]:
        prod = DF["Produksi_Beras_Ton"]
        q1, q3 = prod.quantile([0.25, 0.75])
        batas = q3 + 1.5 * (q3 - q1)
        ekstrem = DF[prod > batas]
        selisih = (prod - DF.groupby("Bulan_No")["Produksi_Beras_Ton"].transform("mean")) / \
                  DF.groupby("Bulan_No")["Produksi_Beras_Ton"].transform("mean") * 100
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            with kartu():
                kepala_kartu("Metode IQR", "radar", MERAH, f"{len(ekstrem)} bulan di atas batas {fmt(batas)} ton.")
                fig = go.Figure(go.Box(y=prod, boxpoints="outliers", fillcolor="rgba(242,176,30,0.22)", line=dict(color=PADI_TUA),
                                       marker=dict(color=MERAH, size=9), name="", text=DF["Label"],
                                       hovertemplate="%{text}<br>%{y:,.0f} ton<extra></extra>"))
                gaya(fig, 390, legend=False)
                tampil(fig)
        with c2:
            with kartu():
                kepala_kartu("Selisih dari rata-rata bulan yang sama", "difference", ORANYE,
                             "Merah berarti menyimpang lebih dari 50% dari pola bulannya. Penyimpangan terbesar ada di 2023.")
                fig = go.Figure(go.Bar(x=DF["Label"], y=selisih, marker_color=[MERAH if abs(v) > 50 else "#B9CBD3" for v in selisih],
                                       hovertemplate="%{x}<br>%{y:,.1f}%<extra></extra>"))
                fig.add_hline(y=50, line_dash="dot", line_color=MERAH, line_width=1)
                fig.add_hline(y=-50, line_dash="dot", line_color=MERAH, line_width=1)
                gaya(fig, 390, legend=False)
                sumbu_bulan(fig, DF["Label"].tolist(), 3)
                tampil(fig)
        st.caption("Semua nilai ekstrem dipertahankan karena terkonfirmasi dari publikasi BPS dan konsisten dengan luas panen.")


# =====================================================================
# HALAMAN 3: MODEL REGRESI
# =====================================================================
def halaman_model():
    m1, m2 = A["model1"], A["model2"]
    hero(f"Trend dan musim menjelaskan {fmt(m2.rsquared_adj * 100, 0)}% variasi produksi",
         "Model memperkirakan produksi setiap bulan dari posisinya dalam waktu (trend) dan bulan apa ia jatuh "
         "(musim), dengan Januari sebagai bulan pembanding.", tag="Regresi deret waktu", ringkas=True)

    with kartu():
        kepala_kartu("Persamaan model final", "function", BPS)
        st.latex(r"\ln(Y_t) = \beta_0 + \beta_1\,t + \sum_{j=2}^{12} \beta_j\,D_{j,t} + \varepsilon_t")
        st.caption("Y = produksi beras bulan ke-t, t = 1 sampai 36, D = dummy bulan Februari sampai Desember.")

    def panel(nama, sub, model, uji, final):
        baris = ""
        for _, r in uji.iterrows():
            nilai = f"nilai {fmt(r['Statistik'], 3)}" if np.isnan(r["p_value"]) else f"p = {fmt(r['p_value'], 4)}"
            chip = '<span class="chip chip-ya">Terpenuhi</span>' if r["Terpenuhi"] else '<span class="chip chip-tidak">Tidak</span>'
            baris += f"<tr><td>{r['Uji']}<br><small>{r['Asumsi']}, {nilai}</small></td><td>{chip}</td></tr>"
        latar = "linear-gradient(120deg,#0E9F8E,#0B3C78)" if final else "linear-gradient(120deg,#8A9BA6,#5F7480)"
        lencana = '<span class="lencana">Model final</span>' if final else ""
        return f"""
        <div class="panel-model">
            <div class="panel-kepala" style="background:{latar}"><div><b>{nama}</b><small>{sub}</small></div>{lencana}</div>
            <div class="panel-badan">
                <div class="panel-angka">
                    <div><div>R²</div><div>{fmt(model.rsquared, 3)}</div></div>
                    <div><div>Adjusted R²</div><div>{fmt(model.rsquared_adj, 3)}</div></div>
                    <div><div>Uji F</div><div>p &lt; 0,001</div></div>
                </div>
                <table class="tabel-uji">{baris}</table>
            </div>
        </div>"""

    c1, c2 = st.columns(2, gap="medium")
    md(panel("Model 1", "Produksi beras dalam ton", m1, A["uji1"], False), c1)
    md(panel("Model 2", "Logaritma natural produksi beras", m2, A["uji2"], True), c2)
    st.caption("R² kedua model tidak dibandingkan langsung karena skala variabel responsnya berbeda. "
               "Model 2 dipilih karena memenuhi seluruh asumsi klasik.")

    koef = A["koef"]
    dummy = koef[koef["Variabel"].str.startswith("D_")].copy()
    dummy["Bulan"] = dummy["Variabel"].str.replace("D_", "", regex=False)
    tab = st.tabs(["Pengaruh tiap bulan", "Aktual dan hasil model", "Diagnostik residual", "Tabel koefisien"])

    with tab[0]:
        with kartu():
            kepala_kartu("Produksi setiap bulan dibanding Januari", "bar_chart", TEAL,
                         "Warna teal berarti signifikan pada taraf 5%. Garis tipis adalah selang kepercayaan 95%.")
            fig = go.Figure(go.Bar(
                y=dummy["Bulan"], x=dummy["Persen"], orientation="h",
                marker_color=[TEAL if p < 0.05 else "#C3D2D9" for p in dummy["p_value"]],
                error_x=dict(type="data", symmetric=False, array=dummy["Persen_Atas"] - dummy["Persen"],
                             arrayminus=dummy["Persen"] - dummy["Persen_Bawah"], color="#7D919B", thickness=1, width=4),
                customdata=np.stack([dummy["p_value"], dummy["Koefisien"]], axis=1),
                hovertemplate="%{y}: %{x:,.0f}% dibanding Januari<br>koefisien %{customdata[1]:.3f}, p = %{customdata[0]:.4f}<extra></extra>"))
            fig.add_vline(x=0, line_color="#9FB3BC", line_width=1)
            gaya(fig, 460, legend=False)
            fig.update_yaxes(autorange="reversed", type="category")
            fig.update_xaxes(title_text="perubahan terhadap Januari (%)")
            tampil(fig)
            tt = koef.set_index("Variabel").loc["t"]
            st.caption(f"Trend (t): koefisien {fmt(tt['Koefisien'], 4)}, p = {fmt(tt['p_value'], 3)}. Tidak ada kenaikan "
                       "atau penurunan produksi yang signifikan setelah pola musim dikendalikan.")

    with tab[1]:
        with kartu():
            kepala_kartu("Seberapa dekat model dengan kenyataan", "insights", PADI_TUA,
                         "Selisih terbesar ada di 2023 ketika kalender panen bergeser. Klik Model 1 di legenda untuk membandingkan.")
            label = DF["Label"].tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=label, y=DF["Produksi_Beras_Ton"], name="Aktual", mode="lines+markers",
                                     line=dict(color=PADI, width=3), marker=dict(size=7), fill="tozeroy", fillcolor="rgba(242,176,30,0.12)"))
            fig.add_trace(go.Scatter(x=label, y=np.exp(m2.fittedvalues) * A["smearing"], name="Model 2", mode="lines",
                                     line=dict(color=TEAL, width=2.6, dash="dash")))
            fig.add_trace(go.Scatter(x=label, y=m1.fittedvalues, name="Model 1", mode="lines",
                                     line=dict(color="#9FB3BC", width=1.8, dash="dot"), visible="legendonly"))
            gaya(fig, 420)
            sumbu_bulan(fig, label, 3)
            fig.update_traces(hovertemplate="%{x}<br>%{y:,.0f} ton")
            tampil(fig)

    with tab[2]:
        with kartu():
            kepala_kartu("Diagnostik residual", "monitor_heart", MERAH,
                         "Residual yang baik menyebar acak, mengikuti garis Q-Q, dan batang ACF berada di antara garis merah.")
            pilihan = pilih("Model", ["Model 2 (final)", "Model 1"], "Model 2 (final)", "diag_model")
            model = m2 if pilihan.startswith("Model 2") else m1
            resid = model.resid.values
            fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.18,
                                subplot_titles=["Residual dan nilai prediksi", "Q-Q plot residual",
                                                "Residual menurut waktu", "Autokorelasi residual (ACF)"])
            fig.add_trace(go.Scatter(x=model.fittedvalues, y=resid, mode="markers", text=DF["Label"],
                                     marker=dict(color=TEAL, size=9, line=dict(color="white", width=1)),
                                     hovertemplate="%{text}<br>residual %{y:,.3f}<extra></extra>"), row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color=MERAH, row=1, col=1)
            (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=BPS, size=8)), row=1, col=2)
            fig.add_trace(go.Scatter(x=osm, y=slope * np.asarray(osm) + intercept, mode="lines", line=dict(color=MERAH, width=2)), row=1, col=2)
            fig.add_trace(go.Bar(x=DF["Label"], y=resid, marker_color=np.where(resid > 0, HIJAU, MERAH),
                                 hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>"), row=2, col=1)
            nilai_acf = acf(resid, nlags=12)
            batas = 1.96 / np.sqrt(len(resid))
            fig.add_trace(go.Bar(x=list(range(1, 13)), y=nilai_acf[1:], marker_color=TEAL, width=0.45,
                                 hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"), row=2, col=2)
            for b in (batas, -batas):
                fig.add_hline(y=b, line_dash="dot", line_color=MERAH, line_width=1, row=2, col=2)
            gaya(fig, 640, legend=False)
            fig.update_xaxes(type="category", tickmode="array", tickvals=DF["Label"].tolist()[::6], row=2, col=1)
            fig.update_annotations(font=dict(family=FONT, size=13, color=TEKS))
            tampil(fig)

    with tab[3]:
        with kartu():
            kepala_kartu("Koefisien Model 2", "table_rows", BPS,
                         "Persentase dihitung dengan (e^β − 1) × 100% karena variabel respons dalam bentuk logaritma.")
            nama = {"const": "Konstanta", "t": "Trend (t)"}
            nama.update({f"D_{s}": f"Dummy {b}" for s, b in zip(SINGKAT, BULAN)})
            tabel(pd.DataFrame({
                "Variabel": koef["Variabel"].map(nama),
                "Koefisien": koef["Koefisien"].map(lambda v: fmt(v, 4)),
                "Std. error": koef["Std_Error"].map(lambda v: fmt(v, 4)),
                "t hitung": koef["t_hitung"].map(lambda v: fmt(v, 3)),
                "p-value": koef["p_value"].map(lambda v: fmt(v, 4)),
                "Dibanding Januari": ["" if v in ("const", "t") else fmt(p, 1) + "%" for v, p in zip(koef["Variabel"], koef["Persen"])],
                "Signifikan (5%)": np.where(koef["p_value"] < 0.05, "Ya", "Tidak"),
            }))


# =====================================================================
# HALAMAN 4: PROYEKSI
# =====================================================================
def halaman_proyeksi():
    th = A["tahun_proy"]
    slot_hero = st.empty()
    with kartu():
        kepala_kartu("Simulasi skenario", "tune", TEAL,
                     "Geser untuk melihat dampak perubahan kebutuhan atau hasil panen terhadap risiko defisit. "
                     "Posisi nol adalah proyeksi dasar dari model.")
        s1, s2 = st.columns(2, gap="large")
        naik_k = s1.slider("Perubahan kebutuhan konsumsi (%)", -10, 20, 0, 1)
        ubah_p = s2.slider("Perubahan hasil produksi (%)", -30, 30, 0, 1)

    P = hitung_proyeksi(A, naik_k, ubah_p)
    rawan = P[P["Kategori"] == "Rawan"]
    skenario = "proyeksi dasar" if (naik_k == 0 and ubah_p == 0) else "skenario yang dipilih"
    if len(rawan):
        hero(f"{gabung(rawan['Bulan'].tolist())} perlu disiapkan sejak awal {th}",
             f"Dengan {skenario}, bulan-bulan tersebut berpeluang lebih dari 50% tidak tercukupi oleh produksi sendiri.",
             tag=f"Peringatan dini {th}", ringkas=True, wadah=slot_hero)
    else:
        hero(f"Tidak ada bulan rawan defisit pada {th}",
             f"Dengan {skenario}, peluang defisit setiap bulan berada di bawah 50%.", tag=f"Peringatan dini {th}", ringkas=True,
             wadah=slot_hero)

    n_waspada = int((P["Kategori"] == "Waspada").sum())
    kartu_kpi([
        dict(label="Proyeksi produksi", nilai=fmt(P["Produksi"].sum()), satuan="ton", ikon="agriculture", warna=PADI_TUA,
             grafik=f'<div class="kpi-grafik"><img src="{svg_sparkline(P["Produksi"], PADI)}"></div>'),
        dict(label="Proyeksi konsumsi", nilai=fmt(P["Konsumsi"].sum()), satuan="ton", ikon="rice_bowl", warna=BPS,
             grafik=f'<div class="kpi-grafik"><img src="{svg_sparkline(P["Konsumsi"], BPS)}"></div>'),
        dict(label="Proyeksi surplus", nilai=fmt(P["Surplus"].sum()), satuan="ton", ikon="inventory_2", warna=HIJAU,
             grafik=f'<div class="kpi-grafik"><img src="{svg_batang_mini(P["Surplus"], HIJAU, MERAH)}"></div>'),
        dict(label="Bulan rawan", ikon="warning", warna=MERAH,
             badan=f'<div class="kpi-ring"><div class="ring"><img src="{svg_cincin(len(rawan) / 12 * 100, MERAH, "#FBE7E7")}">'
                   f'<span>{len(rawan)}</span></div><div><div class="kpi-nilai" style="margin-top:0">{len(rawan)}<small>dari 12</small></div>'
                   f'<div class="kpi-catatan">waspada {n_waspada} bulan</div></div></div>'),
    ])

    with kartu():
        kepala_kartu(f"Kalender risiko pangan {th}", "event_upcoming", MERAH,
                     "Angka besar adalah peluang produksi bulan itu lebih kecil dari kebutuhan. Aman di bawah 20%, "
                     "waspada 20 sampai 50%, rawan di atas 50%.")
        sel = ""
        for _, r in P.iterrows():
            w = WARNA_KATEGORI[r["Kategori"]]
            sel += f"""
            <div class="bln" style="background:linear-gradient(160deg,{tint(w, .16)},{tint(w, .05)});color:{w}">
                <div class="bln-atas"><span class="bln-nama" style="color:{TEKS}">{r['Bulan']}</span>{ikon(IKON_KATEGORI[r['Kategori']], w)}</div>
                <div class="bln-persen">{fmt(r['Peluang'], 0)}%</div>
                <div class="bln-meter"><span style="width:{max(r['Peluang'], 3):.0f}%;background:{w}"></span></div>
                <div class="bln-ket">{r['Kategori']}, {'surplus' if r['Surplus'] > 0 else 'defisit'} {fmt(abs(r['Surplus']))} ton</div>
            </div>"""
        md(f'<div class="kalender">{sel}</div>')

    with kartu():
        kepala_kartu("Produksi aktual dan proyeksi", "timeline", PADI_TUA,
                     "Pita kuning adalah rentang 95% tempat produksi sebenarnya kemungkinan besar berada.")
        lab_h, lab_p = DF["Label"].tolist(), P["Label"].tolist()
        semua = lab_h + lab_p
        batas_y = max(DF["Produksi_Beras_Ton"].max(), P["Produksi"].max()) * 1.3
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lab_p, y=np.minimum(P["Atas"], batas_y), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lab_p, y=P["Bawah"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(242,176,30,0.25)", name="Rentang 95%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lab_h, y=DF["Produksi_Beras_Ton"], name="Produksi aktual", mode="lines+markers",
                                 line=dict(color=TEAL, width=2.6), marker=dict(size=6), hovertemplate="%{x}<br>%{y:,.0f} ton<extra></extra>"))
        fig.add_trace(go.Scatter(x=lab_p, y=P["Produksi"], name=f"Proyeksi {th}", mode="lines+markers",
                                 line=dict(color=PADI_TUA, width=3), marker=dict(size=8, color=PADI, line=dict(color="white", width=1.5)),
                                 customdata=np.stack([P["Bawah"], P["Atas"]], axis=1),
                                 hovertemplate="%{x}<br>%{y:,.0f} ton<br>rentang %{customdata[0]:,.0f} sampai %{customdata[1]:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=semua, y=list(DF["Konsumsi_Beras_Ton"]) + list(P["Konsumsi"]), name="Kebutuhan konsumsi",
                                 mode="lines", line=dict(color=MERAH, width=2, dash="dash"), hovertemplate="%{x}<br>kebutuhan %{y:,.0f} ton<extra></extra>"))
        fig.add_vrect(x0=len(lab_h) - 0.5, x1=len(semua) - 0.5, fillcolor=PADI, opacity=0.05, line_width=0, layer="below")
        fig.add_annotation(x=len(lab_h) + 0.2, y=batas_y * 0.95, text=f"Proyeksi {th}", showarrow=False, xanchor="left",
                           font=dict(color=PADI_TUA, size=13))
        gaya(fig, 440)
        sumbu_bulan(fig, semua, 3)
        fig.update_yaxes(range=[0, batas_y])
        tampil(fig)
        if (P["Atas"] > batas_y).any():
            st.caption("Batas atas rentang pada bulan panen raya dipotong agar grafik tetap terbaca.")

    with kartu():
        kepala_kartu(f"Peta surplus dan defisit {TAHUN[0]}–{th}", "grid_on", HIJAU,
                     "Merah berarti defisit, hijau berarti surplus. Skala warna dibatasi ±10 ribu ton agar bulan sepi panen terlihat jelas.")
        peta = DF.pivot(index="Tahun", columns="Bulan_No", values="Surplus_Defisit_Ton")
        peta.loc[th] = P.set_index("Bulan_No")["Surplus"]
        nama_baris = [str(t) for t in peta.index[:-1]] + [f"{th} (proyeksi)"]
        z = peta.values
        fig = go.Figure(go.Heatmap(z=z, x=SINGKAT, y=nama_baris, zmid=0, zmin=-10000, zmax=10000,
                                   colorscale=[[0, MERAH], [0.4, "#FAD2D2"], [0.5, "#FFFFFF"], [0.6, "#CFEFDF"], [1, HIJAU]],
                                   xgap=4, ygap=4, colorbar=dict(title=dict(text="ton", side="top"), thickness=12, len=0.9),
                                   hovertemplate="%{x} %{y}<br>%{z:,.0f} ton<extra></extra>"))
        for i, nb in enumerate(nama_baris):
            for j in range(12):
                v = z[i, j]
                fig.add_annotation(x=SINGKAT[j], y=nb, text=f"{fmt(v / 1000, 1)} rb", showarrow=False,
                                   font=dict(size=11, color="white" if abs(v) > 6000 else TEKS))
        gaya(fig, 300, legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(side="top")
        tampil(fig)

    with kartu():
        kepala_kartu("Tabel proyeksi", "table_rows", BPS)
        tabel(pd.DataFrame({
            "Bulan": P["Bulan"], "Produksi (ton)": P["Produksi"].map(fmt),
            "Rentang 95% (ton)": [f"{fmt(a)} sampai {fmt(b)}" for a, b in zip(P["Bawah"], P["Atas"])],
            "Konsumsi (ton)": P["Konsumsi"].map(fmt), "Surplus/defisit (ton)": P["Surplus"].map(fmt),
            "Peluang defisit": P["Peluang"].round(0), "Kategori": P["Kategori"],
        }), column_config={"Peluang defisit": st.column_config.ProgressColumn("Peluang defisit", min_value=0, max_value=100, format="%.0f%%")})
        unduh = P[["Bulan", "Produksi", "Bawah", "Atas", "Konsumsi", "Surplus", "Peluang", "Kategori"]].round(1)
        unduh.columns = ["Bulan", "Produksi_Ton", "Batas_Bawah_95", "Batas_Atas_95", "Konsumsi_Ton",
                         "Surplus_Defisit_Ton", "Peluang_Defisit_Persen", "Kategori"]
        st.download_button("Unduh tabel proyeksi (CSV)", unduh.to_csv(index=False).encode("utf-8"),
                           file_name=f"proyeksi_neraca_beras_{th}.csv", mime="text/csv")


# =====================================================================
# HALAMAN 5: METODOLOGI
# =====================================================================
def halaman_metodologi():
    hero("Cara angka-angka ini dihasilkan",
         f"Seluruh analisis memakai data resmi BPS Kabupaten Bojonegoro tahun {RENTANG}.", tag="Metodologi", ringkas=True)
    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        with kartu():
            kepala_kartu("Tahapan analisis", "account_tree", TEAL)
            langkah = [
                ("Penyusunan data", "Data bulanan luas panen, produksi gabah, produksi beras, dan konsumsi beras disusun "
                                    f"menjadi satu baris per bulan, sebanyak {len(DF)} pengamatan."),
                ("Eksplorasi data", "Statistik deskriptif, pola musiman, perbandingan antar tahun, sebaran, hubungan luas "
                                    "panen dengan produksi, dan pemeriksaan nilai ekstrem."),
                ("Regresi deret waktu", "Variabel trend dan 11 dummy bulan, dengan Januari sebagai pembanding."),
                ("Uji asumsi klasik", "Shapiro-Wilk dan Jarque-Bera untuk normalitas, Breusch-Pagan untuk homoskedastisitas, "
                                      "Durbin-Watson dan Breusch-Godfrey untuk autokorelasi, serta VIF untuk multikolinearitas."),
                ("Transformasi logaritma", "Produksi ditransformasi logaritma natural agar seluruh asumsi terpenuhi."),
                ("Proyeksi dan peluang defisit", f"Produksi {A['tahun_proy']} diproyeksikan dengan koreksi smearing Duan, "
                                                 "konsumsi dengan regresi trend, lalu peluang defisit dihitung dari sebaran prediksi."),
            ]
            md("".join(f'<div class="langkah"><div class="langkah-no">{i}</div><div><b>{j}</b><span>{k}</span></div></div>'
                       for i, (j, k) in enumerate(langkah, start=1)))
    with c2:
        with kartu():
            kepala_kartu("Sumber data", "database", BPS,
                         f"Badan Pusat Statistik Kabupaten Bojonegoro, data bulanan {RENTANG}. Surplus atau defisit "
                         "dihitung sebagai produksi beras dikurangi konsumsi beras.")
        with kartu():
            kepala_kartu("Kategori risiko", "flag", ORANYE,
                         "Aman bila peluang defisit paling tinggi 20%, waspada bila lebih dari 20% sampai 50%, rawan bila "
                         "lebih dari 50%. Ambang ini ditetapkan sebagai patokan pemantauan, bukan ketentuan baku.")
        with kartu():
            kepala_kartu("Keterbatasan", "info", MERAH,
                         f"Model belajar dari {len(TAHUN)} siklus musim sehingga rentang proyeksi lebar. Dummy bulan "
                         "mengasumsikan pola musim yang sama setiap tahun, sehingga pergeseran kalender tanam seperti pada "
                         "2023 tidak tertangkap. Angka proyeksi sebaiknya dibaca sebagai arah dan pola, bukan angka pasti.")


# =====================================================================
# NAVIGASI
# =====================================================================
if LOGO.exists():
    try:
        st.logo(str(LOGO), size="large")
    except TypeError:
        st.logo(str(LOGO))

halaman = st.navigation([
    st.Page(halaman_ringkasan, title="Ringkasan", icon=":material/space_dashboard:", url_path="ringkasan", default=True),
    st.Page(halaman_eda, title="Eksplorasi data", icon=":material/query_stats:", url_path="eksplorasi"),
    st.Page(halaman_model, title="Model regresi", icon=":material/function:", url_path="model"),
    st.Page(halaman_proyeksi, title=f"Proyeksi {A['tahun_proy']}", icon=":material/event_upcoming:", url_path="proyeksi"),
    st.Page(halaman_metodologi, title="Metodologi", icon=":material/menu_book:", url_path="metodologi"),
])

st.sidebar.markdown(
    f'<div class="sidebar-info"><b>Neraca Beras Bojonegoro</b><br>Sumber data BPS Kabupaten Bojonegoro<br>'
    f'Periode data {RENTANG}</div>'
    f'<img src="{svg_sawah()}" alt="Ilustrasi sawah" style="width:100%;margin-top:18px;border-radius:16px;opacity:.95">',
    unsafe_allow_html=True)

halaman.run()