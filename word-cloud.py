#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persian (RTL) Word Cloud Generator with a basic UI (Streamlit)

Inputs:
- URL (web page)
- PDF file
- Plain text

Key RTL bits:
- Arabic/Persian shaping via arabic_reshaper
- Correct RTL display via python-bidi

Install:
  pip install streamlit wordcloud matplotlib arabic-reshaper python-bidi requests beautifulsoup4 trafilatura pdfplumber hazm

Run:
  streamlit run persian_wordcloud_ui.py

Notes:
- You MUST provide a Persian-capable TTF/OTF font file path (e.g., Vazirmatn.ttf, IRANSans.ttf, NotoNaskhArabic.ttf).
- WordCloud itself is not RTL-aware; shaping + bidi is what makes it look right.
"""

import io
import re
import tempfile
from pathlib import Path
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import requests
from bs4 import BeautifulSoup

import pdfplumber

import arabic_reshaper
from bidi.algorithm import get_display

Normalizer = None
_hazm_spec = importlib.util.find_spec("hazm")
if _hazm_spec:
    hazm = importlib.import_module("hazm")
    Normalizer = getattr(hazm, "Normalizer", None)

trafilatura = None
_trafilatura_spec = importlib.util.find_spec("trafilatura")
if _trafilatura_spec:
    trafilatura = importlib.import_module("trafilatura")


# -----------------------------
# Text utilities (Persian-focused)
# -----------------------------

DEFAULT_PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "آن", "برای", "با", "را", "یا", "اما", "اگر", "تا", "نیز",
    "بر", "هم", "هر", "خود", "شما", "ما", "من", "او", "ای", "یک", "دو", "سه", "چه", "چرا",
    "است", "بود", "باشد", "شد", "شود", "می", "نمی", "کرد", "کرده", "کن", "کنید", "کنیم",
    "همه", "هیچ", "چند", "بسیار", "بیش", "کم", "پس", "قبل", "بعد", "بین", "روی", "زیر",
    "همین", "همان", "چنین", "چطور", "کجا", "وقتی", "وقتی‌که", "زیرا", "چون",
}

PERSIAN_ARABIC_DIGITS_RE = re.compile(r"[0-9۰-۹]+")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PUNCT_RE = re.compile(r"[“”\"'`~!@#$%^&*\(\)\[\]\{\}\|\\:;,.?<>،؛؟«»…ـ+=]+")
EXTRA_SPACE_RE = re.compile(r"\s+")

# Keep Persian + Arabic letters (and joiners), optionally keep Latin for proper nouns.
# You can toggle later in UI.
ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u200c\s]+")
SAMPLE_TEXT = (
    "ایران سرزمینی با تاریخ کهن و فرهنگ پربار است. زبان فارسی در طول قرن‌ها "
    "گنجینه‌ای از ادبیات، شعر و اندیشه را به جهان هدیه داده است. مطالعه‌ی منابع "
    "گوناگون و گفت‌وگو درباره‌ی تجربه‌های مشترک، واژگان تازه و ایده‌های نو می‌آفریند."
)


def safe_request_get(url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PersianWordCloudBot/1.0; +https://example.com/bot)"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_text_from_url(url: str) -> str:
    html = safe_request_get(url)
    # Prefer trafilatura for cleaner article extraction if available
    if trafilatura is not None:
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=False)
        if downloaded and downloaded.strip():
            return downloaded.strip()

    # Fallback: BeautifulSoup text extraction
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    out = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
    return "\n".join(out).strip()


def normalize_persian(text: str, keep_latin: bool = True) -> str:
    t = text

    # Remove URLs/emails (often noise)
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)

    # Replace punctuation with spaces
    t = PUNCT_RE.sub(" ", t)

    # Remove digits
    t = PERSIAN_ARABIC_DIGITS_RE.sub(" ", t)

    # Remove odd characters
    if keep_latin:
        t = ALLOWED_CHARS_RE.sub(" ", t)
    else:
        t = re.sub(r"[^ \u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u200c\s]+", " ", t)

    # Normalize with Hazm if available (handles Arabic/Persian variants, spacing, etc.)
    if Normalizer is not None:
        t = Normalizer().normalize(t)

    # Collapse spaces
    t = EXTRA_SPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> list:
    # Basic tokenization: split on whitespace
    # WordCloud has its own tokenization too, but Persian often benefits from explicit cleanup.
    return [w.strip() for w in text.split() if w.strip()]


def rtl_shape(text: str) -> str:
    # Shape Arabic/Persian letters + make display RTL-correct for rendering contexts
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def build_frequencies(tokens: list, stopwords: set, min_len: int, max_words: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for w in tokens:
        if len(w) < min_len:
            continue
        if w in stopwords:
            continue
        freq[w] = freq.get(w, 0) + 1

    # Keep top max_words by frequency
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    items = items[:max_words]
    return dict(items)


# -----------------------------
# Word cloud generation
# -----------------------------

@dataclass
class WCOptions:
    font_path: str
    width: int = 1200
    height: int = 700
    background_color: str = "white"
    max_words: int = 200
    min_word_length: int = 2
    collocations: bool = False
    prefer_horizontal: float = 0.9
    scale: int = 2
    contour_width: int = 0
    contour_color: str = "black"
    mask_path: Optional[str] = None


def generate_persian_wordcloud(freq: Dict[str, int], opt: WCOptions) -> Tuple[WordCloud, plt.Figure]:
    # For RTL: reshape each token before feeding WordCloud
    shaped_freq = {rtl_shape(k): v for k, v in freq.items()}

    mask = None
    if opt.mask_path:
        mask = plt.imread(opt.mask_path)

    wc = WordCloud(
        font_path=opt.font_path,
        width=opt.width,
        height=opt.height,
        background_color=opt.background_color,
        max_words=opt.max_words,
        collocations=opt.collocations,
        prefer_horizontal=opt.prefer_horizontal,
        scale=opt.scale,
        contour_width=opt.contour_width,
        contour_color=opt.contour_color,
        mask=mask,
        # WordCloud's default regexp is Latin-centric; using frequencies bypasses it.
    ).generate_from_frequencies(shaped_freq)

    fig = plt.figure(figsize=(opt.width / 150, opt.height / 150))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return wc, fig


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Persian RTL Word Cloud", layout="wide")
st.title("Persian (RTL) Word Cloud Generator")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source type", ["URL", "PDF", "Plain text"], index=0)

    st.header("Cleaning / Language")
    keep_latin = st.checkbox("Keep Latin words (proper nouns, acronyms)", value=True)
    min_word_length = st.slider("Minimum word length", 1, 6, 2)

    custom_stopwords = st.text_area(
        "Custom stopwords (comma or newline separated)",
        value="",
        help="These will be added to built-in Persian stopwords.",
        height=120,
    )

    st.header("Word cloud")
    font_upload = st.file_uploader("Upload a Persian-capable font (TTF/OTF)", type=["ttf", "otf"])
    font_path = st.text_input(
        "Or provide a font path on disk",
        value="Vazirmatn-Regular.ttf",
        help="Example: Vazirmatn-Regular.ttf, IRANSans.ttf, NotoNaskhArabic-Regular.ttf",
    )
    max_words = st.slider("Max words", 50, 600, 200, step=25)
    background_color = st.selectbox("Background", ["white", "black"], index=0)
    width = st.select_slider("Width", options=[800, 1000, 1200, 1600, 2000], value=1200)
    height = st.select_slider("Height", options=[400, 600, 700, 900, 1200], value=700)
    prefer_horizontal = st.slider("Prefer horizontal words", 0.0, 1.0, 0.9, 0.05)
    collocations = st.checkbox("Enable collocations (bigrams)", value=False)

    advanced = st.expander("Advanced")
    with advanced:
        scale = st.slider("Render scale (sharper but slower)", 1, 5, 2)
        contour_width = st.slider("Contour width", 0, 5, 0)
        contour_color = st.text_input("Contour color", "black")
        mask_file = st.file_uploader("Optional mask image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Collect input text
text = ""
error = None
font_path_final = ""
mask_path = None

if mode == "URL":
    url = st.text_input("Paste a URL", value="")
    if st.button("Fetch & Generate", type="primary"):
        if not url.strip():
            error = "Please enter a URL."
        else:
            try:
                text = extract_text_from_url(url.strip())
            except Exception as e:
                error = f"Failed to fetch/extract text from URL: {e}"

elif mode == "PDF":
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Extract & Generate", type="primary"):
        if pdf_file is None:
            error = "Please upload a PDF."
        else:
            try:
                text = extract_text_from_pdf(pdf_file.read())
            except Exception as e:
                error = f"Failed to extract text from PDF: {e}"

else:  # Plain text
    if "plain_text" not in st.session_state:
        st.session_state.plain_text = ""
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        raw = st.text_area("Paste Persian text", value=st.session_state.plain_text, height=220)
        st.session_state.plain_text = raw
    with col_sample:
        if st.button("Load sample text"):
            st.session_state.plain_text = SAMPLE_TEXT
            raw = SAMPLE_TEXT
        if st.button("Clear"):
            st.session_state.plain_text = ""
            raw = ""
    if st.button("Generate", type="primary"):
        text = raw

if font_upload is not None:
    suffix = Path(font_upload.name).suffix or ".ttf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(font_upload.getvalue())
        font_path_final = tmp.name
else:
    font_path_final = font_path.strip()

if mask_file is not None:
    suffix = Path(mask_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(mask_file.getvalue())
        mask_path = tmp.name

if error:
    st.error(error)

if text:
    if len(text.strip()) < 20:
        st.warning("The extracted text is very short. Word clouds work better with more text.")

    # Stopwords
    sw = set(DEFAULT_PERSIAN_STOPWORDS)
    if custom_stopwords.strip():
        extra = re.split(r"[,;\n]+", custom_stopwords.strip())
        sw |= {w.strip() for w in extra if w.strip()}

    cleaned = normalize_persian(text, keep_latin=keep_latin)
    tokens = tokenize(cleaned)

    # Build frequency table
    freq = build_frequencies(tokens, stopwords=sw, min_len=min_word_length, max_words=max_words)

    if not freq:
        st.error("No usable words after cleaning/stopword removal. Try lowering stopwords or minimum length.")
    elif not font_path_final:
        st.error("Please provide a valid font path or upload a font file.")
    elif font_upload is None and not Path(font_path_final).exists():
        st.error("Font path does not exist. Upload a font file or provide a valid path.")
    else:
        opt = WCOptions(
            font_path=font_path_final,
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            min_word_length=min_word_length,
            collocations=collocations,
            prefer_horizontal=prefer_horizontal,
            scale=scale,
            contour_width=contour_width,
            contour_color=contour_color,
            mask_path=mask_path,
        )

        try:
            wc, fig = generate_persian_wordcloud(freq, opt)
        except Exception as e:
            st.error(
                "Word cloud generation failed. The most common issue is an invalid font path or a font without Persian glyphs.\n\n"
                f"Error: {e}"
            )
        else:
            col1, col2 = st.columns([2, 1], gap="large")

            with col1:
                st.subheader("Word Cloud")
                st.pyplot(fig, clear_figure=True)

                # Download PNG
                img_bytes = wc.to_image()
                buf = io.BytesIO()
                img_bytes.save(buf, format="PNG")
                st.download_button(
                    "Download PNG",
                    data=buf.getvalue(),
                    file_name="persian_wordcloud.png",
                    mime="image/png",
                )

            with col2:
                st.subheader("Diagnostics")
                st.write(f"Raw text length: {len(text):,} chars")
                st.write(f"Cleaned text length: {len(cleaned):,} chars")
                st.write(f"Token count: {len(tokens):,}")
                st.write(f"Unique words (after filters): {len(freq):,}")

                # Show top words (unshaped, for readability)
                topn = 25
                top_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:topn]
                st.markdown("Top words:")
                st.table([{"word": k, "count": v} for k, v in top_items])

                with st.expander("Preview cleaned text"):
                    st.text(cleaned[:4000] + ("..." if len(cleaned) > 4000 else ""))

                with st.expander("Tip: good free fonts"):
                    st.write(
                        "Try these fonts (download separately) and point 'Font path' to the .ttf:\n"
                        "- Vazirmatn (often best for Persian UI)\n"
                        "- Noto Naskh Arabic / Noto Sans Arabic\n"
                        "- IRANSans (if you have a licensed copy)\n"
                    )

