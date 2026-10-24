"""Product page scraper & rewriter (Streamlit + Crawl4AI + Google Gemini).

Run with:  python -m streamlit run app.py
"""
import json
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)  # must run before importing modules that read env vars

import llm  # noqa: E402
import scraper  # noqa: E402

st.set_page_config(page_title="Product Scraper & Rewriter", page_icon="🛒", layout="centered")
st.title("🛒 Product Page Scraper & Rewriter")
st.caption(f"Page extraction: Crawl4AI · AI: Google Gemini (`{llm.GEMINI_MODEL}`)")

state = st.session_state
state.setdefault("product", None)     # dict with product data + metadata
state.setdefault("rewrite", None)
state.setdefault("error", None)

if not llm.GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY is missing: structured data will still be read, "
               "but AI extraction and rewriting are disabled.")

with st.form("scrape"):
    url = st.text_input("Product page URL", max_chars=scraper.MAX_URL_LENGTH,
                        placeholder="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html")
    use_ai = st.checkbox("Use AI to fill in fields the page's structured data doesn't provide",
                         value=bool(llm.GEMINI_API_KEY), disabled=not llm.GEMINI_API_KEY)
    go = st.form_submit_button("Extract product", type="primary")


def run_scrape(raw_url, use_ai):
    """Returns a result dict. Raises only ScrapeError (page problems)."""
    url = scraper.validate_url(raw_url)
    html, markdown = scraper.fetch_page(url)

    structured = scraper.extract_structured(html)
    sources = {k: "structured data" for k in structured}
    data = dict(structured)
    for field, value in scraper.extract_html_fallback(html).items():
        if not data.get(field):
            data[field] = value
            sources[field] = "page HTML"

    notes, ai_error = [], None
    injection = scraper.find_injection_attempts(markdown + " " + html[:200_000])

    missing = [f for f in scraper.FIELDS if not data.get(f)]
    if use_ai and missing:
        try:
            ai_data = llm.extract_with_llm(scraper.prepare_for_model(markdown))
            for field in missing:
                if ai_data.get(field):
                    data[field] = ai_data[field]
                    sources[field] = "AI extraction"
        except llm.LLMError as e:
            ai_error = str(e)

    product = scraper.Product.safe(**data)
    # A field that failed validation is reported as missing, not with a wrong source.
    sources = {k: v for k, v in sources.items() if getattr(product, k, None) is not None}
    if len(markdown) > scraper.MAX_MODEL_CHARS:
        notes.append(f"Page text was long; only the first {scraper.MAX_MODEL_CHARS:,} characters were analysed.")
    return {"url": url, "product": product, "sources": sources, "ai_error": ai_error,
            "injection": injection, "notes": notes}


if go:
    state.rewrite = None
    state.product = None
    state.error = None
    try:
        with st.spinner("Opening the page in a headless browser… (10–30 seconds)"):
            state.product = run_scrape(url, use_ai)
    except scraper.ScrapeError as e:
        state.error = (type(e).__name__, str(e))
    except Exception as e:  # never crash the page
        state.error = ("Error", f"Unexpected error: {type(e).__name__}: {e}")

def show_text(text):
    """Display untrusted page text safely: markdown characters escaped, HTML never rendered."""
    st.markdown(re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$])", r"\\\1", text))


# ---------- Errors ----------

ICONS = {"BlockedError": "🚫", "NetworkError": "📡", "UnavailableError": "❓",
         "InvalidURLError": "🔗", "SetupError": "🛠️"}
if state.error:
    kind, msg = state.error
    st.error(f"{ICONS.get(kind, '⚠️')} {msg}")

# ---------- Results ----------

res = state.product
if res:
    p = res["product"]
    src = res["sources"]

    def source(field):
        return f"Source: {src[field]}" if field in src else "Not found on the page"

    st.divider()
    if res["ai_error"]:
        st.warning(f"AI extraction was skipped: {res['ai_error']}")
    if res["injection"]:
        with st.container(border=True):
            st.warning("⚠️ This page contains text that looks like instructions aimed at an AI. "
                       "It was treated as untrusted data and ignored.")
            with st.expander("Show suspicious text"):
                for snippet in res["injection"]:
                    st.text(snippet)  # plain text, never rendered as HTML/markdown
    for note in res["notes"]:
        st.caption(note)

    with st.container(border=True):
        st.markdown("#### Product name")
        show_text(p.name or "Not found")
        st.caption(source("name"))

        c1, c2 = st.columns(2)
        with c1:
            price = f"{p.price:,.2f} {p.currency or ''}".strip() if p.price is not None else "Not found"
            st.metric("Price", price)
            st.caption(source("price"))
        with c2:
            if p.rating is not None:
                stars = "★" * round(p.rating) + "☆" * (5 - round(p.rating))
                reviews = f" ({p.review_count:,} reviews)" if p.review_count is not None else ""
                st.metric("Rating", f"{p.rating:.1f} / 5")
                st.caption(f"{stars}{reviews} · {source('rating')}")
            else:
                st.metric("Rating", "Not available")
                st.caption("No rating found on the page")

        st.markdown("#### Description")
        show_text(p.description or "Not found")
        st.caption(source("description"))

    payload = {"url": res["url"], **p.model_dump(mode="json"), "field_sources": src}
    with st.expander("Validated structured data (JSON)"):
        st.json(payload)
    st.download_button("⬇️ Download JSON", json.dumps(payload, indent=2, ensure_ascii=False),
                       file_name="product.json", mime="application/json", on_click="ignore")

    # ---------- Rewrite ----------
    st.subheader("✍️ Rewrite the description with AI")
    if not p.description:
        st.info("No description was extracted, so there is nothing to rewrite.")
    elif not llm.GEMINI_API_KEY:
        st.info("Add GEMINI_API_KEY to .env to enable rewriting.")
    else:
        c1, c2 = st.columns(2)
        tone = c1.selectbox("Tone", list(llm.TONES))
        words = c2.slider("Maximum words", 30, llm.MAX_REWRITE_WORDS, 100, step=10)
        if st.button("Rewrite description"):
            try:
                with st.spinner("Rewriting…"):
                    state.rewrite = {"tone": tone,
                                     "text": llm.rewrite_description(p.name, p.description, tone, words)}
            except llm.LLMError as e:
                state.rewrite = None
                st.error(str(e))
            except Exception as e:
                state.rewrite = None
                st.error(f"Unexpected error: {type(e).__name__}")

        if state.rewrite:
            with st.container(border=True):
                st.markdown(f"**🤖 AI-generated rewrite ({state.rewrite['tone']})**")
                st.caption("Generated by the model from the extracted description; check facts before publishing.")
                show_text(state.rewrite["text"])
