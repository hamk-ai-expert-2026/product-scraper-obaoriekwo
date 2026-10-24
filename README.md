# Product Page Scraper & Rewriter (Assignment #16)

A Streamlit app that takes a product page URL, retrieves the page with **Crawl4AI** (headless Chromium), extracts the product name, price, description and review rating into **validated structured data**, and can have **Google Gemini** rewrite the description.

## Setup

```bash
python -m pip install -r requirements.txt
crawl4ai-setup                 # one-time: installs the headless browser
copy .env.example .env         # macOS/Linux: cp .env.example .env
```

Put your Gemini key in `.env` as `GEMINI_API_KEY=...` and save.

## Run

```bash
python -m streamlit run app.py
```

Good test pages:
- https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html (scraping practice site)
- https://webscraper.io/test-sites/e-commerce/allinone (scraping practice site)
- An Amazon or Temu product page, which usually shows the **blocked** handling

## How it works

1. **URL check:** only `http`/`https`; local and private network addresses are refused.
2. **Fetch:** Crawl4AI loads the page in a headless browser. Crawls and Gemini calls run in their own thread and event loop, so they also work on Windows and Python 3.14.
3. **Structured data first:** schema.org `Product` JSON-LD and meta tags are read without any AI.
4. **HTML patterns:** microdata (`itemprop`), the page's `<h1>`, price elements and star-rating classes fill remaining gaps, still without AI.
5. **AI fallback (optional):** only fields still missing are requested from Gemini, using a fixed schema.
6. **Validation:** every value passes through the Pydantic `Product` model. Prices are parsed into decimals, currencies normalised to ISO codes, ratings checked to be 0–5, and HTML stripped. Invalid values become "Not found" instead of errors.
7. **Rewrite (optional):** after a successful extraction, only the validated description is sent to Gemini with a chosen tone and word limit.

Each field shows where it came from: *structured data*, *page HTML*, *AI extraction* or *Not found on the page*. The validated result can be viewed and downloaded as JSON.

## How it meets the requirements

| Requirement | Implementation |
|---|---|
| Accepts a product URL | URL box with validation (`scraper.validate_url`) |
| Crawl4AI page extraction | `scraper.fetch_page` using `AsyncWebCrawler` |
| Name, price, description, rating | JSON-LD/meta parsing, then common HTML patterns, then a Gemini fallback with a schema |
| Validated structured data | Pydantic `Product` model with field validators; the model is forced into a JSON schema, and Python validates every value again |
| Limits content | 30 s page timeout and 60 s hard stop; HTML capped at 2 MB; at most 15,000 characters sent to the AI; name, description and rewrite length capped; URL length limited |
| Unavailable or blocked pages | HTTP 401/403/429/503 → "blocked"; CAPTCHA and "Robot Check" detection; 404/410 and 5xx → "unavailable"; near-empty pages detected |
| Network failures | DNS failure, no internet, timeouts, connection reset or refused, SSL and proxy errors each get a clear message |
| Webpage text is untrusted | System instruction says page text is data and never instructions; page text is wrapped in `<untrusted_webpage>` tags (and those tags are stripped from the content); output is limited to schema fields and validated; the rewrite sees only the description; page text is shown escaped, never as HTML; links are never followed; suspected injection text is detected and shown as a warning |
| Never crashes on missing fields | All fields optional; `Product.safe()` drops any single invalid value; every step is wrapped in error handling |
| LLM rewrite option | Tone (Professional, Friendly, Persuasive, Short & simple, SEO-friendly) and a word limit, labelled as AI-generated |

## Testing the error handling

- Amazon product page → usually "The website blocked automated access".
- `https://books.toscrape.com/does-not-exist` → "The page was not found (HTTP 404)".
- `https://this-domain-does-not-exist-zz9.com` → "Could not find the website".
- `ftp://example.com` or `http://127.0.0.1` → refused as an invalid or private URL.
- Turn off Wi-Fi → a network error message.

## Files

- `app.py`: Streamlit interface
- `scraper.py`: URL checks, Crawl4AI fetching, blocked/network detection, JSON-LD parsing, Pydantic model, injection detection
- `llm.py`: Gemini extraction and rewriting with safety rules and schemas
