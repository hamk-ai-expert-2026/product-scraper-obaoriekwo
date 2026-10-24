"""Page fetching, extraction and validation for the product scraper.

Everything that comes from a webpage is treated as untrusted data.
"""
import asyncio
import html as html_lib
import ipaddress
import json
import os
import re
import socket
import sys
import threading
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

# ---------- Limits ----------
MAX_URL_LENGTH = 2048
PAGE_TIMEOUT_MS = 30_000          # browser navigation timeout
OVERALL_TIMEOUT_S = 60            # hard stop for the whole crawl
MAX_HTML_CHARS = 2_000_000        # ~2 MB of HTML is parsed at most
MAX_MODEL_CHARS = 15_000          # page text sent to the LLM
MAX_DESCRIPTION_CHARS = 5_000
MAX_NAME_CHARS = 300

ALLOW_PRIVATE_URLS = os.getenv("ALLOW_PRIVATE_URLS", "").lower() in ("1", "true", "yes")


class ScrapeError(Exception):
    """Base class: the message is safe and helpful to show the user."""


class InvalidURLError(ScrapeError): ...
class NetworkError(ScrapeError): ...
class BlockedError(ScrapeError): ...
class UnavailableError(ScrapeError): ...
class SetupError(ScrapeError): ...


# ---------- Validated data model ----------

CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR", "₦": "NGN", "kr": "SEK"}


def parse_price(value):
    """Turn '£51.77', '1,299.00', '1.299,00' or 19.9 into Decimal, or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if re.search(r"-\s*[\d.]", str(value)):  # negative or a range like "10-20": not a usable price
        return None
    text = re.sub(r"[^\d.,]", "", str(value))
    if not text or not re.search(r"\d", text):
        return None
    if "," in text and "." in text:  # the later separator is the decimal point
        text = text.replace(",", "") if text.rfind(".") > text.rfind(",") else text.replace(".", "").replace(",", ".")
    elif "," in text:
        whole, _, frac = text.rpartition(",")
        text = f"{whole.replace(',', '')}.{frac}" if len(frac) in (1, 2) else text.replace(",", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def clean_text(value, limit):
    if value is None:
        return None
    text = BeautifulSoup(html_lib.unescape(str(value)), "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] or None


class Product(BaseModel):
    """Validated product data. Every field is optional; bad values become None."""
    name: str | None = None
    price: Decimal | None = Field(None, ge=0, lt=10_000_000)
    currency: str | None = None
    description: str | None = None
    rating: float | None = Field(None, ge=0, le=5)
    review_count: int | None = Field(None, ge=0)

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v):
        return clean_text(v, MAX_NAME_CHARS)

    @field_validator("description", mode="before")
    @classmethod
    def _description(cls, v):
        return clean_text(v, MAX_DESCRIPTION_CHARS)

    @field_validator("price", mode="before")
    @classmethod
    def _price(cls, v):
        p = parse_price(v)
        return p if p is not None and 0 <= p < 10_000_000 else None

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v):
        if not v:
            return None
        s = str(v).strip()
        if s in CURRENCY_SYMBOLS:
            return CURRENCY_SYMBOLS[s]
        s = s.upper()
        return s if re.fullmatch(r"[A-Z]{3}", s) else None

    @field_validator("rating", mode="before")
    @classmethod
    def _rating(cls, v):
        try:
            r = float(str(v).split("/")[0].replace(",", ".")) if v is not None else None
        except ValueError:
            return None
        return r if r is not None and 0 <= r <= 5 else None

    @field_validator("review_count", mode="before")
    @classmethod
    def _reviews(cls, v):
        if v is None:
            return None
        digits = re.sub(r"[^\d]", "", str(v))
        return int(digits) if digits and len(digits) < 10 else None

    @classmethod
    def safe(cls, **data):
        """Build a Product, dropping any single field that still fails validation."""
        clean = {}
        for key, value in data.items():
            if key not in cls.model_fields:
                continue
            try:
                cls.model_validate({key: value})
                clean[key] = value
            except Exception:
                pass
        return cls.model_validate(clean)


FIELDS = ["name", "price", "currency", "description", "rating", "review_count"]


# ---------- URL validation ----------

def validate_url(url):
    url = (url or "").strip()
    if not url:
        raise InvalidURLError("Please enter a product page URL.")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError("That URL is too long.")
    if re.match(r"^[a-z][a-z0-9+.-]*:", url, re.I) and not re.match(r"^https?://", url, re.I):
        raise InvalidURLError("Only http:// and https:// web page URLs are allowed.")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InvalidURLError("Only http:// and https:// web page URLs are allowed.")
    if not ALLOW_PRIVATE_URLS:
        try:
            infos = socket.getaddrinfo(parts.hostname, None)
        except socket.gaierror:
            raise NetworkError(f"Could not find the website '{parts.hostname}'. "
                               "Check the address and your internet connection.")
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise InvalidURLError("Local and private network addresses are not allowed.")
    return url


# ---------- Fetching with Crawl4AI ----------

def run_sync_in_thread(fn, timeout):
    """Run a normal (blocking) function in a fresh thread that has its own event loop.

    Some libraries call asyncio.get_event_loop(), which raises RuntimeError on
    Python 3.14 in Streamlit's script thread. A clean thread avoids that.
    """
    box = {}

    def runner():
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = fn()
        except BaseException as e:
            box["error"] = e
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError("timed out")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _run_in_thread(coro_factory, timeout):
    """Run an async function in its own thread and event loop.

    Streamlit's event loop on Windows can't start the browser subprocess, so the
    crawl gets a fresh Proactor loop (Windows) or default loop (elsewhere).
    """
    box = {}

    def runner():
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(asyncio.wait_for(coro_factory(), timeout))
        except BaseException as e:  # passed back to the caller
            box["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout + 10)
    if t.is_alive():
        raise NetworkError("The page took too long to load and was stopped.")
    if "error" in box:
        raise box["error"]
    return box["value"]


async def _crawl(url):
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    browser = BrowserConfig(headless=True, verbose=False)
    run = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=PAGE_TIMEOUT_MS,
        wait_until="domcontentloaded",
        delay_before_return_html=1.5,
        remove_overlay_elements=True,
        excluded_tags=["nav", "footer", "header", "form", "aside"],
        verbose=False,
    )
    async with AsyncWebCrawler(config=browser) as crawler:
        return await crawler.arun(url, config=run)


NETWORK_HINTS = [
    ("ERR_NAME_NOT_RESOLVED", "Could not find that website. Check the address."),
    ("ERR_INTERNET_DISCONNECTED", "No internet connection. Check your network and try again."),
    ("ERR_TIMED_OUT", "The website took too long to respond."),
    ("Timeout", "The website took too long to respond."),
    ("ERR_CONNECTION_REFUSED", "The website refused the connection."),
    ("ERR_CONNECTION_RESET", "The connection was reset. Try again."),
    ("ERR_CONNECTION_CLOSED", "The connection was closed unexpectedly. Try again."),
    ("ERR_CERT", "The website's security certificate is invalid."),
    ("ERR_SSL", "A secure connection to the website could not be made."),
    ("ERR_TUNNEL", "Network error: the connection could not be established (proxy/firewall)."),
    ("ERR_PROXY", "Network error: a proxy problem stopped the connection."),
    ("ERR_TOO_MANY_REDIRECTS", "The page redirects too many times."),
    ("ERR_ABORTED", "The page load was aborted, possibly a download link or a block."),
]

BLOCK_MARKERS = [
    "captcha", "robot check", "are you a robot", "not a robot", "verify you are human",
    "unusual traffic", "access denied", "pardon our interruption", "request blocked",
    "enable javascript and cookies", "checking your browser", "attention required",
    "automated access", "bot detection",
]


def fetch_page(url):
    """Return (html, markdown) for a URL, or raise a ScrapeError with a clear message."""
    try:
        result = _run_in_thread(lambda: _crawl(url), OVERALL_TIMEOUT_S)
    except ScrapeError:
        raise
    except ImportError:
        raise SetupError("Crawl4AI is not installed. Run: python -m pip install -r requirements.txt")
    except asyncio.TimeoutError:
        raise NetworkError("The page took too long to load and was stopped.")
    except Exception as e:
        msg = str(e)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            raise SetupError("The browser for Crawl4AI is not installed. Run: crawl4ai-setup")
        if isinstance(e, NotImplementedError):
            raise SetupError("Your system could not start the browser (event loop issue).")
        raise NetworkError(f"Could not load the page ({type(e).__name__}).")

    status = getattr(result, "status_code", None)
    if not getattr(result, "success", False):
        err = str(getattr(result, "error_message", "") or "")
        if "Executable doesn't exist" in err or "playwright install" in err:
            raise SetupError("The browser for Crawl4AI is not installed. Run: crawl4ai-setup")
        for marker, message in NETWORK_HINTS:
            if marker in err:
                raise NetworkError(message)
        if status:
            _check_status(status)
        raise UnavailableError("The page could not be loaded. It may be unavailable or blocking automated access.")

    if status:
        _check_status(status)

    html = (result.html or "")[:MAX_HTML_CHARS]
    md = getattr(result, "markdown", None)
    markdown = str(getattr(md, "raw_markdown", md) or "")

    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html[:20_000], re.I | re.S)
    if m:
        title = m.group(1).lower()
    head_text = (title + " " + markdown[:3000]).lower()
    if any(k in head_text for k in BLOCK_MARKERS) and len(markdown) < 6000:
        raise BlockedError("The website blocked automated access (bot check / CAPTCHA). "
                           "Large shops like Amazon and Temu often do this. Try another product page or site.")
    if len(markdown.strip()) < 150 and "ld+json" not in html:
        raise UnavailableError("The page loaded but had almost no content. It may require a login, "
                               "be region-locked, or rely on scripts that block automated browsers.")
    return html, markdown


def _check_status(status):
    if status in (401, 403, 429):
        raise BlockedError(f"The website refused access (HTTP {status}). It is probably blocking scrapers.")
    if status in (404, 410):
        raise UnavailableError(f"The page was not found (HTTP {status}). Check the URL.")
    if status == 503:
        raise BlockedError("The website is unavailable or blocking automated access (HTTP 503).")
    if status >= 500:
        raise UnavailableError(f"The website had a server error (HTTP {status}). Try again later.")
    if status >= 400:
        raise UnavailableError(f"The page could not be loaded (HTTP {status}).")


# ---------- Structured data (schema.org JSON-LD + meta tags) ----------

def _iter_nodes(data):
    if isinstance(data, list):
        for item in data:
            yield from _iter_nodes(item)
    elif isinstance(data, dict):
        yield data
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in data:
                yield from _iter_nodes(data[key])


def _is_product(node):
    t = node.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(isinstance(x, str) and x.lower() in ("product", "productgroup") for x in types)


def extract_structured(html):
    """Read product fields from JSON-LD and meta tags. Never raises."""
    found = {}
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return found

    for script in soup.find_all("script", type=re.compile("ld\\+json", re.I))[:30]:
        try:
            data = json.loads(script.string or "")
        except (ValueError, TypeError):
            continue
        for node in _iter_nodes(data):
            if not _is_product(node):
                continue
            found.setdefault("name", node.get("name"))
            found.setdefault("description", node.get("description"))
            offers = node.get("offers")
            offers = offers[0] if isinstance(offers, list) and offers else offers
            if isinstance(offers, dict):
                found.setdefault("price", offers.get("price") or offers.get("lowPrice"))
                found.setdefault("currency", offers.get("priceCurrency"))
            rating = node.get("aggregateRating")
            if isinstance(rating, dict):
                value, best = rating.get("ratingValue"), rating.get("bestRating")
                try:
                    if value is not None and best and float(best) not in (0, 5):
                        value = round(float(value) / float(best) * 5, 2)  # normalise to /5
                except (TypeError, ValueError):
                    pass
                found.setdefault("rating", value)
                found.setdefault("review_count", rating.get("reviewCount") or rating.get("ratingCount"))
            break

    def meta(*names):
        for n in names:
            tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if tag and tag.get("content"):
                return tag["content"]
        return None

    if not found.get("name"):
        found["name"] = meta("og:title", "twitter:title")
    if not found.get("price"):
        found["price"] = meta("product:price:amount", "og:price:amount")
        found.setdefault("currency", None)
        if not found.get("currency"):
            found["currency"] = meta("product:price:currency", "og:price:currency")
    if not found.get("description"):
        found["description"] = meta("og:description", "description")

    return {k: v for k, v in found.items() if v not in (None, "", [])}


# ---------- Common HTML patterns (no AI) ----------

STAR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
PRICE_RE = re.compile(r"(?:[$£€¥₹₦]|\b(?:USD|EUR|GBP|SEK|NGN)\b)\s?\d[\d.,]*|\d[\d.,]*\s?(?:[$£€]|\b(?:USD|EUR|GBP|SEK|kr)\b)")


def _currency_from_text(text):
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    m = re.search(r"\b(USD|EUR|GBP|SEK|NGN|JPY|INR)\b", text)
    return m.group(1) if m else None


def extract_html_fallback(html):
    """Read fields from microdata (itemprop) and common shop markup. Never raises."""
    found = {}
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return found

    def itemprop(name):
        tag = soup.find(attrs={"itemprop": name})
        if not tag:
            return None
        return tag.get("content") or tag.get_text(" ", strip=True) or None

    try:
        found["name"] = itemprop("name") or (soup.h1.get_text(" ", strip=True) if soup.h1 else None)

        price_text = itemprop("price")
        if not price_text:
            for tag in soup.find_all(class_=re.compile("price", re.I), limit=20):
                text = tag.get_text(" ", strip=True)
                if PRICE_RE.search(text):
                    price_text = PRICE_RE.search(text).group(0)
                    break
        if price_text:
            found["price"] = price_text
            found["currency"] = itemprop("priceCurrency") or _currency_from_text(price_text)

        rating = itemprop("ratingValue")
        if rating is None:
            tag = soup.find(class_=re.compile(r"star-rating", re.I))
            if tag:
                for cls in tag.get("class", []):
                    if cls.lower() in STAR_WORDS:
                        rating = STAR_WORDS[cls.lower()]
                        break
        found["rating"] = rating
        found["review_count"] = itemprop("reviewCount") or itemprop("ratingCount")
        found["description"] = itemprop("description")
    except Exception:
        pass
    return {k: v for k, v in found.items() if v not in (None, "", [])}


# ---------- Prompt-injection signals ----------

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"you are now", r"system prompt", r"new instructions", r"act as (an? )?(ai|assistant|chatgpt)",
    r"respond only with", r"do not (tell|reveal)", r"<\s*/?\s*(system|instructions?)\s*>",
]


def find_injection_attempts(text):
    """Return suspicious snippets found in page text (for display only)."""
    hits = []
    for pattern in INJECTION_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            start = max(0, m.start() - 40)
            hits.append(text[start:m.end() + 60].replace("\n", " "))
            if len(hits) >= 5:
                return hits
    return hits


def prepare_for_model(markdown):
    """Trim page text and neutralise anything that could close our data tags."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)      # drop images
    text = re.sub(r"\]\((https?://[^)]+)\)", "]", text)          # drop link targets
    text = re.sub(r"</?\s*untrusted[^>]*>", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_MODEL_CHARS]
