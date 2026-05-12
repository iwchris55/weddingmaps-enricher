#!/usr/bin/env python3
"""
venue_enricher.py — Wedding Maps venue content enrichment pipeline
==================================================================
Fetches venue pages that lack enriched content, generates AI descriptions +
FAQ pairs via Claude Haiku (with prompt caching), builds EventVenue/LocalBusiness
JSON-LD schema, optionally geocodes missing lat/lng via Google Geocoding API,
and writes everything back via WP REST API.

Configuration
-------------
Copy config.example.json → config.json and fill in credentials.
Or set environment variables (see CONFIG section below).

Usage
-----
  python3 venue_enricher.py                  # normal run (all tiers)
  python3 venue_enricher.py --dry-run        # preview without writing
  python3 venue_enricher.py --tier 1         # only indexed venues
  python3 venue_enricher.py --limit 50       # cap at 50 venues
  python3 venue_enricher.py --post-id 12345  # enrich a single venue

Priority tiers
--------------
  Tier 1 — Already indexed by Google (~3,700 venues)
           → enrich regardless of photos
  Tier 2 — Has attached photos, not yet indexed (~1,400)
           → good candidate for indexing
  Tier 3 — Has GSC impressions but not indexed (~2,500)
           → Google has seen these, thin content is the blocker
  Tier 4 — Remainder (~6,400), enrich last
           → processed over months to stay within Places free credit

Resume / idempotency
--------------------
Progress is tracked in progress.json (same directory as this script).
Already-enriched venues (wm_enriched_at is set) are skipped unless
--force is passed.
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import urllib.parse
import warnings

# Suppress urllib3 OpenSSL warning on macOS
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import requests
from anthropic import Anthropic

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"
PROGRESS_FILE = SCRIPT_DIR / "enrichment_progress.json"

PIPELINE_VERSION = "1.0"

# Defaults — override via config.json or environment variables
DEFAULT_CONFIG = {
    # WordPress REST API
    "wp_base_url": "https://weddingmaps.com",            # no trailing slash
    "wp_username": "",                                    # WP Application Password username
    "wp_app_password": "",                               # WP Application Password (spaces OK)

    # Anthropic
    "anthropic_api_key": "",                             # sk-ant-...

    # Google APIs (optional — leave blank to skip)
    "google_geocoding_api_key": "",   # Used for geocoding missing lat/lng
    "google_places_api_key": "",      # Used for Places lookup (phone, website, rating)
                                      # Can be the same key if both APIs are enabled on it

    # Tuning
    "per_page": 50,                # venues fetched per REST page (WP max)
    "delay_between_venues": 0.3,   # seconds between venue API calls
    "delay_between_pages": 1.0,    # seconds between WP REST pages
    "max_retries": 3,
    "retry_delay": 5.0,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    # Environment variable overrides
    env_map = {
        "WP_BASE_URL":              "wp_base_url",
        "WP_USERNAME":              "wp_username",
        "WP_APP_PASSWORD":          "wp_app_password",
        "WM_SECRET":                "wm_secret",
        "ANTHROPIC_API_KEY":        "anthropic_api_key",
        "GOOGLE_GEOCODING_API_KEY": "google_geocoding_api_key",
        "GOOGLE_PLACES_API_KEY":    "google_places_api_key",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val
    return cfg


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"enriched": [], "failed": [], "noindexed": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ── WordPress REST API helpers ────────────────────────────────────────────────

class WPClient:
    def __init__(self, base_url: str, secret: str):
        self.base_url = base_url.rstrip("/")
        self.secret   = secret
        # No Authorization header — authenticated REST requests cause 502 on this host.
        # Writes use X-WM-Secret header against the WM_ENRICHER_SECRET PHP constant.
        # Reads use the public wm/v1 endpoints (venue data is not sensitive).
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "curl/7.88.1",
        })

    def get_venues_page(self, page: int, per_page: int) -> list[dict]:
        """Fetch a page of venues via the lightweight wm/v1/venues endpoint."""
        url = f"{self.base_url}/wp-json/wm/v1/venues"
        params = {"page": page, "per_page": per_page}
        resp = self.session.get(url, params=params, timeout=60)
        if resp.status_code in (400, 404):
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("venues", [])

    def get_venue(self, post_id: int) -> dict:
        """Fetch a single venue via the lightweight wm/v1/venue endpoint."""
        url = f"{self.base_url}/wp-json/wm/v1/venue/{post_id}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def update_venue_meta(self, post_id: int, meta: dict, dry_run: bool = False) -> bool:
        """Write wm_* meta fields via the lightweight wm/v1/venue endpoint."""
        if dry_run:
            print(f"    [DRY RUN] Would write meta to post {post_id}:")
            for k, v in meta.items():
                preview = str(v)[:80] + ("…" if len(str(v)) > 80 else "")
                print(f"      {k}: {preview}")
            return True
        url = f"{self.base_url}/wp-json/wm/v1/venue/{post_id}"
        resp = self.session.post(
            url, json=meta, timeout=30,
            params={"_wm_token": self.secret},
        )
        if not resp.ok:
            print(f"    ✗ REST write failed ({resp.status_code}): {resp.text[:200]}")
            return False
        return True

    def sideload_featured_image(self, post_id: int, image_url: str, dry_run: bool = False) -> bool:
        """Download a remote image and set it as the venue's featured image."""
        if dry_run:
            print(f"    [DRY RUN] Would sideload image: {image_url[:80]}")
            return True
        url = f"{self.base_url}/wp-json/wm/v1/venue/{post_id}/image"
        resp = self.session.post(
            url,
            json={"image_url": image_url},
            timeout=60,  # sideloading can be slow
            params={"_wm_token": self.secret},
        )
        if not resp.ok:
            print(f"    ✗ Image upload failed ({resp.status_code}): {resp.text[:200]}")
            return False
        data = resp.json()
        if data.get("status") == "skipped":
            print(f"    ↷ Image: {data.get('reason', 'skipped')}")
        else:
            print(f"    ✓ Image uploaded (attachment {data.get('attachment_id')})")
        return True

    def set_yoast_noindex(self, post_id: int, noindex: bool = True, dry_run: bool = False) -> bool:
        """Set or clear Yoast noindex via post meta."""
        meta = {"_yoast_wpseo_meta-robots-noindex": "1" if noindex else "0"}
        return self.update_venue_meta(post_id, meta, dry_run=dry_run)


# ── Google Geocoding ──────────────────────────────────────────────────────────

def geocode_address(address: str, api_key: str) -> Optional[tuple[float, float]]:
    """Return (lat, lng) or None if geocoding fails."""
    if not api_key or not address:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return (loc["lat"], loc["lng"])
    except Exception as e:
        print(f"    Geocoding error: {e}")
    return None


# ── Google Places lookup ──────────────────────────────────────────────────────

def lookup_google_place(name: str, address: str, api_key: str) -> Optional[dict]:
    """
    Search Google Places (New API) for a venue by name + address.
    Returns a dict with keys: place_id, phone, website, rating, review_count
    or None if no confident match is found.

    Requires the "Places API (New)" to be enabled on the Google Cloud project.
    Cost: ~$0.032/call for Text Search + ~$0.017/call for Place Details = ~$0.05/venue.
    """
    if not api_key or not name:
        return None

    # ── Step 1: Text Search to find the Place ID
    search_url = "https://places.googleapis.com/v1/places:searchText"
    query = f"{name} wedding venue {address}" if address else f"{name} wedding venue"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress",
    }
    try:
        resp = requests.post(
            search_url,
            json={"textQuery": query, "maxResultCount": 1},
            headers=headers,
            timeout=10,
        )
        if not resp.ok:
            print(f"    Places search error {resp.status_code}: {resp.text[:100]}")
            return None
        results = resp.json().get("places", [])
        if not results:
            return None
        place = results[0]
        place_id = place.get("id")
        if not place_id:
            return None

        # Basic name-match sanity check (avoid wild mismatches)
        returned_name = place.get("displayName", {}).get("text", "").lower()
        venue_name_lower = name.lower()
        # Accept if there's meaningful overlap (at least 3 consecutive words or >50% of words match)
        venue_words = set(venue_name_lower.split())
        match_words = set(returned_name.split())
        overlap = venue_words & match_words
        if len(overlap) == 0 and len(venue_words) > 1:
            print(f"    Places: no name match (got '{returned_name}')")
            return None

    except Exception as e:
        print(f"    Places search error: {e}")
        return None

    # ── Step 2: Place Details — request all useful fields in one call
    # Field mask tiers: Basic (rating, priceLevel, addressComponents, parking, accessibility)
    # + Advanced (phone, website) — billed as Advanced SKU (~$0.020/call)
    details_url = f"https://places.googleapis.com/v1/places/{place_id}"
    detail_headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join([
            "id",
            "nationalPhoneNumber",
            "internationalPhoneNumber",
            "websiteUri",
            "rating",
            "userRatingCount",
            "priceLevel",
            "businessStatus",
            "addressComponents",
            "parkingOptions",
            "accessibilityOptions",
        ]),
    }
    try:
        resp = requests.get(details_url, headers=detail_headers, timeout=10)
        if not resp.ok:
            print(f"    Places details error {resp.status_code}: {resp.text[:100]}")
            return {"place_id": place_id}
        d = resp.json()

        # ── Parse addressComponents → city, state
        city = state = ""
        for comp in d.get("addressComponents", []):
            types = comp.get("types", [])
            if "locality" in types:
                city = comp.get("longText", "")
            elif "administrative_area_level_1" in types:
                state = comp.get("shortText", "")  # e.g. "CA"

        # ── Parse priceLevel enum → symbol
        price_map = {
            "PRICE_LEVEL_FREE":          "Free",
            "PRICE_LEVEL_INEXPENSIVE":   "$",
            "PRICE_LEVEL_MODERATE":      "$$",
            "PRICE_LEVEL_EXPENSIVE":     "$$$",
            "PRICE_LEVEL_VERY_EXPENSIVE":"$$$$",
        }
        price_level = price_map.get(d.get("priceLevel", ""), "")

        # ── Parse parkingOptions → human-readable list
        parking_opts = d.get("parkingOptions", {})
        parking_labels = {
            "freeParkingLot":   "Free parking lot",
            "paidParkingLot":   "Paid parking lot",
            "freeStreetParking":"Free street parking",
            "paidStreetParking":"Paid street parking",
            "valetParking":     "Valet parking",
            "freeGarageParking":"Free garage parking",
            "paidGarageParking":"Paid garage parking",
        }
        parking = ", ".join(
            label for key, label in parking_labels.items()
            if parking_opts.get(key) is True
        )

        # ── Parse accessibilityOptions
        access_opts = d.get("accessibilityOptions", {})
        accessible = (
            access_opts.get("wheelchairAccessibleEntrance") is True
            or access_opts.get("wheelchairAccessibleSeating") is True
        )

        return {
            "place_id":        place_id,
            "phone":           d.get("nationalPhoneNumber") or d.get("internationalPhoneNumber") or "",
            "website":         d.get("websiteUri", ""),
            "rating":          d.get("rating", ""),
            "review_count":    d.get("userRatingCount", ""),
            "price_level":     price_level,
            "business_status": d.get("businessStatus", ""),
            "city":            city,
            "state":           state,
            "parking":         parking,
            "accessible":      "Yes" if accessible else "",
        }
    except Exception as e:
        print(f"    Places details error: {e}")
        return {"place_id": place_id}


# ── Claude Haiku enrichment ───────────────────────────────────────────────────

# System prompt — cached across all venue calls
SYSTEM_PROMPT = """\
You are a wedding venue content writer for WeddingMaps.com. For each venue produce
all of the following fields exactly as specified.

1. DESCRIPTION (120–180 words)
   - Target engaged couples searching Google for wedding venues
   - Write in second person ("your wedding", "you'll find")
   - Focus on atmosphere, architectural style, location context, and what makes
     the venue distinctive for a wedding
   - Do NOT invent specific prices, exact capacities, or claim specific amenities
     unless they appear in the venue data provided

2. TAGLINE (1–2 sentences, max 155 characters total)
   - A punchy SEO meta description hook
   - Capture the venue's single most compelling quality and city/location
   - Example: "A restored 1914 railway station in Oakland's arts district — soaring
     ceilings and raw industrial character for couples who want a wedding with soul."

3. VENUE_STYLE — pick the single best match from this exact list (copy the string exactly):
   Ballrooms | Banquet Hall/Restaurant | Barn/Farm/Ranch | Beach | Bed & Breakfast/Inn |
   Botanical Garden | Castle | Chapel | Church/Temple | City Hall | City/Skyline View |
   Community Center | Courthouse | Cruise Ship/Yacht | Estate | Event Center | Glacier |
   Golf Course | Historic/Landmark Building | Hotel/Resort | Island | Lake | Lodge | Loft |
   Manor | Mansion | Modern | Mountain | Museum/Gallery | Ocean/Waterfront View | Outdoor |
   Park/Garden | Plantation | Private Club | Private Estate | Rainforest | Rustic & Barn |
   Town | University | Villa

4. INDOOR_OUTDOOR — pick exactly ONE: Indoor | Outdoor | Both

5. GUEST_CAPACITY — estimate venue size. Pick ONE of these exact values:
   Under 100 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500
   Use "500" for venues that clearly hold more than 500.
   ONLY provide this if the venue data does NOT already include guest capacity.
   If real capacity data was provided, return an empty string "".

6. AMENITIES — array of amenity terms that clearly apply to this venue.
   Copy strings EXACTLY from this list (parking/accessibility terms are added from
   Google Places data separately — do NOT include them here):

   "Bride's dressing area" | "Bride's dressing area available for an additional fee" |
   "Ceremony Area" | "Ceremony arch" | "Champagne toast" | "Coat check room" |
   "Complimentary bridal suite" | "Dance Area" | "Dance floor" | "Day-of coordinator" |
   "Elopement Photography Location" | "Engagement photo location" |
   "Full kitchen facilities" | "Gazebo" | "Groom's dressing area" |
   "Groom's dressing area available for an additional fee" | "Indoor Event Space" |
   "Kitchen for prep only" | "Liability Insurance" |
   "Linens, silverware, glassware provided" | "No kitchen" | "Outdoor lighting" |
   "Overnight accommodations available" | "Piano" |
   "Piano available for an additional fee" | "Podium and/or stage" |
   "Reception tables and chairs provided" | "Security" | "Tables and chairs provided" |
   "Upgraded chairs" | "Venue set up and clean up" | "Votive candles" |
   "Wedding planning services" | "Wireless Internet"

   Be conservative — only include terms you are confident apply. Return [] if uncertain.
   Guidance: hotels/resorts → overnight + coordinator + kitchen + tables + linens + setup;
   ballrooms → dance floor + tables + linens + setup + podium;
   banquet halls/restaurants → kitchen + tables + linens + setup;
   barn/farm/ranch → outdoor lighting + dance floor + tables + setup;
   churches/chapels → ceremony arch + podium;
   indoor venues → bride's + groom's dressing area;
   gardens/parks → ceremony arch + outdoor lighting.

7. FIVE FAQ PAIRS
   Rules:
   - Each question targets a real Google search: "[venue name] wedding [topic]"
   - Cover DIFFERENT aspects: style/architecture, location, ceremony space,
     photography, and what type of couple the venue suits best
   - Every answer MUST be substantive (3–5 sentences)
   - NEVER lead with "contact the venue" — give a real, knowledgeable answer
     based on venue type, location, and style; optionally mention specifics
     can be confirmed directly (at most once across all five answers)
   - Draw on your knowledge of the venue (if a known landmark), the city,
     and the venue category

Respond ONLY with valid JSON — no markdown, no extra keys:
{
  "description": "...",
  "tagline": "...",
  "venue_style": "...",
  "indoor_outdoor": "...",
  "guest_capacity": "...",
  "amenities": ["...", "..."],
  "faqs": [
    {"q": "...", "a": "..."},
    {"q": "...", "a": "..."},
    {"q": "...", "a": "..."},
    {"q": "...", "a": "..."},
    {"q": "...", "a": "..."}
  ]
}
"""

# Map indoor_outdoor value → listing-services taxonomy term names
INDOOR_OUTDOOR_SERVICES = {
    "Indoor": ["Indoor Ceremony", "Indoor Reception"],
    "Outdoor": ["Outdoor Ceremony", "Outdoor Reception"],
    "Both":   ["Indoor Ceremony", "Indoor Reception", "Outdoor Ceremony", "Outdoor Reception"],
}

# Map Google Places price level symbol → qodef_listing_single_budget key
# (weddingmaps_price_range: 1=$10k, 2=$20k, 3=$40k, 4=Over $40k)
PRICE_LEVEL_TO_BUDGET = {
    "$":    "1",
    "$$":   "2",
    "$$$":  "3",
    "$$$$": "4",
}

# Full set of listing-amenity taxonomy terms.
# Claude selects from the non-parking/non-accessibility subset.
# Parking and accessibility terms are mapped deterministically from Places data.
AMENITY_TERMS = [
    # Venue features — Claude selects these
    "Bride's dressing area",
    "Bride's dressing area available for an additional fee",
    "Ceremony Area",
    "Ceremony arch",
    "Champagne toast",
    "Coat check room",
    "Complimentary bridal suite",
    "Dance Area",
    "Dance floor",
    "Day-of coordinator",
    "Elopement Photography Location",
    "Engagement photo location",
    "Full kitchen facilities",
    "Gazebo",
    "Groom's dressing area",
    "Groom's dressing area available for an additional fee",
    "Indoor Event Space",
    "Kitchen for prep only",
    "Liability Insurance",
    "Linens, silverware, glassware provided",
    "No kitchen",
    "Outdoor lighting",
    "Overnight accommodations available",
    "Piano",
    "Piano available for an additional fee",
    "Podium and/or stage",
    "Reception tables and chairs provided",
    "Security",
    "Tables and chairs provided",
    "Upgraded chairs",
    "Venue set up and clean up",
    "Votive candles",
    "Wedding planning services",
    "Wireless Internet",
    # Parking/accessibility — mapped from Google Places data
    "Handicap Accessible",
    "Large parking lot",
    "Parking can be arranged",
    "Public garage",
    "Public parking",
    "Valet or public parking for a fee",
    "Valet/shuttle service provided",
]

# Fast lookup set for validating Claude output
AMENITY_TERMS_SET = set(AMENITY_TERMS)

# Terms Claude should NOT select (handled deterministically from Places data)
PLACES_AMENITY_TERMS = {
    "Handicap Accessible",
    "Large parking lot",
    "Parking can be arranged",
    "Public garage",
    "Public parking",
    "Valet or public parking for a fee",
    "Valet/shuttle service provided",
}

# Terms Claude CAN select (everything except parking/accessibility)
CLAUDE_AMENITY_TERMS = [t for t in AMENITY_TERMS if t not in PLACES_AMENITY_TERMS]


def places_to_amenities(place_data: dict) -> list[str]:
    """
    Map Google Places parking/accessibility fields to listing-amenity term names.
    The Places API returns parking as a human-readable comma-separated string
    (e.g. "Free parking lot, Valet parking") built in lookup_google_place().
    """
    amenities = []
    parking_str = (place_data.get("parking") or "").lower()

    if parking_str:
        if "parking lot" in parking_str:
            amenities.append("Large parking lot")
        if "valet" in parking_str:
            amenities.append("Valet/shuttle service provided")
        if "garage" in parking_str:
            amenities.append("Public garage")
        if "paid street parking" in parking_str:
            amenities.append("Valet or public parking for a fee")
        elif "street parking" in parking_str:
            amenities.append("Public parking")

    if place_data.get("accessible") == "Yes":
        amenities.append("Handicap Accessible")

    return amenities

def build_user_message(venue: dict, extra_meta: dict) -> str:
    """Build the per-venue user message for Claude."""
    name = venue.get("title", {}).get("rendered", "Unknown Venue")

    # Pull meta fields (REST response + any enriched data from Places/geocoding)
    meta = venue.get("meta", {})
    meta.update(extra_meta)

    address    = meta.get("qodef_listing_single_full_address", "") or ""
    capacity   = meta.get("qodef_listing_single_guest_capacity", "") or ""
    phone      = meta.get("qodef_listing_single_phone", "") or ""
    website    = meta.get("qodef_listing_single_site_url", "") or ""
    lat        = meta.get("qodef_listing_single_latitude", "") or ""
    lng        = meta.get("qodef_listing_single_longitude", "") or ""
    price_level   = meta.get("wm_price_level", "") or ""
    parking       = meta.get("wm_parking", "") or ""
    accessible    = meta.get("wm_accessible", "") or ""
    city          = meta.get("wm_city", "") or ""
    state         = meta.get("wm_state", "") or ""

    lines = [f"Venue name: {name}"]
    if address:  lines.append(f"Address: {address}")
    elif city and state: lines.append(f"Location: {city}, {state}")
    if capacity: lines.append(f"Guest capacity: {capacity} (real data — do NOT generate a capacity_tier)")
    if phone:    lines.append(f"Phone: {phone}")
    if website:  lines.append(f"Website: {website}")
    if price_level: lines.append(f"Price level: {price_level}")
    if parking:  lines.append(f"Parking: {parking}")
    if accessible: lines.append(f"Accessibility: Wheelchair accessible")
    if lat and lng: lines.append(f"Coordinates: {lat}, {lng}")

    return "\n".join(lines)


def fetch_og_image(website_url: str) -> Optional[str]:
    """
    Fetch the og:image URL from a venue's website.
    Returns the absolute image URL or None if not found.
    Falls back to twitter:image if og:image is absent.
    """
    if not website_url:
        return None
    try:
        resp = requests.get(
            website_url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WeddingMapsBot/1.0)"},
            allow_redirects=True,
        )
        if not resp.ok:
            return None
        html = resp.text[:50_000]  # only parse the <head>
        import re
        for prop in ("og:image", "twitter:image"):
            # property="og:image" content="URL"
            m = re.search(
                r'<meta[^>]+(?:property|name)=["\']' + re.escape(prop) + r'["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            if not m:
                # content="URL" property="og:image"
                m = re.search(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']' + re.escape(prop) + r'["\']',
                    html, re.IGNORECASE,
                )
            if m:
                img = m.group(1).strip()
                # Make relative URLs absolute
                if img.startswith("//"):
                    img = "https:" + img
                elif img.startswith("/"):
                    from urllib.parse import urlparse
                    base = urlparse(website_url)
                    img = f"{base.scheme}://{base.netloc}{img}"
                if img.startswith("http"):
                    return img
    except Exception:
        pass
    return None


def enrich_venue_with_claude(
    client: Anthropic,
    venue: dict,
    extra_meta: dict,
) -> Optional[dict]:
    """
    Call Claude Haiku with prompt caching.
    Returns {"description": str, "faqs": list} or None on failure.
    """
    user_msg = build_user_message(venue, extra_meta)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1200,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},   # cache the system prompt
                }
            ],
            messages=[
                {"role": "user", "content": user_msg}
            ],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if Claude wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        if "description" not in data or "faqs" not in data:
            print(f"    ✗ Claude response missing required keys")
            return None
        for key in ("tagline", "venue_style", "indoor_outdoor", "guest_capacity"):
            data.setdefault(key, "")
        data.setdefault("amenities", [])
        return data

    except json.JSONDecodeError as e:
        print(f"    ✗ Claude returned invalid JSON: {e}")
        return None
    except Exception as e:
        print(f"    ✗ Claude API error: {e}")
        return None


# ── Schema builder ────────────────────────────────────────────────────────────

def build_schema(venue: dict, meta: dict, lat: Optional[float], lng: Optional[float]) -> dict:
    """Build EventVenue + LocalBusiness JSON-LD dict."""
    name     = venue.get("title", {}).get("rendered", "")
    url      = venue.get("link", "")
    address  = meta.get("qodef_listing_single_full_address", "") or ""
    phone    = meta.get("qodef_listing_single_phone", "") or ""
    website  = meta.get("qodef_listing_single_site_url", "") or ""
    capacity = meta.get("qodef_listing_single_guest_capacity", "") or ""
    city     = meta.get("wm_city", "") or ""
    state    = meta.get("wm_state", "") or ""
    rating   = meta.get("wm_rating", "") or ""
    reviews  = meta.get("wm_review_count", "") or ""

    schema: dict = {
        "@context": "https://schema.org",
        "@type": ["EventVenue", "LocalBusiness"],
        "name": name,
        "url": url,
    }

    if address or city:
        addr_block: dict = {"@type": "PostalAddress"}
        if address:
            addr_block["streetAddress"] = address
        if city:
            addr_block["addressLocality"] = city
        if state:
            addr_block["addressRegion"] = state
        addr_block["addressCountry"] = "US"
        schema["address"] = addr_block

    if phone:
        schema["telephone"] = phone

    if website:
        schema["sameAs"] = website

    if lat is not None and lng is not None:
        schema["geo"] = {
            "@type": "GeoCoordinates",
            "latitude": lat,
            "longitude": lng,
        }

    if capacity:
        try:
            schema["maximumAttendeeCapacity"] = int(str(capacity).replace(",", ""))
        except ValueError:
            pass

    # aggregateRating — only include when we have real Google data
    if rating and reviews:
        try:
            schema["aggregateRating"] = {
                "@type":       "AggregateRating",
                "ratingValue": float(rating),
                "reviewCount": int(reviews),
                "bestRating":  5,
                "worstRating": 1,
            }
        except (ValueError, TypeError):
            pass

    return schema


# ── Priority tier classification ──────────────────────────────────────────────

def classify_tier(venue: dict, indexed_slugs: set, impressed_slugs: set) -> int:
    """
    Return priority tier 1–4 for a venue.
    1 = already indexed by Google
    2 = has photos, not yet indexed
    3 = has GSC impressions, not yet indexed
    4 = everything else — enrich last, publish when done
    """
    slug = venue.get("slug", "")
    if slug in indexed_slugs:
        return 1
    # Check if venue has any attached photos
    # (The REST API _fields doesn't include featured_media count easily;
    #  we use a heuristic: if the post has a featured_media ID > 0)
    has_photo = bool(venue.get("featured_media", 0))
    if has_photo:
        return 2
    if slug in impressed_slugs:
        return 3
    return 4


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_slugs_from_gsc(gsc_pages_csv: Optional[Path]) -> set:
    """Load venue slugs that have GSC data from the Pages.csv export."""
    slugs = set()
    if not gsc_pages_csv or not gsc_pages_csv.exists():
        return slugs
    import csv
    with open(gsc_pages_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("Top pages", "") or row.get("Page", "") or ""
            if "/venue/" in url:
                slug = url.rstrip("/").split("/")[-1]
                if slug:
                    slugs.add(slug)
    return slugs


def run_pipeline(args, config: dict):
    # ── Setup clients
    wp = WPClient(
        config["wp_base_url"],
        config["wm_secret"],
    )
    anthropic_client = Anthropic(api_key=config["anthropic_api_key"])

    progress = load_progress()
    already_done = set(progress["enriched"])
    already_failed = set(progress["failed"])

    # ── Load GSC slug sets for tier classification
    gsc_pages_csv = SCRIPT_DIR / "gsc-data" / "Pages.csv"
    impressed_slugs = load_slugs_from_gsc(gsc_pages_csv)
    # Tier 1 = venues that appear in GSC (have impressions = Google has indexed them).
    # These are the highest-priority targets: Google already shows them but content is thin.
    indexed_slugs: set = impressed_slugs

    print(f"Wedding Maps Venue Enricher v{PIPELINE_VERSION}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Tier filter: {args.tier or 'all'}")
    print(f"  Limit: {args.limit or 'none'}")
    print(f"  GSC slugs with impressions: {len(impressed_slugs)}")
    print()

    if args.post_id:
        # Single-venue mode
        venues_to_process = [wp.get_venue(args.post_id)]
    else:
        venues_to_process = None  # will page through all

    enriched_count = 0
    error_count = 0

    def process_venue(venue: dict) -> bool:
        """Process one venue. Returns True if actual work was done, False if skipped."""
        nonlocal enriched_count, error_count

        post_id = venue["id"]
        slug    = venue.get("slug", "")
        title   = venue.get("title", {}).get("rendered", f"[post {post_id}]")

        # Skip already processed (silent — caller tallies these)
        if str(post_id) in already_done and not args.force:
            return False

        # Tier classification
        tier = classify_tier(venue, indexed_slugs, impressed_slugs)

        # Filter by requested tier
        if args.tier and tier != args.tier:
            return False

        # Check if already enriched via meta (belt-and-suspenders)
        meta = venue.get("meta", {}) or {}
        if meta.get("wm_enriched_at") and not args.force:
            already_done.add(str(post_id))
            return False

        print(f"  ► [{tier}] {title} (id={post_id})")

        # ── Geocoding (if missing lat/lng)
        lat  = meta.get("qodef_listing_single_latitude")
        lng  = meta.get("qodef_listing_single_longitude")
        geocoded = False
        if (not lat or not lng) and config.get("google_geocoding_api_key"):
            address = meta.get("qodef_listing_single_full_address", "")
            if address:
                coords = geocode_address(address, config["google_geocoding_api_key"])
                if coords:
                    lat, lng = coords
                    geocoded = True
                    print(f"    Geocoded → {lat:.5f}, {lng:.5f}")

        lat_f  = float(lat)  if lat  else None
        lng_f  = float(lng)  if lng  else None

        # ── Google Places enrichment (fills phone, website, rating when missing)
        places_key = config.get("google_places_api_key", "")
        place_data = None   # populated below if Places API is configured + match found
        has_phone   = bool(meta.get("qodef_listing_single_phone"))
        has_website = bool(meta.get("qodef_listing_single_site_url"))
        has_rating  = bool(meta.get("wm_rating"))
        # Skip Places API if we already have the key contact/rating data — saves ~$0.017/venue
        # Also skip if --no-places flag is set (e.g. monthly credit exhausted)
        need_places = places_key and not (has_phone and has_website and has_rating) and not args.no_places
        if need_places:
            address = meta.get("qodef_listing_single_full_address", "")
            place_data = lookup_google_place(title, address, places_key)
            if place_data:  # re-assign so outer scope picks it up
                place_id = place_data.get("place_id", "")
                if place_id:
                    print(f"    Places: found {place_id}")

                # Fill in phone if not already set
                if place_data.get("phone") and not meta.get("qodef_listing_single_phone"):
                    meta["qodef_listing_single_phone"] = place_data["phone"]

                # Fill in website if not already set
                if place_data.get("website") and not meta.get("qodef_listing_single_site_url"):
                    meta["qodef_listing_single_site_url"] = place_data["website"]

                # Store all Places fields in meta for use by schema builder + AI message
                if place_id:
                    meta["wm_place_id"] = place_id
                for src_key, meta_key in [
                    ("rating",          "wm_rating"),
                    ("review_count",    "wm_review_count"),
                    ("price_level",     "wm_price_level"),
                    ("business_status", "wm_business_status"),
                    ("city",            "wm_city"),
                    ("state",           "wm_state"),
                    ("parking",         "wm_parking"),
                    ("accessible",      "wm_accessible"),
                ]:
                    val = place_data.get(src_key)
                    if val is not None and val != "":
                        meta[meta_key] = str(val)

        # ── Claude enrichment
        ai_data = enrich_venue_with_claude(anthropic_client, venue, meta)
        if not ai_data:
            error_count += 1
            progress["failed"].append(str(post_id))
            save_progress(progress)
            return

        description    = ai_data["description"]
        tagline        = ai_data.get("tagline", "")
        venue_style    = ai_data.get("venue_style", "")
        indoor_outdoor = ai_data.get("indoor_outdoor", "")
        guest_capacity = ai_data.get("guest_capacity", "")
        faqs           = ai_data["faqs"]   # list of {q, a}

        # ── Merge Claude + Places amenities
        # Claude selects venue-characteristic terms; Places supplies parking/accessibility.
        claude_amenities = [
            t for t in ai_data.get("amenities", [])
            if t in AMENITY_TERMS_SET and t not in PLACES_AMENITY_TERMS
        ]
        places_amenities = places_to_amenities(place_data) if place_data else []
        all_amenities = list(set(claude_amenities + places_amenities))

        orig_meta = venue.get("meta", {}) or {}

        # ── Build schema JSON-LD (includes aggregateRating if Places data available)
        schema = build_schema(venue, meta, lat_f, lng_f)
        schema_json = json.dumps(schema, ensure_ascii=False)

        # ── Assemble meta payload
        now_iso = datetime.now(timezone.utc).isoformat()
        new_meta: dict = {
            "wm_description":    description,
            "wm_faq_json":       json.dumps(faqs, ensure_ascii=False),
            "wm_schema_json":    schema_json,
            "wm_enriched_at":    now_iso,
            "wm_enrichment_ver": PIPELINE_VERSION,
        }

        # Tagline → wm_tagline + Yoast meta description
        if tagline:
            new_meta["wm_tagline"]              = tagline
            new_meta["_yoast_wpseo_metadesc"]   = tagline

        # ── Taxonomy terms
        # Uses existing WordPress taxonomies — no duplicate wm_* fields needed.
        _terms: dict = {}
        if venue_style:
            _terms["listing-styles"] = [venue_style]
            print(f"    Style: {venue_style}")
        if indoor_outdoor in INDOOR_OUTDOOR_SERVICES:
            _terms["listing-services"] = INDOOR_OUTDOOR_SERVICES[indoor_outdoor]
            print(f"    Indoor/Outdoor: {indoor_outdoor}")
        if all_amenities:
            _terms["listing-amenity"] = all_amenities
            print(f"    Amenities ({len(all_amenities)}): {', '.join(sorted(all_amenities))}")
        if _terms:
            new_meta["_terms"] = _terms

        # ── Guest capacity → existing qodef_ dropdown (only if not already set)
        if guest_capacity and not orig_meta.get("qodef_listing_single_guest_capacity"):
            new_meta["qodef_listing_single_guest_capacity"] = guest_capacity
            print(f"    Capacity: {guest_capacity} (AI estimate)")

        # ── Budget from Places price level → existing qodef_listing_single_budget
        budget_key = PRICE_LEVEL_TO_BUDGET.get(meta.get("wm_price_level", ""), "")
        if budget_key and not orig_meta.get("qodef_listing_single_budget"):
            new_meta["qodef_listing_single_budget"] = budget_key

        # ── Geocoded coordinates
        if geocoded:
            new_meta["qodef_listing_single_latitude"]  = str(lat_f)
            new_meta["qodef_listing_single_longitude"] = str(lng_f)

        # ── Places-sourced wm_* fields (no existing qodef_ equivalent)
        for wm_key in ("wm_place_id", "wm_rating", "wm_review_count",
                       "wm_parking", "wm_accessible", "wm_business_status"):
            if meta.get(wm_key):
                new_meta[wm_key] = meta[wm_key]

        # ── Backfill phone/website from Places into existing qodef_ fields
        if meta.get("qodef_listing_single_phone") and not orig_meta.get("qodef_listing_single_phone"):
            new_meta["qodef_listing_single_phone"] = meta["qodef_listing_single_phone"]
        if meta.get("qodef_listing_single_site_url") and not orig_meta.get("qodef_listing_single_site_url"):
            new_meta["qodef_listing_single_site_url"] = meta["qodef_listing_single_site_url"]

        # ── Fetch og:image from venue website (if no featured image yet)
        if not venue.get("featured_media"):
            website_url = (
                meta.get("qodef_listing_single_site_url") or
                (place_data.get("website") if place_data else None)
            )
            og_image = fetch_og_image(website_url) if website_url else None
            if og_image:
                print(f"    Image: {og_image[:80]}")
                wp.sideload_featured_image(post_id, og_image, dry_run=args.dry_run)
            else:
                print(f"    Image: none found")

        # ── Write back
        if wp.update_venue_meta(post_id, new_meta, dry_run=args.dry_run):
            enriched_count += 1
            already_done.add(str(post_id))
            progress["enriched"].append(str(post_id))
            save_progress(progress)
            print(f"    ✓ enriched (tier {tier})")
            return True
        else:
            error_count += 1
            progress["failed"].append(str(post_id))
            save_progress(progress)
            return True  # did real work, still throttle

    # ── Process venues
    if venues_to_process is not None:
        for v in venues_to_process:
            process_venue(v)
    else:
        page = 1
        while True:
            if args.limit and enriched_count >= args.limit:
                print(f"\nLimit of {args.limit} reached.")
                break

            try:
                venues = wp.get_venues_page(page, config["per_page"])
            except requests.HTTPError as e:
                print(f"REST API error on page {page}: {e}")
                break

            if not venues:
                print("No more venues.")
                break

            skipped = 0
            worked  = 0
            for v in venues:
                if args.limit and enriched_count >= args.limit:
                    break
                did_work = process_venue(v)
                if did_work:
                    worked += 1
                    time.sleep(config["delay_between_venues"])
                else:
                    skipped += 1

            # Only print the page header when there's something to report
            if worked or page == 1:
                label = f"── Page {page} ──"
                parts = []
                if worked:
                    parts.append(f"enriched {worked}")
                if skipped:
                    parts.append(f"skipped {skipped} already done")
                print(f"\n{label} {', '.join(parts)}" if parts else f"\n{label}")
            elif skipped:
                # Quiet pages with only skips: print a dot so the user sees progress
                print(".", end="", flush=True)

            page += 1
            time.sleep(config["delay_between_pages"])

    # ── Summary
    print(f"\n{'='*50}")
    print(f"Done.")
    print(f"  Enriched:  {enriched_count}")
    print(f"  Errors:    {error_count}")
    if args.dry_run:
        print("  (Dry run — no changes were written)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enrich Wedding Maps venue listings with AI-generated content"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing to WordPress"
    )
    parser.add_argument(
        "--tier", type=int, choices=[1, 2, 3, 4],
        help="Only process venues in this priority tier"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Stop after enriching this many venues"
    )
    parser.add_argument(
        "--post-id", type=int, dest="post_id",
        help="Enrich a single venue by WordPress post ID"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-enrich venues that already have wm_enriched_at set"
    )
    parser.add_argument(
        "--no-places", action="store_true", dest="no_places",
        help="Skip Google Places API calls (use when monthly credit is exhausted)"
    )
    args = parser.parse_args()

    config = load_config()

    # Validate required config
    missing = []
    for key in ("wm_secret", "anthropic_api_key"):
        if not config.get(key):
            missing.append(key)
    if missing:
        print("ERROR: Missing required config keys:", ", ".join(missing))
        print(f"Create {CONFIG_FILE} or set environment variables.")
        sys.exit(1)

    run_pipeline(args, config)


if __name__ == "__main__":
    main()
