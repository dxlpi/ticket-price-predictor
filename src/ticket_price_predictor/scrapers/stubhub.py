"""StubHub web scraper for ticket listings.

Scrapes ticket listings from StubHub website to collect:
- Event search results
- Individual ticket listings with seat-level detail
- Pricing information (face value and listing price)
"""

import asyncio
import contextlib
import json
import random
import re
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import httpx
from selectolax.parser import HTMLParser

from ticket_price_predictor.schemas import ScrapedEvent, ScrapedListing

# User agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class StubHubScraper:
    """Scrapes ticket listings from StubHub website.

    Uses StubHub's internal API endpoints where possible, falling back
    to HTML parsing when needed. Implements polite scraping with delays
    and user agent rotation.
    """

    BASE_URL = "https://www.stubhub.com"
    API_BASE = "https://www.stubhub.com/api"

    def __init__(
        self,
        delay_seconds: float = 2.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the scraper.

        Args:
            delay_seconds: Delay between requests (be polite)
            max_retries: Maximum retry attempts on failure
        """
        self._delay = delay_seconds
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "StubHubScraper":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers=self._get_headers(),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with random user agent."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    async def _request(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """Make a request with retry logic and delay."""
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        # Update headers for each request (rotate user agent)
        self._client.headers.update(self._get_headers())

        for attempt in range(self._max_retries):
            try:
                # Polite delay
                await asyncio.sleep(self._delay + random.uniform(0, 1))

                response = await self._client.get(url, params=params)
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait = (attempt + 1) * 10  # Exponential backoff
                    await asyncio.sleep(wait)
                elif e.response.status_code >= 500:  # Server error
                    await asyncio.sleep(5)
                else:
                    raise
            except httpx.RequestError:
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(5)

        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries} attempts")

    async def search_events(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[ScrapedEvent]:
        """Search for events by artist name or query.

        Args:
            query: Search query (artist name, event name, etc.)
            max_results: Maximum number of events to return

        Returns:
            List of ScrapedEvent objects
        """
        # Try the search API first
        search_url = f"{self.BASE_URL}/search"
        params = {"q": query}

        response = await self._request(search_url, params)
        html = response.text

        # Parse the search results page
        events = self._parse_search_results(html, max_results)

        return events

    def _parse_search_results(self, html: str, max_results: int) -> list[ScrapedEvent]:
        """Parse search results HTML to extract events."""
        events: list[ScrapedEvent] = []
        parser = HTMLParser(html)

        # Look for event data in JSON script tags
        for script in parser.css("script"):
            text = script.text() or ""
            if "__NEXT_DATA__" in text or "application/json" in str(script.attributes):
                try:
                    # Extract JSON from script
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        events.extend(self._extract_events_from_json(data, max_results))
                        if len(events) >= max_results:
                            return events[:max_results]
                except (json.JSONDecodeError, KeyError):
                    continue

        # Fallback: parse HTML directly
        for card in parser.css("[data-testid='event-card'], .event-card, .EventItem"):
            try:
                event = self._parse_event_card(card)
                if event:
                    events.append(event)
                    if len(events) >= max_results:
                        break
            except (ValueError, KeyError):
                continue

        return events[:max_results]

    def _extract_events_from_json(
        self, data: dict[str, Any], max_results: int
    ) -> list[ScrapedEvent]:
        """Extract events from JSON data structure."""
        events: list[ScrapedEvent] = []

        # Navigate through possible JSON structures
        def find_events(obj: Any, depth: int = 0) -> None:
            if depth > 10 or len(events) >= max_results:
                return

            if isinstance(obj, dict):
                # Check if this looks like an event
                if "eventId" in obj or "id" in obj:
                    event = self._dict_to_event(obj)
                    if event:
                        events.append(event)
                        return

                # Recurse into dict values
                for value in obj.values():
                    find_events(value, depth + 1)

            elif isinstance(obj, list):
                for item in obj:
                    find_events(item, depth + 1)

        find_events(data)
        return events

    def _dict_to_event(self, data: dict[str, Any]) -> ScrapedEvent | None:
        """Convert a dictionary to ScrapedEvent."""
        try:
            event_id = str(data.get("eventId") or data.get("id", ""))
            if not event_id:
                return None

            # Extract event name
            name = data.get("name") or data.get("title") or data.get("eventName", "Unknown")

            # Extract performer/artist
            performers = data.get("performers") or data.get("acts") or []
            if isinstance(performers, list) and performers:
                artist = (
                    performers[0].get("name", name)
                    if isinstance(performers[0], dict)
                    else str(performers[0])
                )
            else:
                artist = data.get("performerName") or name

            # Extract venue
            venue_data = data.get("venue") or {}
            venue_name = venue_data.get("name") or data.get("venueName", "Unknown Venue")
            city = venue_data.get("city") or data.get("city", "Unknown")

            # Extract date
            date_str = data.get("eventDate") or data.get("date") or data.get("startDate", "")
            try:
                if "T" in str(date_str):
                    event_dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
                else:
                    event_dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                event_dt = datetime.now()

            # Build URL
            url = data.get("url") or data.get("eventUrl") or f"/event/{event_id}"
            if not url.startswith("http"):
                url = urljoin(self.BASE_URL, url)

            # Extract pricing info
            min_price = data.get("minPrice") or data.get("minListPrice")
            ticket_count = data.get("ticketCount") or data.get("totalTickets")

            return ScrapedEvent(
                stubhub_event_id=event_id,
                event_name=str(name),
                artist_or_team=str(artist),
                venue_name=str(venue_name),
                city=str(city),
                event_datetime=event_dt,
                event_url=url,
                min_price=float(min_price) if min_price else None,
                ticket_count=int(ticket_count) if ticket_count else None,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _parse_event_card(self, node: Any) -> ScrapedEvent | None:
        """Parse an event card HTML element."""
        try:
            # Extract link and event ID
            link = node.css_first("a")
            if not link:
                return None

            href = link.attributes.get("href", "")
            event_id_match = re.search(r"/event/(\d+)", href)
            if not event_id_match:
                return None

            event_id = event_id_match.group(1)

            # Extract text content
            name_el = node.css_first(".event-name, h3, [data-testid='event-name']")
            name = name_el.text().strip() if name_el else "Unknown"

            venue_el = node.css_first(".venue-name, [data-testid='venue']")
            venue = venue_el.text().strip() if venue_el else "Unknown Venue"

            date_el = node.css_first(".event-date, time, [data-testid='date']")
            date_str = date_el.text().strip() if date_el else ""

            # Parse date (various formats)
            event_dt = datetime.now()
            if date_str:
                with contextlib.suppress(ValueError):
                    event_dt = datetime.strptime(date_str, "%b %d, %Y")

            return ScrapedEvent(
                stubhub_event_id=event_id,
                event_name=name,
                artist_or_team=name.split(" at ")[0] if " at " in name else name,
                venue_name=venue,
                city="Unknown",
                event_datetime=event_dt,
                event_url=urljoin(self.BASE_URL, href),
            )
        except (AttributeError, ValueError):
            return None

    async def get_event_listings(
        self,
        event_url: str,
        max_listings: int = 500,
    ) -> list[ScrapedListing]:
        """Get all ticket listings for an event.

        Args:
            event_url: StubHub event page URL
            max_listings: Maximum listings to fetch

        Returns:
            List of ScrapedListing objects
        """
        response = await self._request(event_url)
        html = response.text

        listings = self._parse_listings_page(html)

        # If we need more and there's pagination, fetch more pages
        # (StubHub often loads listings via JS, so we may get limited results)

        return listings[:max_listings]

    def _parse_listings_page(self, html: str) -> list[ScrapedListing]:
        """Parse listings from event page HTML."""
        listings: list[ScrapedListing] = []
        parser = HTMLParser(html)

        # Try to find listings in JSON data first
        for script in parser.css("script"):
            text = script.text() or ""
            if "listings" in text.lower() or "__NEXT_DATA__" in text:
                try:
                    json_match = re.search(r"\{.*\}", text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        listings.extend(self._extract_listings_from_json(data))
                except (json.JSONDecodeError, KeyError):
                    continue

        # Fallback: parse HTML listings
        if not listings:
            for row in parser.css("[data-testid='listing-row'], .listing-row, .TicketCard"):
                try:
                    listing = self._parse_listing_row(row)
                    if listing:
                        listings.append(listing)
                except (ValueError, KeyError):
                    continue

        return listings

    def _extract_listings_from_json(self, data: dict[str, Any]) -> list[ScrapedListing]:
        """Extract listings from JSON data."""
        listings: list[ScrapedListing] = []

        def find_listings(obj: Any, depth: int = 0) -> None:
            if depth > 15:
                return

            if isinstance(obj, dict):
                # Check if this looks like a listing
                if "listingId" in obj or ("section" in obj and "price" in obj):
                    listing = self._dict_to_listing(obj)
                    if listing:
                        listings.append(listing)
                        return

                # Check for listings array
                if "listings" in obj and isinstance(obj["listings"], list):
                    for item in obj["listings"]:
                        if isinstance(item, dict):
                            listing = self._dict_to_listing(item)
                            if listing:
                                listings.append(listing)
                    return

                # Recurse
                for value in obj.values():
                    find_listings(value, depth + 1)

            elif isinstance(obj, list):
                for item in obj:
                    find_listings(item, depth + 1)

        find_listings(data)
        return listings

    def _dict_to_listing(self, data: dict[str, Any]) -> ScrapedListing | None:
        """Convert a dictionary to ScrapedListing."""
        try:
            listing_id = str(data.get("listingId") or data.get("id", ""))
            if not listing_id:
                return None

            section = str(data.get("section") or data.get("sectionName", "Unknown"))
            row = str(data.get("row") or data.get("rowName", "GA"))

            # Seats
            seat_from = data.get("seatFrom") or data.get("lowSeat")
            seat_to = data.get("seatTo") or data.get("highSeat")

            # Quantity
            quantity = int(data.get("quantity") or data.get("ticketQuantity", 1))

            # Price - look for various price fields
            price_data = data.get("price") or data.get("pricePerTicket") or data
            if isinstance(price_data, dict):
                price = float(price_data.get("amount") or price_data.get("value", 0))
            else:
                price = float(price_data) if price_data else 0

            total = float(data.get("totalPrice") or data.get("totalWithFees") or price * quantity)

            face_value = data.get("faceValue")
            if isinstance(face_value, dict):
                face_value = face_value.get("amount")

            return ScrapedListing(
                listing_id=listing_id,
                section=section,
                row=row,
                seat_from=str(seat_from) if seat_from else None,
                seat_to=str(seat_to) if seat_to else None,
                quantity=quantity,
                price_per_ticket=price,
                total_price=total,
                face_value=float(face_value) if face_value else None,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _parse_listing_row(self, node: Any) -> ScrapedListing | None:
        """Parse a listing row HTML element."""
        try:
            # Extract section and row
            section_el = node.css_first(".section, [data-testid='section']")
            section = section_el.text().strip() if section_el else "Unknown"

            row_el = node.css_first(".row, [data-testid='row']")
            row = row_el.text().strip() if row_el else "GA"

            # Extract price
            price_el = node.css_first(".price, [data-testid='price']")
            price_text = price_el.text().strip() if price_el else "0"
            price = float(re.sub(r"[^\d.]", "", price_text) or 0)

            # Extract quantity
            qty_el = node.css_first(".quantity, [data-testid='quantity']")
            qty_text = qty_el.text().strip() if qty_el else "1"
            quantity = int(re.sub(r"[^\d]", "", qty_text) or 1)

            # Generate listing ID from content
            listing_id = f"{section}_{row}_{price}".replace(" ", "_")

            return ScrapedListing(
                listing_id=listing_id,
                section=section,
                row=row,
                quantity=quantity,
                price_per_ticket=price,
                total_price=price * quantity,
            )
        except (AttributeError, ValueError):
            return None
