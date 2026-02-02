"""Vivid Seats scraper using Playwright browser automation.

Collects real seat-level ticket listings with pricing data.
"""

import asyncio
import json
import random
import re
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Response, async_playwright
from playwright_stealth import Stealth

from ticket_price_predictor.schemas import ScrapedEvent, ScrapedListing


class VividSeatsScraper:
    """Playwright-based Vivid Seats scraper.

    Extracts seat-level ticket listings including:
    - Section, row, seat numbers
    - Price per ticket
    - Quantity available
    """

    BASE_URL = "https://www.vividseats.com"

    def __init__(
        self,
        headless: bool = True,
        delay_seconds: float = 3.0,
    ) -> None:
        """Initialize the Vivid Seats scraper.

        Args:
            headless: Run browser in headless mode
            delay_seconds: Base delay between requests
        """
        self._headless = headless
        self._delay = delay_seconds
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self) -> "VividSeatsScraper":
        """Enter async context - launch browser."""
        playwright = await async_playwright().start()
        self._playwright = playwright

        self._browser = await playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/Los_Angeles",
        )

        self._page = await self._context.new_page()

        # Apply stealth to avoid detection
        stealth = Stealth()
        await stealth.apply_stealth_async(self._page)

        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context - close browser."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if hasattr(self, "_playwright"):
            await self._playwright.stop()

    async def _delay_random(self) -> None:
        """Add random delay between actions."""
        delay = self._delay + random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)

    async def search_events(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[ScrapedEvent]:
        """Search for events/performers.

        Args:
            query: Search query (artist name, team name, etc.)
            max_results: Maximum number of events to return

        Returns:
            List of ScrapedEvent objects
        """
        if not self._page:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        await self._delay_random()

        # Navigate to search
        search_url = f"{self.BASE_URL}/search?searchTerm={query.replace(' ', '+')}"
        await self._page.goto(search_url, wait_until="load")
        await asyncio.sleep(3)

        html = await self._page.content()

        # Extract __NEXT_DATA__
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)
        if not match:
            return []

        data = json.loads(match.group(1))
        page_props = data.get("props", {}).get("pageProps", {})

        # Check if we got redirected to a performer page
        if "initialProductionListData" in page_props:
            return self._extract_events_from_production_list(page_props, max_results)

        return []

    def _extract_events_from_production_list(
        self,
        page_props: dict[str, Any],
        max_results: int,
    ) -> list[ScrapedEvent]:
        """Extract events from production list data."""
        events: list[ScrapedEvent] = []
        prod_data = page_props.get("initialProductionListData", {})
        items = prod_data.get("items", [])

        for item in items[:max_results]:
            try:
                event_id = str(item.get("id", ""))
                name = item.get("name", "Unknown Event")
                venue = item.get("venue", {})
                venue_name = venue.get("name", "Unknown Venue")
                city = venue.get("city", "Unknown")
                state = venue.get("state", "")

                # Parse date
                from datetime import datetime

                local_date = item.get("localDate", "")
                try:
                    # Format: 2026-08-01T12:55:00-04:00[America/New_York]
                    date_str = local_date.split("[")[0] if "[" in local_date else local_date
                    event_dt = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    event_dt = datetime.now()

                # Build URL
                web_path = item.get("webPath", "")
                event_url = f"{self.BASE_URL}{web_path}" if web_path else ""

                # Price info
                min_price = item.get("minPrice")
                ticket_count = item.get("ticketCount")

                event = ScrapedEvent(
                    stubhub_event_id=event_id,  # Reusing field for Vivid Seats ID
                    event_name=name,
                    artist_or_team=name.split(" at ")[0] if " at " in name else name,
                    venue_name=venue_name,
                    city=f"{city}, {state}".strip(", "),
                    event_datetime=event_dt,
                    event_url=event_url,
                    min_price=float(min_price) if min_price else None,
                    ticket_count=int(ticket_count) if ticket_count else None,
                )
                events.append(event)

            except (KeyError, ValueError, TypeError):
                continue

        return events

    async def get_event_listings(
        self,
        event_url: str,
        max_listings: int = 500,
    ) -> list[ScrapedListing]:
        """Get seat-level listings for an event.

        Args:
            event_url: Vivid Seats event page URL
            max_listings: Maximum listings to fetch

        Returns:
            List of ScrapedListing objects
        """
        if not self._page:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        await self._delay_random()

        # Capture the listings API response
        listings_data: dict[str, Any] | None = None

        async def capture_listings(response: Response) -> None:
            nonlocal listings_data
            if "/hermes/api/v1/listings?" in response.url and "productionId" in response.url:
                try:
                    if "json" in response.headers.get("content-type", ""):
                        listings_data = await response.json()
                except Exception:
                    pass

        self._page.on("response", capture_listings)

        # Navigate to event page
        await self._page.goto(event_url, wait_until="load")
        await asyncio.sleep(8)  # Wait for API calls

        # Remove listener
        self._page.remove_listener("response", capture_listings)

        if not listings_data:
            return []

        return self._extract_listings(listings_data, max_listings)

    def _extract_listings(
        self,
        data: dict[str, Any],
        max_listings: int,
    ) -> list[ScrapedListing]:
        """Extract listings from API response."""
        listings: list[ScrapedListing] = []
        tickets = data.get("tickets", [])

        for ticket in tickets[:max_listings]:
            try:
                # Extract fields using short keys or full names
                listing_id = ticket.get("i", ticket.get("id", ""))
                section = ticket.get("sectionName", ticket.get("s", "Unknown"))
                row = ticket.get("row", ticket.get("r", "GA"))

                # Quantity
                qty_str = ticket.get("quantity", ticket.get("q", "1"))
                quantity = int(qty_str) if qty_str else 1

                # Price
                price_str = ticket.get("allInPricePerTicket", ticket.get("p", "0"))
                price = float(price_str) if price_str else 0.0

                # Seats
                low_seat = ticket.get("ls")
                high_seat = ticket.get("hs")

                listing = ScrapedListing(
                    listing_id=str(listing_id),
                    section=section,
                    row=row,
                    seat_from=str(low_seat) if low_seat else None,
                    seat_to=str(high_seat) if high_seat else None,
                    quantity=quantity,
                    price_per_ticket=price,
                    total_price=price * quantity,
                    face_value=None,  # Vivid Seats doesn't provide face value
                )
                listings.append(listing)

            except (KeyError, ValueError, TypeError):
                continue

        return listings
