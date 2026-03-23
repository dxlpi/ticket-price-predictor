"""Vivid Seats scraper using Playwright browser automation.

Collects real seat-level ticket listings with pricing data.
"""

import asyncio
import json
import logging
import random
import re
from datetime import UTC, datetime
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Response, async_playwright
from playwright_stealth import Stealth

from ticket_price_predictor.schemas import ScrapedEvent, ScrapedListing

logger = logging.getLogger(__name__)


# VividSeats hermes API category IDs
# Category 2 = concerts is confirmed. Others to be expanded after API probing.
CATEGORY_IDS: dict[str, int] = {
    "concerts": 2,
}

CATEGORY_TO_EVENT_TYPE: dict[int, str] = {
    2: "concert",
}


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

    async def _fetch_productions_page(
        self,
        category_id: int,
        page: int,
        rows: int,
    ) -> dict[str, Any]:
        """Fetch a page of productions from the hermes API via browser context.

        Args:
            category_id: Category ID (2 = concerts)
            page: 1-based page number
            rows: Number of results per page

        Returns:
            Parsed JSON response dict
        """
        if not self._page:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        url = (
            f"{self.BASE_URL}/hermes/api/v1/productions"
            f"?categoryId={category_id}&rows={rows}&page={page}"
        )
        result: dict[str, Any] = await self._page.evaluate(
            """async (url) => {
                const resp = await fetch(url, {
                    headers: {
                        'Accept': 'application/json',
                        'Accept-Language': 'en-US,en;q=0.9',
                    },
                    credentials: 'include',
                });
                if (!resp.ok) {
                    throw new Error('HTTP ' + resp.status);
                }
                return await resp.json();
            }""",
            url,
        )
        return result

    async def browse_events(
        self,
        category_ids: list[int] | None = None,
        max_pages: int = 10,
        max_events: int = 200,
        rows_per_page: int = 25,
    ) -> list[ScrapedEvent]:
        """Browse upcoming events via the hermes productions API.

        Supports multiple event categories by looping over category IDs.
        Events are deduplicated by ID across categories.

        Args:
            category_ids: Category IDs to browse. None = all known IDs from CATEGORY_IDS.
            max_pages: Maximum API pages per category
            max_events: Total events cap across all categories
            rows_per_page: Results per API page (max 25)

        Returns:
            List of ScrapedEvent objects with event_type populated
        """
        if not self._page:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        if category_ids is None:
            category_ids = list(CATEGORY_TO_EVENT_TYPE.keys())

        # Navigate to the site first so browser cookies/session are established
        await self._page.goto(self.BASE_URL, wait_until="domcontentloaded", timeout=60000)
        await self._delay_random()

        events: list[ScrapedEvent] = []
        seen_ids: set[str] = set()

        for category_id in category_ids:
            if len(events) >= max_events:
                break

            event_type = CATEGORY_TO_EVENT_TYPE.get(category_id, "concert")
            is_concert = category_id == 2

            for page_num in range(1, max_pages + 1):
                if len(events) >= max_events:
                    break

                logger.debug(f"Fetching category {category_id} page {page_num}")
                try:
                    data = await self._fetch_productions_page(category_id, page_num, rows_per_page)
                except Exception as e:
                    logger.warning(
                        f"Productions API category {category_id} page {page_num} failed: {e}"
                    )
                    break

                items = data.get("items", [])
                if not items:
                    logger.debug("No items returned — stopping pagination")
                    break

                for item in items:
                    if len(events) >= max_events:
                        break

                    # Skip parking listings
                    if item.get("subCategoryId") == 75:
                        continue

                    # Determine artist/team name based on category
                    if is_concert:
                        # Concerts: require a master performer
                        performers = item.get("performers", [])
                        artist_or_team = next(
                            (p["name"] for p in performers if p.get("master")),
                            None,
                        )
                        if artist_or_team is None:
                            logger.debug(
                                f"Skipped event with no master performer: {item.get('name')}"
                            )
                            continue
                    else:
                        # Sports/other: use event name directly
                        artist_or_team = item.get("name", "Unknown Event")

                    try:
                        event_id = str(item.get("id", ""))

                        # Deduplicate across categories
                        if event_id in seen_ids:
                            continue
                        seen_ids.add(event_id)

                        name = item.get("name", "Unknown Event")
                        venue = item.get("venue", {})
                        venue_name = venue.get("name", "Unknown Venue")
                        city = venue.get("city", "Unknown")
                        state = venue.get("state", "")

                        # Parse date — format: 2026-08-01T12:55:00-04:00[America/New_York]
                        local_date = item.get("localDate", "")
                        try:
                            date_str = local_date.split("[")[0] if "[" in local_date else local_date
                            event_dt = datetime.fromisoformat(date_str)
                        except (ValueError, TypeError):
                            event_dt = datetime.now(UTC)

                        web_path = item.get("webPath", "")
                        event_url = f"{self.BASE_URL}{web_path}" if web_path else ""

                        min_price = item.get("minPrice")
                        ticket_count = item.get("ticketCount")

                        event = ScrapedEvent(
                            stubhub_event_id=event_id,
                            event_name=name,
                            artist_or_team=artist_or_team,
                            venue_name=venue_name,
                            city=f"{city}, {state}".strip(", "),
                            event_datetime=event_dt,
                            event_url=event_url,
                            min_price=float(min_price) if min_price else None,
                            ticket_count=int(ticket_count) if ticket_count else None,
                            event_type=event_type,
                        )
                        events.append(event)

                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug(f"Skipped event item: {e}")
                        continue

                total_pages = data.get("numberOfPages", 1)
                logger.info(
                    f"Category {category_id} page {page_num}/{total_pages}: "
                    f"collected {len(events)} events so far"
                )

                if page_num >= total_pages:
                    break

                await self._delay_random()

        logger.info(f"browse_events() returned {len(events)} events")
        return events

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
                local_date = item.get("localDate", "")
                try:
                    # Format: 2026-08-01T12:55:00-04:00[America/New_York]
                    date_str = local_date.split("[")[0] if "[" in local_date else local_date
                    event_dt = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    event_dt = datetime.now(UTC)

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

            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipped event item: {e}")
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

        logger.debug(f"Navigating to {event_url}")

        # Capture the listings API response
        listings_data: dict[str, Any] | None = None
        via_api = False

        async def capture_listings(response: Response) -> None:
            nonlocal listings_data, via_api
            if (
                "/hermes/api/v" in response.url
                and "/listings" in response.url
                and "productionId" in response.url
            ):
                try:
                    if "json" in response.headers.get("content-type", ""):
                        listings_data = await response.json()
                        via_api = True
                except Exception:
                    pass

        self._page.on("response", capture_listings)

        # Navigate to event page with extended timeout
        try:
            await self._page.goto(event_url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning(f"Page navigation timeout for {event_url}: {e}")
            self._page.remove_listener("response", capture_listings)
            return []

        # Check for redirects (VividSeats redirects expired events to random pages)
        final_url = self._page.url
        # Extract production ID from original URL
        original_prod_id = (
            event_url.split("/production/")[-1].split("/")[0]
            if "/production/" in event_url
            else None
        )
        if original_prod_id and f"/production/{original_prod_id}" not in final_url:
            logger.warning(f"URL redirected from {event_url} to {final_url} - event may be expired")
            self._page.remove_listener("response", capture_listings)
            return []

        # Wait for listings API specifically (this is the key wait)
        try:
            await self._page.wait_for_event(
                "response",
                predicate=lambda r: (
                    "/hermes/api/v" in r.url and "/listings" in r.url and "productionId" in r.url
                ),
                timeout=25000,
            )
        except Exception:
            # API call may have already happened or may not exist
            logger.debug("Listings API wait timed out, checking captured data")

        # Additional wait for any pending responses
        await asyncio.sleep(5)

        # Remove listener
        self._page.remove_listener("response", capture_listings)

        # Fallback: try __NEXT_DATA__ if API capture failed
        if not listings_data:
            listings_data = await self._extract_from_next_data()
            via_api = False

        # Retry logic: scroll and try again if no listings captured
        if not listings_data:
            logger.debug("No listings captured, retrying with scroll")
            # Scroll to trigger lazy load
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(3)

            # Try __NEXT_DATA__ again after scroll
            listings_data = await self._extract_from_next_data()

        if not listings_data:
            logger.warning(f"No listings data captured for {event_url}")
            return []

        listings = self._extract_listings(listings_data, max_listings)
        logger.info(
            f"Captured {len(listings)} listings via {'API' if via_api else '__NEXT_DATA__'}"
        )
        return listings

    async def _extract_from_next_data(self) -> dict[str, Any] | None:
        """Fallback: extract listings from __NEXT_DATA__ script tag."""
        if not self._page:
            return None

        try:
            html = await self._page.content()
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)
            if not match:
                return None

            data = json.loads(match.group(1))
            page_props = data.get("props", {}).get("pageProps", {})

            # Check if listings are embedded in page props
            if "initialListingsData" in page_props:
                return page_props["initialListingsData"]  # type: ignore[no-any-return]

            # Alternative: check for tickets directly
            if "tickets" in page_props:
                return {"tickets": page_props["tickets"]}

        except (json.JSONDecodeError, AttributeError, KeyError):
            pass

        return None

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

            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipped listing ticket: {e}")
                continue

        return listings
