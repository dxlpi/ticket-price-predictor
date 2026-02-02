"""StubHub scraper using Playwright browser automation.

Uses a real browser to bypass anti-bot protection and render JavaScript content.
Reuses parsing logic from the httpx-based scraper.
"""

import asyncio
import random
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

from ticket_price_predictor.schemas import ScrapedEvent, ScrapedListing
from ticket_price_predictor.scrapers.stubhub import StubHubScraper


class PlaywrightStubHubScraper:
    """Playwright-based StubHub scraper for bypassing anti-bot protection.

    Uses a real Chromium browser to:
    - Render JavaScript-heavy pages
    - Bypass Cloudflare and other anti-bot systems
    - Extract data from dynamically-loaded content
    """

    BASE_URL = "https://www.stubhub.com"

    def __init__(
        self,
        headless: bool = True,
        delay_seconds: float = 3.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Playwright scraper.

        Args:
            headless: Run browser in headless mode (set False for debugging)
            delay_seconds: Base delay between requests
            max_retries: Maximum retry attempts on failure
        """
        self._headless = headless
        self._delay = delay_seconds
        self._max_retries = max_retries
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        # Reuse parsing logic from httpx scraper
        self._parser = StubHubScraper()

    async def __aenter__(self) -> "PlaywrightStubHubScraper":
        """Enter async context - launch browser."""
        playwright = await async_playwright().start()
        self._playwright = playwright

        # Launch browser with anti-detection settings
        self._browser = await playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu",
            ],
        )

        # Create browser context with realistic settings
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/Los_Angeles",
            java_script_enabled=True,
            has_touch=False,
            is_mobile=False,
        )

        self._page = await self._context.new_page()

        # Apply stealth plugin to hide automation signals
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

    async def _navigate(self, url: str) -> str:
        """Navigate to URL and return page content.

        Args:
            url: URL to navigate to

        Returns:
            Page HTML content after JavaScript rendering
        """
        if not self._page:
            raise RuntimeError("Browser not initialized. Use 'async with' context.")

        for attempt in range(self._max_retries):
            try:
                await self._delay_random()

                # Navigate with timeout
                response = await self._page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )

                if response and response.status >= 400:
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"HTTP {response.status} for {url}")

                # Wait for page to fully load
                await self._page.wait_for_load_state("networkidle", timeout=15000)

                # Additional wait for dynamic content
                await asyncio.sleep(1.0)

                return await self._page.content()

            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise RuntimeError(f"Failed to load {url}: {e}") from e
                await asyncio.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Failed to load {url} after {self._max_retries} attempts")

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
        search_url = f"{self.BASE_URL}/search?q={query.replace(' ', '+')}"
        html = await self._navigate(search_url)

        # Reuse parsing logic from httpx scraper
        events = self._parser._parse_search_results(html, max_results)

        return events

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
        html = await self._navigate(event_url)

        # Try to scroll and load more listings
        if self._page:
            await self._scroll_to_load_more()

            # Get updated HTML after scrolling
            html = await self._page.content()

        # Reuse parsing logic from httpx scraper
        listings = self._parser._parse_listings_page(html)

        return listings[:max_listings]

    async def _scroll_to_load_more(self, max_scrolls: int = 5) -> None:
        """Scroll down to load dynamically-loaded listings.

        Args:
            max_scrolls: Maximum number of scroll attempts
        """
        if not self._page:
            return

        for _ in range(max_scrolls):
            # Get current scroll height
            prev_height = await self._page.evaluate("document.body.scrollHeight")

            # Scroll to bottom
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for new content to load
            await asyncio.sleep(1.5)

            # Check if we've reached the bottom
            new_height = await self._page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break
