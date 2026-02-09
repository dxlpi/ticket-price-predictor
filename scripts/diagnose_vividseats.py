#!/usr/bin/env python3
"""Diagnostic script to investigate VividSeats data structure.

Usage:
    python scripts/diagnose_vividseats.py "https://www.vividseats.com/bruno-mars-tickets-..."
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from playwright.async_api import async_playwright, Page, Route


class VividSeatsDiagnostic:
    """Diagnostic tool for VividSeats page structure."""

    BASE_URL = "https://www.vividseats.com"

    def __init__(self):
        self.network_requests = []
        self.next_data = None

    async def search_and_find_event(self, query: str) -> str | None:
        """Search for events and return first event URL with listings.

        Args:
            query: Search query (artist name, etc.)

        Returns:
            Event URL if found, None otherwise
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to search
            search_url = f"{self.BASE_URL}/search?searchTerm={query.replace(' ', '+')}"
            print(f"Navigating to search: {search_url}")
            await page.goto(search_url, wait_until="load")
            await asyncio.sleep(3)

            # Extract __NEXT_DATA__
            html = await page.content()
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)
            if not match:
                await browser.close()
                return None

            data = json.loads(match.group(1))
            page_props = data.get("props", {}).get("pageProps", {})

            # Look for production list
            prod_list = page_props.get("initialProductionListData", {})
            items = prod_list.get("items", [])

            print(f"Found {len(items)} events in search results")

            if items:
                # Get first event
                first_event = items[0]
                production_id = first_event.get("productionId") or first_event.get("id")
                web_path = first_event.get("webPath")

                if web_path:
                    event_url = f"{self.BASE_URL}{web_path}"
                    print(f"  Event: {first_event.get('name', 'Unknown')}")
                    print(f"  Production ID: {production_id}")
                    print(f"  Date: {first_event.get('eventDateLocal', 'Unknown')}")
                    await browser.close()
                    return event_url

            await browser.close()
            return None

    async def capture_network_request(self, route: Route):
        """Capture network request details."""
        request = route.request

        try:
            # Continue the request
            response = await route.fetch()

            # Get response body
            body = None
            body_size = 0
            if response.ok:
                try:
                    body_bytes = await response.body()
                    body_size = len(body_bytes)

                    # Try to parse JSON for API calls
                    if "/hermes/api/v1/listings" in request.url:
                        try:
                            body = body_bytes.decode('utf-8')
                        except:
                            pass
                except:
                    pass

            # Capture details
            request_data = {
                "url": request.url,
                "method": request.method,
                "resource_type": request.resource_type,
                "status": response.status,
                "size": body_size,
                "body": body if "/hermes/api/v1/listings" in request.url else None,
            }

            self.network_requests.append(request_data)

            # Fulfill the request
            await route.fulfill(response=response)

        except Exception as e:
            print(f"Error capturing request {request.url}: {e}")
            await route.continue_()

    async def diagnose_page(self, url: str) -> dict:
        """Navigate to page and extract diagnostic information.

        Args:
            url: Event URL to diagnose

        Returns:
            Dictionary with diagnostic data
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Set up network interception
            await page.route("**/*", self.capture_network_request)

            print(f"Navigating to: {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for dynamic content
            await asyncio.sleep(3)

            # Extract HTML
            html = await page.content()

            # Extract __NEXT_DATA__
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)
            if match:
                try:
                    self.next_data = json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    print(f"Error parsing __NEXT_DATA__: {e}")

            await browser.close()

        return {
            "network_requests": self.network_requests,
            "next_data": self.next_data,
        }

    def analyze_next_data(self) -> dict:
        """Analyze __NEXT_DATA__ structure for listing data.

        Returns:
            Dictionary with analysis results
        """
        if not self.next_data:
            return {"error": "__NEXT_DATA__ not found"}

        analysis = {
            "top_level_keys": list(self.next_data.keys()),
            "page_props_keys": [],
            "potential_listing_keys": [],
            "sample_data": {},
        }

        # Analyze pageProps
        page_props = self.next_data.get("props", {}).get("pageProps", {})
        if page_props:
            analysis["page_props_keys"] = list(page_props.keys())

            # Look for listing-related keys
            listing_keywords = ["ticket", "listing", "inventory", "seat", "offer", "price", "zone"]
            for key in page_props.keys():
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in listing_keywords):
                    analysis["potential_listing_keys"].append(key)
                    # Sample the data structure
                    data = page_props[key]
                    if isinstance(data, (list, dict)):
                        analysis["sample_data"][key] = self._sample_structure(data)

        return analysis

    def _sample_structure(self, data, max_items=3, max_depth=3, current_depth=0):
        """Create a sample of the data structure.

        Args:
            data: Data to sample
            max_items: Maximum items to show in lists/dicts
            max_depth: Maximum depth to traverse
            current_depth: Current recursion depth

        Returns:
            Sampled structure
        """
        if current_depth >= max_depth:
            return f"<{type(data).__name__}>"

        if isinstance(data, dict):
            return {
                k: self._sample_structure(v, max_items, max_depth, current_depth + 1)
                for k, v in list(data.items())[:max_items]
            }
        elif isinstance(data, list):
            if not data:
                return []
            return [
                self._sample_structure(item, max_items, max_depth, current_depth + 1)
                for item in data[:max_items]
            ]
        else:
            return data

    def print_network_summary(self):
        """Print summary of captured network requests."""
        print("\n" + "=" * 80)
        print("NETWORK REQUESTS")
        print("=" * 80)

        # Check for listings API call
        listings_calls = [
            req for req in self.network_requests
            if "/hermes/api/v1/listings" in req["url"]
        ]

        if listings_calls:
            print(f"\n\nLISTINGS API CALLS ({len(listings_calls)} found):")
            for req in listings_calls:
                print(f"  [{req['method']}] {req['url']}")
                print(f"    Status: {req['status']}, Size: {req['size']:,} bytes")
                if req.get("body"):
                    print(f"    Response body: {req['body'][:500]}...")
                print()
        else:
            print("\n\n⚠️  NO /hermes/api/v1/listings API calls found!")
            print("    This page may not have ticket listings.")

        # Highlight other API calls
        api_requests = [
            req for req in self.network_requests
            if req["resource_type"] in ["xhr", "fetch"] and "/hermes/api/v1/listings" not in req["url"]
        ]

        if api_requests:
            print(f"\n\nOTHER API CALLS ({len(api_requests)} total):")
            for req in api_requests[:10]:  # Show first 10
                print(f"  [{req['method']}] {req['url']}")
                print(f"    Status: {req['status']}, Size: {req['size']:,} bytes")

    def print_next_data_analysis(self, analysis: dict):
        """Print __NEXT_DATA__ analysis."""
        print("\n" + "=" * 80)
        print("__NEXT_DATA__ ANALYSIS")
        print("=" * 80)

        if "error" in analysis:
            print(f"\nError: {analysis['error']}")
            return

        print(f"\nTop-level keys: {', '.join(analysis['top_level_keys'])}")
        print(f"\nPageProps keys ({len(analysis['page_props_keys'])}):")
        for key in analysis['page_props_keys']:
            print(f"  - {key}")

        if analysis["potential_listing_keys"]:
            print(f"\n\nPOTENTIAL LISTING KEYS ({len(analysis['potential_listing_keys'])}):")
            for key in analysis["potential_listing_keys"]:
                print(f"\n  {key}:")
                if key in analysis["sample_data"]:
                    sample = json.dumps(analysis["sample_data"][key], indent=4)
                    # Indent the sample
                    sample_lines = sample.split("\n")
                    for line in sample_lines:
                        print(f"    {line}")
        else:
            print("\n⚠️  No obvious listing-related keys found")
            print("    Check the full data structure for nested listing data")

    def save_full_data(self, output_path: Path):
        """Save full __NEXT_DATA__ to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.next_data:
            with open(output_path, "w") as f:
                json.dump(self.next_data, f, indent=2)
            print(f"\n✓ Full __NEXT_DATA__ saved to: {output_path}")
        else:
            print("\n⚠️  No __NEXT_DATA__ to save")

        # Save listings API response if found
        listings_calls = [
            req for req in self.network_requests
            if "/hermes/api/v1/listings" in req["url"] and req.get("body")
        ]

        if listings_calls:
            listings_path = output_path.parent / "vividseats_listings_api.json"
            try:
                listings_json = json.loads(listings_calls[0]["body"])
                with open(listings_path, "w") as f:
                    json.dump(listings_json, f, indent=2)
                print(f"✓ Listings API response saved to: {listings_path}")

                # Print structure summary
                print(f"\n📊 Listings API Structure:")
                print(f"  URL: {listings_calls[0]['url']}")
                if "global" in listings_json:
                    global_data = listings_json["global"][0] if listings_json["global"] else {}
                    print(f"  Listing count: {global_data.get('listingCount', 'N/A')}")
                    print(f"  Ticket count: {global_data.get('ticketCount', 'N/A')}")
                if "tickets" in listings_json:
                    print(f"  Tickets array: {len(listings_json['tickets'])} items")
                    if listings_json["tickets"]:
                        print(f"  First ticket keys: {list(listings_json['tickets'][0].keys())}")
            except Exception as e:
                print(f"⚠️  Error saving listings data: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Diagnose VividSeats page structure for listing data"
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="VividSeats event URL (or use --search to find events)",
    )
    parser.add_argument(
        "--search",
        "-s",
        help="Search for artist/performer and diagnose first event found",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/.cache/vividseats_diagnostic.json",
        help="Output file for full __NEXT_DATA__ (default: data/.cache/vividseats_diagnostic.json)",
    )

    args = parser.parse_args()

    if not args.url and not args.search:
        parser.error("Either provide a URL or use --search")

    # Run diagnostic
    diagnostic = VividSeatsDiagnostic()

    target_url = args.url
    if args.search:
        print(f"Searching for: {args.search}")
        target_url = await diagnostic.search_and_find_event(args.search)
        if not target_url:
            print("❌ No events found")
            return
        print(f"Found event URL: {target_url}\n")

    print("Starting diagnostic...")
    await diagnostic.diagnose_page(target_url)

    # Analyze results
    analysis = diagnostic.analyze_next_data()

    # Print reports
    diagnostic.print_network_summary()
    diagnostic.print_next_data_analysis(analysis)

    # Save full data
    diagnostic.save_full_data(Path(args.output))

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
