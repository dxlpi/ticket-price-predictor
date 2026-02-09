# Plan: Fix VividSeats Scraper Listing Collection

## Context

### Original Request
Debug and fix the VividSeats scraper that is failing to capture ticket listings.

### Problem Summary
The scraper at `src/ticket_price_predictor/scrapers/vividseats.py` successfully finds events but `get_event_listings()` returns empty for most events. The scraper waits for API endpoint `/hermes/api/v1/listings?` but it is not being captured.

### Evidence from Logs
From `/Users/heather/ticket-price-predictor/logs/monitor.log`:

1. **Feb 2, 2026 run**: 15/15 events returned "No listings found" with network errors
2. **Feb 3, 2026 run**: Partial success - collected 18, 8, 1, 1, 10, 16 listings (variable results)
3. **Feb 9, 2026 runs**: All failing with `ERR_INTERNET_DISCONNECTED` - EC2 network issue

The inconsistent collection (some events get 96-100 listings, others get 0-1) suggests the issue is NOT purely network-related when the scraper can connect.

### Root Cause Hypotheses (Ranked by Likelihood)

| # | Hypothesis | Evidence | Likelihood |
|---|------------|----------|------------|
| 1 | **VividSeats changed API endpoint** | Scraper looks for `/hermes/api/v1/listings?` but site may use different path now | HIGH |
| 2 | **Listings embedded in `__NEXT_DATA__`** | Event search already uses `__NEXT_DATA__` successfully; listings may be there too | HIGH |
| 3 | **Lazy-load requires scroll/interaction** | 8-second wait may not be enough; listings may load on scroll | MEDIUM |
| 4 | **Anti-bot blocking responses** | Using playwright-stealth but may be detected | MEDIUM |
| 5 | **Race condition in response capture** | Listener registered after navigation starts | LOW |

---

## Work Objectives

### Core Objective
Restore reliable ticket listing collection from VividSeats event pages, achieving 80%+ success rate on events that have available tickets.

### Deliverables
1. Updated `get_event_listings()` method with working data extraction
2. Diagnostic script to test scraper behavior interactively
3. Updated monitoring script with better error reporting
4. Deployment instructions for EC2 update

### Definition of Done
- [ ] Scraper collects listings from at least 12/15 events (80%+)
- [ ] Average listings per successful event >= 50 (not 1-2)
- [ ] Tests pass for scraper module
- [ ] EC2 instance updated and collecting data

---

## Guardrails

### Must Have
- Maintain backward compatibility with existing `ScrapedListing` schema
- Keep polite scraping delays (3+ seconds between requests)
- Use stealth mode to avoid detection
- Handle edge cases gracefully (sold-out events, restricted events)

### Must NOT Have
- Aggressive scraping that could get IP banned
- Breaking changes to `ListingCollector` interface
- Hardcoded test data or mock responses in production
- Removal of the existing API capture approach (keep as fallback)

---

## Task Flow

```
[TASK 1: Investigate] ──────────────────────────────────────────────────┐
   Create diagnostic script to capture all network traffic              │
   and inspect page HTML for listing data sources                       │
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
[TASK 2: Identify Data Source] ─────────────────────────────────────────┐
   Run diagnostic on multiple events to determine:                      │
   - Current API endpoint pattern (if any)                              │
   - Whether listings are in __NEXT_DATA__                              │
   - Whether interaction triggers data load                             │
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
[TASK 3: Implement Fix] ────────────────────────────────────────────────┐
   Update get_event_listings() based on findings:                       │
   - Add __NEXT_DATA__ extraction as primary method                     │
   - Update API endpoint pattern if changed                             │
   - Add scroll/interaction if needed                                   │
   - Keep API capture as fallback                                       │
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
[TASK 4: Add Resilience] ───────────────────────────────────────────────┐
   - Add retry logic for transient failures                             │
   - Add better logging/diagnostics                                     │
   - Add fallback extraction methods                                    │
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
[TASK 5: Test Locally] ─────────────────────────────────────────────────┐
   - Run scraper against 5+ diverse events                              │
   - Verify listing count and data quality                              │
   - Run existing tests                                                 │
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
[TASK 6: Deploy to EC2] ────────────────────────────────────────────────┐
   - Push changes to repo                                               │
   - SSH to EC2, pull changes                                           │
   - Run monitor script, verify collection                              │
   - Monitor logs for 1-2 collection cycles                             │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Detailed TODOs

### TASK 1: Create Diagnostic Script
**File**: `scripts/diagnose_vividseats.py`
**Purpose**: Capture and analyze what data is available on VividSeats event pages

**Acceptance Criteria**:
- Script launches browser and navigates to a VividSeats event URL
- Captures ALL network responses (not just `/hermes/api/v1/listings?`)
- Extracts and saves `__NEXT_DATA__` JSON to file for inspection
- Logs all XHR/Fetch requests with URLs and response sizes
- Saves page HTML for offline analysis

**Implementation Notes**:
```python
# Key capabilities needed:
# 1. Broader response capture (any JSON response)
# 2. __NEXT_DATA__ extraction
# 3. Network request logging
# 4. Optional: screenshot capture
```

---

### TASK 2: Analyze Data Sources
**Purpose**: Determine where VividSeats now stores listing data

**Steps**:
1. Run diagnostic script on 3 known events with tickets:
   - Bruno Mars (high volume)
   - A smaller event
   - An event that previously failed
2. Inspect captured network traffic for listing-related endpoints
3. Search `__NEXT_DATA__` for listing structures
4. Document findings

**Acceptance Criteria**:
- Document identifies primary data source for listings
- Document includes example data structure
- Document notes any required interactions (scroll, click)

---

### TASK 3: Update Scraper Implementation
**File**: `src/ticket_price_predictor/scrapers/vividseats.py`
**Method**: `get_event_listings()` (lines 192-235)

**Current Implementation Issues**:
```python
# Current code looks for specific endpoint:
if "/hermes/api/v1/listings?" in response.url and "productionId" in response.url:
```

**Proposed Changes** (based on likely findings):

#### Option A: If listings are in `__NEXT_DATA__`
```python
async def get_event_listings(self, event_url: str, max_listings: int = 500):
    # Navigate to page
    await self._page.goto(event_url, wait_until="networkidle")
    await asyncio.sleep(3)

    # Extract __NEXT_DATA__ (same pattern as search_events)
    html = await self._page.content()
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)
    if not match:
        return []

    data = json.loads(match.group(1))
    page_props = data.get("props", {}).get("pageProps", {})

    # Extract listings from page props
    return self._extract_listings_from_next_data(page_props, max_listings)
```

#### Option B: If API endpoint changed
```python
# Update pattern to match new endpoint structure
# e.g., "/api/v2/listings" or "/graphql" with listings query
```

#### Option C: If interaction required
```python
# Scroll page to trigger lazy loading
await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
await asyncio.sleep(2)
# Then capture response or extract from DOM
```

**Acceptance Criteria**:
- Method returns listings for events that have tickets
- Preserves existing `ScrapedListing` schema compatibility
- Includes fallback to API capture for resilience
- Has proper error handling and logging

---

### TASK 4: Add Resilience and Logging
**Files**:
- `src/ticket_price_predictor/scrapers/vividseats.py`
- `src/ticket_price_predictor/ingestion/listings.py`

**Changes**:

1. **Add debug logging to scraper**:
```python
import logging
logger = logging.getLogger(__name__)

# In get_event_listings:
logger.debug(f"Navigating to {event_url}")
logger.debug(f"Found {len(listings)} listings via {method}")
```

2. **Add retry logic in ListingCollector**:
```python
# Retry failed events once with longer delay
if result.listings_saved == 0 and not result.errors:
    logger.info(f"Retrying {event_name} with extended wait")
    # Retry with longer wait
```

3. **Add extraction method indicator**:
```python
# Track which method succeeded for debugging
listings, method = await self._extract_listings_multi_method(...)
logger.info(f"Extracted {len(listings)} listings via {method}")
```

**Acceptance Criteria**:
- Logs clearly indicate extraction method used
- Failed extractions include diagnostic info
- Retry logic handles transient failures

---

### TASK 5: Local Testing
**Purpose**: Verify fix works before deployment

**Test Plan**:

1. **Unit test for extraction**:
```bash
pytest tests/test_vividseats_scraper.py -v
```

2. **Integration test with real site**:
```bash
# Test single event
python -c "
import asyncio
from ticket_price_predictor.scrapers import VividSeatsScraper

async def test():
    async with VividSeatsScraper(headless=False) as scraper:
        listings = await scraper.get_event_listings(
            'https://www.vividseats.com/bruno-mars-tickets-park-mgm-park-theater-3-29-2026--concerts-pop/production/5477319'
        )
        print(f'Found {len(listings)} listings')
        for l in listings[:5]:
            print(f'  {l.section} Row {l.row}: ${l.price_per_ticket}')

asyncio.run(test())
"
```

3. **Full collection test**:
```bash
# Run monitor with limited events
python scripts/monitor_popular.py 2>&1 | head -100
```

**Acceptance Criteria**:
- Single event test returns 50+ listings
- 4/5 test events return listings
- No new errors in output

---

### TASK 6: EC2 Deployment
**Purpose**: Deploy fix and verify production collection

**Steps**:

1. **Commit and push changes**:
```bash
git add src/ticket_price_predictor/scrapers/vividseats.py
git add scripts/diagnose_vividseats.py  # if created
git commit -m "fix(scraper): Update VividSeats listing extraction method"
git push origin main
```

2. **Deploy to EC2**:
```bash
# SSH to EC2 (update with actual connection details)
ssh -i ~/.ssh/ticket-predictor.pem ec2-user@<ec2-ip>

# Pull changes
cd /path/to/ticket-price-predictor
git pull origin main

# Install any new dependencies
pip install -r requirements.txt

# Test scraper
python -c "from ticket_price_predictor.scrapers import VividSeatsScraper; print('OK')"
```

3. **Run collection and monitor**:
```bash
# Manual test run
python scripts/monitor_popular.py

# Check logs
tail -f logs/monitor.log
```

4. **Verify success**:
- Check that events_succeeded > 12 (80%+)
- Check that total_listings > 500
- Monitor next scheduled run

**Acceptance Criteria**:
- EC2 deployment successful (no import errors)
- First manual run collects 80%+ of events
- Cron job runs successfully

---

## Commit Strategy

| Commit | Description | Files |
|--------|-------------|-------|
| 1 | Add diagnostic script for VividSeats debugging | `scripts/diagnose_vividseats.py` |
| 2 | Fix VividSeats listing extraction with new method | `src/ticket_price_predictor/scrapers/vividseats.py` |
| 3 | Add resilience and logging to scraper | `vividseats.py`, `listings.py` |
| 4 | Add/update tests for scraper changes | `tests/test_vividseats_scraper.py` |

---

## Success Criteria

### Quantitative
- **Collection rate**: 80%+ of events with tickets return listings
- **Listing volume**: Average 50+ listings per successful event (not 1-2)
- **Error rate**: <10% of attempts result in errors
- **Test coverage**: All scraper tests pass

### Qualitative
- Clear logging shows which extraction method succeeded
- Graceful degradation when primary method fails
- No IP bans or rate limiting from VividSeats

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| VividSeats blocks scraper | Keep delays, use stealth, rotate user agents |
| __NEXT_DATA__ structure changes | Add multiple extraction paths, log structure for debugging |
| EC2 network issues recur | Add network health check before scraping, alert on failures |
| API endpoint changes again | Monitor for 0-listing events, add alerting |

---

## Notes

### Current EC2 Status
Recent logs show `ERR_INTERNET_DISCONNECTED` errors on EC2 since Feb 9. This is a separate infrastructure issue that should be investigated:
- Check EC2 security groups
- Verify outbound internet access
- Check if VividSeats is blocking EC2 IP range

### Alternative Data Sources
If VividSeats becomes unreliable, consider:
- StubHub scraper (already implemented but may be blocked)
- SeatGeek API (requires API key)
- Direct venue APIs

---

## Appendix: Key File Locations

| Purpose | Path |
|---------|------|
| VividSeats scraper | `src/ticket_price_predictor/scrapers/vividseats.py` |
| StubHub scraper (reference) | `src/ticket_price_predictor/scrapers/stubhub.py` |
| Listing collector | `src/ticket_price_predictor/ingestion/listings.py` |
| Monitor script | `scripts/monitor_popular.py` |
| Monitor logs | `logs/monitor.log` |
| Scraper tests | `tests/test_vividseats_scraper.py` (create if missing) |

---

*Plan created: 2026-02-09*
*Plan author: Prometheus (planner agent)*
