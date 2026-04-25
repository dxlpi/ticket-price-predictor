# VividSeats Hermes API Architecture
Promoted: 2026-03-23 | Updated: 2026-03-23

## Rule
VividSeats event discovery uses `/hermes/api/v1/productions` (public JSON API, no auth). The `__NEXT_DATA__` / `initialProductionListData` pattern does NOT exist on any VS page — code checking for it is dead. Listing collection uses Playwright to intercept `/hermes/api/v*/listings` responses.

## Why
Without this knowledge, you might write scraper code checking `__NEXT_DATA__` that silently never fires, or try to avoid Playwright for listing collection when it's required.

## Pattern
```python
# Discovery — direct API call, no browser needed
GET /hermes/api/v1/productions?categoryId=2&rows=48&page=1
# Returns: performers[master=true].name, venue.capacity, rich event data

# Listing collection — Playwright response interception required
# intercepts: /hermes/api/v*/listings
# (client-side API, needs browser rendering to trigger)
```

Key fields in productions API:
- `performers[master=true].name` — clean artist/team name
- `venue.capacity` — venue size
- `categoryId` — 2=concerts (others unknown, need probing)
