# Testing Pydantic Models

## Rule
Use `model.model_copy(update={...})` to create test variants of Pydantic models, not direct field mutation or re-construction. For type annotations in test helpers that use a local import, drop the type annotation entirely rather than using a string forward reference — ruff F821 will flag the string annotation as undefined.

## Why
Direct field assignment on Pydantic v2 models works when not frozen, but `model_copy` is the idiomatic pattern. String annotations like `-> "TicketListing"` in a method where `TicketListing` is only imported inside the function body cause ruff F821 (undefined name in annotation), even though they'd work at runtime.

## Pattern

**Correct — model_copy for test variants**:
```python
def test_zero_price_invalid(self, validator, valid_listing):
    listing = valid_listing.model_copy(update={"listing_price": 0.0})
    result = validator.validate_listing(listing)
    assert not result.is_valid
```

**Correct — drop annotation when using local import**:
```python
def _make_listing(self, price: float):  # no return type annotation
    from ticket_price_predictor.schemas import TicketListing
    return TicketListing(...)
```

**Wrong — string annotation with local import**:
```python
def _make_listing(self, price: float) -> "TicketListing":  # F821: ruff error
    from ticket_price_predictor.schemas import TicketListing
    return TicketListing(...)
```
