# Conventions

**Commits**: Conventional commits — `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
**Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
**Tests**: pytest with pytest-asyncio (`asyncio_mode = "auto"`), files as `tests/test_<module>.py`
**Formatting**: ruff (rules: E, W, F, I, B, C4, UP, ARG, SIM) + mypy strict with pydantic plugin
