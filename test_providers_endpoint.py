"""Quick sanity test for /api/providers logic without HTTP layer."""
import asyncio
from app.main import get_providers

def test_get_providers_fn():
    data = asyncio.run(get_providers())
    assert "providers" in data, "Missing 'providers' key"
    assert isinstance(data["providers"], dict), "Providers should be a dict"
    assert len(data["providers"]) >= 1, "Expected at least one provider configured"

if __name__ == "__main__":
    test_get_providers_fn()
    print("get_providers() OK")