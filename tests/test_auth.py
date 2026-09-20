from app.auth import APIKeyAuth


def test_valid_key_returns_tenant():
    auth = APIKeyAuth()
    assert auth.validate("test_key") == "test_tenant"
    assert auth.validate("demo_key") == "demo_tenant"


def test_invalid_key_returns_none():
    auth = APIKeyAuth()
    assert auth.validate("wrong_key") is None
    assert auth.validate(None) is None
    assert auth.validate("") is None


def test_is_valid():
    auth = APIKeyAuth()
    assert auth.is_valid("test_key") is True
    assert auth.is_valid("nope") is False