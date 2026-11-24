from app.core.config import settings
from app.main import app, cors_allow_credentials


def test_wildcard_origin_does_not_allow_credentials():
    assert cors_allow_credentials(["*"]) is False
    assert cors_allow_credentials([]) is False


def test_explicit_origin_allows_credentials():
    assert cors_allow_credentials(["https://clinic.example"]) is True


def test_app_cors_follows_the_configured_origins():
    cors = next(layer for layer in app.user_middleware if layer.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_credentials"] is cors_allow_credentials(list(settings.ALLOWED_ORIGINS))
