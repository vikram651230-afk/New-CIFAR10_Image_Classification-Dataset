import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.config import Config

class TestingConfig(Config):
    TESTING = True
    DEBUG = False

@pytest.fixture
def app():
    """Create and configure a Flask application for testing."""
    application = create_app(TestingConfig)
    return application

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()
