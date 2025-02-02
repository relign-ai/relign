import pytest

from relign.guidance.llms._mock import Mock

@pytest.fixture
def mock_guidance():
    return Mock()