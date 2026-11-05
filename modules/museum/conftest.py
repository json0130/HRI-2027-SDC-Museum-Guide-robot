import pytest

from modules import ephemeral


@pytest.fixture(autouse=True)
def _reset_ephemeral():
    ephemeral.reset()
    yield
    ephemeral.reset()      # a test that forgot end_visit must not leak "museum mode" into other tests
