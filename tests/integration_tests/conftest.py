"""Re-export fixtures from tests/unit_tests/conftest.py.

pytest fixtures are scoped to a conftest.py's own directory and its
children; tests/unit_tests/conftest.py's `appointment_nodes` fixture is
therefore invisible to tests under the sibling tests/integration_tests/
directory unless re-exported here.
"""

from unit_tests.conftest import appointment_nodes  # noqa: F401
