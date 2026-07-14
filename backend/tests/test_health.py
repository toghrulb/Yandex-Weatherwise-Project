from backend.app.main import app


def test_health_route_registered() -> None:
    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths
