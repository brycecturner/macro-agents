from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_status_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json() == {"status": "ok"}
