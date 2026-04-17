def pytest_configure(config: object) -> None:
    config.addinivalue_line("markers", "slow: downloads DINOv2 weights / full Dinomaly forward")
