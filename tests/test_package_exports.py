def test_public_import_from_package() -> None:
    from cuvis_ai_dinomaly import DinomalyDetector, DinomalyTrainLossBridge

    assert DinomalyDetector.__name__ == "DinomalyDetector"
    assert DinomalyTrainLossBridge.__name__ == "DinomalyTrainLossBridge"
