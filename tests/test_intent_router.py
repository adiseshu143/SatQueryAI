from backend.controller.intent_router import router_engine

def test_single_image_intent():
    res = router_engine.parse_query("What type of land cover is visible?", has_image2=False)
    assert res.intent == "LAND_COVER"
    assert res.requires_image2 is False

def test_change_detection_intent():
    res = router_engine.parse_query("What changed between these images?", has_image2=True)
    assert res.intent == "CHANGE_DETECTION"
    assert res.requires_image2 is True

def test_deforestation_intent():
    res = router_engine.parse_query("Has forest cover decreased between dates?", has_image2=True)
    assert res.intent == "DEFORESTATION"
    assert "vegetation" in res.entities

def test_urban_growth_intent():
    res = router_engine.parse_query("Where has new building construction appeared?", has_image2=True)
    assert res.intent == "URBAN_GROWTH"
    assert "urban" in res.entities
