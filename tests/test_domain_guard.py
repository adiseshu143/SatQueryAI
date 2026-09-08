import io
import asyncio
import pytest
from PIL import Image
from backend.controller.domain_classifier import domain_classifier, DomainClassifier
from backend.controller.intent_router import router_engine
from backend.controller.agent import agent_controller
from backend.config import get_system_prompt

def create_test_image_bytes() -> bytes:
    img = Image.new('RGB', (100, 100), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def test_system_prompt_loaded():
    """Verify that verbatim system prompt is loaded from config file."""
    prompt = get_system_prompt()
    assert "SatQuery AI" in prompt
    assert "domain-specialist assistant for remote sensing" in prompt
    assert "Out-of-domain handling" in prompt

def test_scenario_1_indomain_image_based():
    """
    Scenario 1: In-Domain Image-Based Query.
    Query: 'How many buildings are in this scene?' with image.
    Expected: Processes Image Detection VQA workflow.
    """
    img_bytes = create_test_image_bytes()
    query = "How many buildings are in this scene?"

    # Check classifier directly
    res = domain_classifier.classify(query, has_image=True)
    assert res.is_in_domain is True
    assert res.category == "IN_DOMAIN_IMAGE_WORKFLOW"

    # Check controller execution
    response = asyncio.run(agent_controller.process_request(
        query=query,
        img1_bytes=img_bytes,
        img1_name="test_scene.png"
    ))
    assert response.intent != "OUT_OF_DOMAIN"
    assert response.answer is not None
    assert "building" in response.answer.lower() or "count" in response.answer.lower() or "structure" in response.answer.lower()

def test_scenario_2_indomain_general_rs_knowledge():
    """
    Scenario 2: In-Domain General Remote Sensing Knowledge Query (No Image).
    Query: 'What does SAR stand for and how is it complementary to optical satellite imagery?'
    Expected: Responds directly using Remote Sensing Specialist Domain Knowledge without needing an image.
    """
    query = "What does SAR stand for and how is it complementary to optical satellite imagery?"

    res = domain_classifier.classify(query, has_image=False)
    assert res.is_in_domain is True
    assert res.category == "IN_DOMAIN_GENERAL_KNOWLEDGE"
    assert "Synthetic Aperture Radar" in res.direct_answer

    response = asyncio.run(agent_controller.process_text_chat(query=query))
    assert response.intent == "GENERAL_REMOTE_SENSING_KNOWLEDGE"
    assert "Synthetic Aperture Radar" in response.answer
    assert "day and night" in response.answer.lower() or "optical" in response.answer.lower()

def test_scenario_3_borderline_weather_query():
    """
    Scenario 3: Borderline Query.
    Query: 'What's the weather today?'
    Expected: Intercepted and returned polite domain redirect message.
    """
    query = "What's the weather today?"

    res = domain_classifier.classify(query, has_image=False)
    assert res.is_in_domain is False
    assert res.category == "OUT_OF_DOMAIN"
    assert res.redirect_message == DomainClassifier.REDIRECT_MESSAGE

    response = asyncio.run(agent_controller.process_text_chat(query=query))
    assert response.intent == "OUT_OF_DOMAIN"
    assert response.answer == DomainClassifier.REDIRECT_MESSAGE

def test_scenario_4_out_of_domain_chitchat():
    """
    Scenario 4: Out-of-Domain General Chit-Chat Query.
    Query: 'Tell me a joke or write a song'
    Expected: Intercepted and returned polite domain redirect message.
    """
    query = "Tell me a joke or write a song"

    res = domain_classifier.classify(query, has_image=False)
    assert res.is_in_domain is False
    assert res.category == "OUT_OF_DOMAIN"
    assert res.redirect_message == DomainClassifier.REDIRECT_MESSAGE

    response = asyncio.run(agent_controller.process_text_chat(query=query))
    assert response.intent == "OUT_OF_DOMAIN"
    assert response.answer == DomainClassifier.REDIRECT_MESSAGE

def test_scenario_5_out_of_domain_unrelated_coding():
    """
    Scenario 5: Out-of-Domain Unrelated Coding Query.
    Query: 'Write a Python script to sort a list using quicksort'
    Expected: Intercepted and returned polite domain redirect message.
    """
    query = "Write a Python script to sort a list using quicksort"

    res = domain_classifier.classify(query, has_image=False)
    assert res.is_in_domain is False
    assert res.category == "OUT_OF_DOMAIN"
    assert res.redirect_message == DomainClassifier.REDIRECT_MESSAGE

    response = asyncio.run(agent_controller.process_text_chat(query=query))
    assert response.intent == "OUT_OF_DOMAIN"
    assert response.answer == DomainClassifier.REDIRECT_MESSAGE
