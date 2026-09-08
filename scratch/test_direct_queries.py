import sys
sys.stdout.reconfigure(encoding='utf-8')
from backend.controller.domain_classifier import domain_classifier

queries = [
    "What's the difference between SAR and optical satellite imagery?",
    "Explain how NDVI is calculated and what it indicates",
    "What does SAR stand for and how is it complementary to optical satellite imagery?",
    "Explain NDVI, NDWI, and MNDWI spectral indices",
    "What are the resolution trade-offs between spatial and temporal resolution?",
    "what is a satellite",
    "how does flood mapping work in remote sensing"
]

for q in queries:
    res = domain_classifier.classify(q)
    print("=" * 60)
    print(f"QUERY: {q}")
    print(f"CATEGORY: {res.category}")
    print(f"REASON: {res.reason}")
    print(f"ANSWER:\n{res.direct_answer or res.redirect_message}\n")
