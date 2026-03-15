from nova.engine import NeuralInteractionEngine

nie = NeuralInteractionEngine()
test_queries = [
    "run this code",
    "solve a math problem using python",
    "plot a sine wave",
    "execute interpreter",
    "what is my name", # Should NOT be RUN_CODE
]

print("NIE Intent Classification Test:")
print("-" * 30)
for q in test_queries:
    intent, conf, _ = nie.classify_intent(q)
    print(f"Query: '{q}' -> Intent: {intent} (Conf: {conf:.2f})")
