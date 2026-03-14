import sys
import os

# Ensure we can import from current dir
sys.path.append(os.getcwd())

try:
    print("Importing engine...")
    from nova.engine import NeuralInteractionEngine
    print("Initializing NIE...")
    nie = NeuralInteractionEngine()
    print("NIE Initialized.")
    
    if nie.model:
        print("Model detected.")
        intent, conf, _ = nie.classify_intent("shutdown the computer")
        print(f"Test 'shutdown the computer': Intent={intent}, Conf={conf}")
        
        intent, conf, _ = nie.classify_intent("restart my pc")
        print(f"Test 'restart my pc': Intent={intent}, Conf={conf}")
    else:
        print("Model NOT detected (None).")

except Exception as e:
    import traceback
    traceback.print_exc()
