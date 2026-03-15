import os
import sys
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

# Expects GROQ_API_KEY to be set in environment
if not os.getenv("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY not set in environment")

def test_aider():
    main_model = Model("groq/llama-3.3-70b-versatile")
    my_io = InputOutput(yes=True, pretty=False)
    # Use a dummy file to see if it tracks it
    with open("test_aider_file.txt", "w") as f:
        f.write("hello")
        
    coder = Coder.create(main_model=main_model, io=my_io, fnames=["test_aider_file.txt"])
    print(f"Coder attributes: {dir(coder)}")
    
    # Try common attribute names
    for attr in ["fnames", "abs_fnames", "cur_messages", "repo"]:
        if hasattr(coder, attr):
            print(f"Attribute '{attr}': {getattr(coder, attr)}")

    
    os.remove("test_aider_file.txt")

if __name__ == "__main__":
    test_aider()
