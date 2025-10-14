from greet6 import greet

def test_greet_with_mixed_names():
    result = greet(["Amy", "BRIAN", "Charlotte"])
    assert result == "Hello, Amy and Charlotte. AND HELLO BRIAN!"