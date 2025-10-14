from greet2 import greet

def test_greet_with_name():
    result = greet("Bob")
    assert result == "Hello, Bob."

def test_greet_with_none():
    result = greet(None)
    assert result == "Hello, my friend."