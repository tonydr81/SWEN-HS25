from greet5 import greet

def test_greet_with_multiple_names():
    result = greet(["Amy", "Brian", "Charlotte"])
    assert result == "Hello, Amy, Brian, and Charlotte."