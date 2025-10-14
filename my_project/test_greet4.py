from greet4 import greet

def test_greet_with_two_names():
    result = greet(["Jill", "Jane"])
    assert result == "Hello, Jill and Jane."