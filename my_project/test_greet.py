from greet import greet

def test_greet_with_name():
    result = greet("Bob")
    assert result == "Hello, Bob."