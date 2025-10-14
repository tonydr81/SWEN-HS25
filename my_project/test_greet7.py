from greet7 import greet

def test_greet_with_comma_in_name():
    result = greet(["Bob", "Charlie, Dianne"])
    assert result == "Hello, Bob, Charlie, and Dianne."