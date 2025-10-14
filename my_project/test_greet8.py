from greet8 import greet

def test_greet_with_escaped_commas():
    names = ["Bob", '"Charlie, Dianne"']
    result = greet(names)
    assert result == "Hello, Bob and Charlie, Dianne."