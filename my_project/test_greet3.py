from greet3 import greet

def test_greet_shouting():
    result = greet("JERRY")
    assert result == "HELLO JERRY!"