def meine_funktion(a, b, c=None):
    if isinstance(a, int) and isinstance(b, int) and isinstance(c, int):
        return 42
    if isinstance(a, str) and isinstance(b, str):
        return "hello"