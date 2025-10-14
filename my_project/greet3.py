def greet(name):
    if name is None:
        return "Hello, my friend."
    elif name.isupper():
        return f"HELLO {name}!"
    else:
        return f"Hello, {name}."