def greet(name):
    # Wenn kein Name übergeben wurde (None)
    if name is None:
        return "Hello, my friend."
    
    # Wenn Name eine Liste ist
    if isinstance(name, list):
        # Beispiel: ["Jill", "Jane"]
        if len(name) == 2:
            return f"Hello, {name[0]} and {name[1]}."
    
    # Wenn Name in Grossbuchstaben ist → schreien
    if isinstance(name, str) and name.isupper():
        return f"HELLO {name}!"
    
    # Standardfall: einfacher Name
    return f"Hello, {name}."