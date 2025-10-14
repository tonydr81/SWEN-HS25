def greet(name):
    # Wenn keine Namen angegeben sind
    if name is None:
        return "Hello, my friend."

    # Wenn "name" ein einzelner String ist
    if isinstance(name, str):
        if name.isupper():
            return f"HELLO {name}!"
        return f"Hello, {name}."

    # Wenn "name" eine Liste ist
    if isinstance(name, list):
        # Wenn Liste nur 2 Namen enthält
        if len(name) == 2:
            return f"Hello, {name[0]} and {name[1]}."
        # Wenn Liste mehr als 2 Namen enthält
        else:
            # Alle Namen bis auf den letzten mit Komma trennen
            alle_bis_letzter = ", ".join(name[:-1])
            letzter = name[-1]
            return f"Hello, {alle_bis_letzter}, and {letzter}."

    # Fallback (sollte nie erreicht werden)
    return "Hello!"