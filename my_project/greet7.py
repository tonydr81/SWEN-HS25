def greet(name):
    # Wenn der Name None ist
    if name is None:
        return "Hello, my friend."

    # Wenn es kein String und keine Liste ist, ignorieren
    if not isinstance(name, (str, list)):
        return "Hello, my friend."

    # Wenn es ein einzelner String ist
    if isinstance(name, str):
        if name.isupper():
            return f"HELLO {name}!"
        return f"Hello, {name}."

    # Jetzt wissen wir: name ist eine Liste
    all_names = []

    # Durch alle Einträge gehen
    for n in name:
        if "," in n:
            # Aufteilen am Komma und Leerzeichen entfernen
            parts = [p.strip() for p in n.split(",")]
            all_names.extend(parts)
        else:
            all_names.append(n)

    # Aufteilen in normale und geschriene Namen
    normal_names = [n for n in all_names if not n.isupper()]
    shout_names = [n for n in all_names if n.isupper()]

    # Normale Namen in einen schönen String umwandeln
    greeting = ""
    if normal_names:
        if len(normal_names) == 1:
            greeting = f"Hello, {normal_names[0]}."
        elif len(normal_names) == 2:
            greeting = f"Hello, {normal_names[0]} and {normal_names[1]}."
        else:
            greeting = f"Hello, {', '.join(normal_names[:-1])}, and {normal_names[-1]}."

    # Geschriene Namen extra anhängen
    if shout_names:
        if greeting:
            greeting += " "
        greeting += f"AND HELLO {', '.join(shout_names)}!"

    return greeting