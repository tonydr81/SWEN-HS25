def greet(name):
    if isinstance(name, list):
        # zwei leere Listen vorbereiten
        normal_names = []
        shouted_names = []

        # alle Namen durchgehen und trennen
        for n in name:
            if n.isupper():
                shouted_names.append(n)
            else:
                normal_names.append(n)

        # Begrüssung für normale Namen
        greeting = ""
        if normal_names:
            if len(normal_names) == 1:
                greeting = f"Hello, {normal_names[0]}."
            elif len(normal_names) == 2:
                greeting = f"Hello, {normal_names[0]} and {normal_names[1]}."
            else:
                greeting = f"Hello, {', '.join(normal_names[:-1])}, and {normal_names[-1]}."

        # Begrüssung für geschriene Namen
        if shouted_names:
            shouted_greeting = f" AND HELLO {', '.join(shouted_names)}!"
            greeting += shouted_greeting

        return greeting

    # Falls kein Name oder None
    if name is None:
        return "Hello, my friend."

    # Falls geschrien
    if isinstance(name, str) and name.isupper():
        return f"HELLO {name}!"

    # Normale Begrüssung
    return f"Hello, {name}."