def greet(names):
    if names is None:
        return "Hello, my friend."

    # Falls nur ein einzelner Name übergeben wurde
    if isinstance(names, str):
        names = [names]

    # Liste für normale Namen (inkl. Quote-Check)
    processed_names = []
    for name in names:
        # Wenn der Name in Anführungszeichen steht → als Ganzes behalten
        if name.startswith('"') and name.endswith('"'):
            processed_names.append(name.strip('"'))
        # Wenn Komma enthalten ist und keine Quotes → splitten
        elif ',' in name:
            split_names = [n.strip() for n in name.split(',')]
            processed_names.extend(split_names)
        else:
            processed_names.append(name)

    # Jetzt Formatierung der Begrüßung
    if len(processed_names) == 1:
        return f"Hello, {processed_names[0]}."
    elif len(processed_names) == 2:
        return f"Hello, {processed_names[0]} and {processed_names[1]}."
    else:
        all_but_last = ", ".join(processed_names[:-1])
        return f"Hello, {all_but_last}, and {processed_names[-1]}."