def is_password_valid(password):
    errors = []

    # Regel 1: Mindestlänge 8 Zeichen
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")

    # Regel 2: mindestens 2 Ziffern
    digit_count = 0
    for character in password:
        if character.isdigit():
            digit_count += 1
    if digit_count < 2:
        errors.append("The password must contain at least 2 numbers")

    # Regel 3: mindestens 1 Grossbuchstabe
    has_capital = False
    for character in password:
        if character.isupper():
            has_capital = True
            break
    if not has_capital:
        errors.append("Password must contain at least one capital letter")

    # Rückgabe: entweder True oder alle Fehler mit Zeilenumbruch
    if errors:
        return "\n".join(errors)
    return True