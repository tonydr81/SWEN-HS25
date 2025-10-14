def is_password_valid(password):
    errors = []  # Liste für alle möglichen Fehlermeldungen

    # Regel 1: Passwortlänge prüfen
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")

    # Regel 2: Anzahl der Ziffern prüfen
    digit_count = 0
    for character in password:
        if character.isdigit():
            digit_count += 1
    if digit_count < 2:
        errors.append("The password must contain at least 2 numbers")

    # Fehler zurückgeben oder True
    if errors:
        return "\n".join(errors)
    else:
        return True