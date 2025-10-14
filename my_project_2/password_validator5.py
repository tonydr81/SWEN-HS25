def is_password_valid(password):
    # Liste der erlaubten Sonderzeichen
    special_characters = "#$!"

    # Liste zum Sammeln aller Fehlermeldungen
    errors = []

    # Bedingung 1: Länge prüfen
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")

    # Bedingung 2: Mindestens 2 Ziffern prüfen
    digit_count = 0
    for char in password:
        if char.isdigit():
            digit_count += 1
    if digit_count < 2:
        errors.append("The password must contain at least 2 numbers")

    # Bedingung 3: Mindestens ein Grossbuchstabe prüfen
    has_capital = any(char.isupper() for char in password)
    if not has_capital:
        errors.append("Password must contain at least one capital letter")

    # Bedingung 4 (neu): Mindestens ein Sonderzeichen prüfen
    has_special = any(char in special_characters for char in password)
    if not has_special:
        errors.append("Password must contain at least one special character")

    # Rückgabe aller Fehler oder True
    if errors:
        return "\n".join(errors)
    else:
        return True