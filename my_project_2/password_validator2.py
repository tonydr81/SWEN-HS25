def is_password_valid(password):
    # Zählt, wie viele Ziffern (0–9) im Passwort vorkommen
    digit_count = 0
    for character in password:
        if character.isdigit():
            digit_count += 1

    # Wenn weniger als 2 Ziffern vorkommen, gib Fehlermeldung zurück
    if digit_count < 2:
        return "The password must contain at least 2 numbers"
    else:
        return True