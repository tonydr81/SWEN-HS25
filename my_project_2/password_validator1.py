def is_password_valid(password):
    # Prüfe, ob das Passwort mindestens 8 Zeichen hat
    if len(password) < 8:
        return "Password must be at least 8 characters"
    else:
        return True