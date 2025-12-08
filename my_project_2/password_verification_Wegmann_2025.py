# Beispiellösung Dozent "Passwortverifikation" (Wegmann, 2025)

# Aus https://tddmanifesto.com/exercises/
# Create a function that can be used as a validator for the password field of
# a user registration form. The validation function takes a string as an input
# and returns a validation result. The validation result should contain a
# boolean indicating if the password is valid or not, and also a field with
# the possible validation errors.

# Requirements
# 1. The password must be at least 8 characters long. If it is not met, then
#    the following error message should be returned: "Password must be at
#    least 8 characters"
# 2. The password must contain at least 2 numbers. If it is not met, then the
#    following error message should be returned: "The password must contain at
#    least 2 numbers"
# 3. The validation function should handle multiple validation errors.
#    For example, "somepw" should an error message: "Password must be at least
#    8 characters\nThe password must contain at least 2 numbers"
# 4. The password must contain at least one capital letter. If it is not met,
#    then the following error message should be returned: "password must
#    contain at least one capital letter"
# 5. The password must contain at least one special character. If it is not
#    met, then the following error message should be returned: "password must
#    contain at least one special character"


def count_digits(string):
    count = 0
    for character in string:
        if character.isdigit():
            count += 1
    return count


def count_capital_letters(string):
    count = 0
    for character in string:
        if character.isupper():
            count += 1
    return count


def count_special_characters(string):
    count = 0
    for character in string:
        if character in '+"*%&/()?@#!${}-;:_,.-':
            count += 1
    return count


def is_password_valid(password):
    msgs = []
    if len(password) < 8:
        msgs.append("Password must be at least 8 characters")
    if count_digits(password) < 2:
        msgs.append("Password must contain at least 2 numbers")
    if count_capital_letters(password) < 1:
        msgs.append("Password must contain at least one capital letter")
    if count_special_characters(password) < 1:
        msgs.append("Password must contain at least one special character")
    if len(msgs) > 0:
        return "\n".join(msgs)
    return True
