from password_validator4 import is_password_valid


def test_password_with_no_capital_letter():
    result = is_password_valid("long12word")
    assert result == "Password must contain at least one capital letter"


def test_password_with_capital_letter():
    result = is_password_valid("Long12word")
    assert result is True


def test_password_with_multiple_errors():
    result = is_password_valid("somepw")
    assert result == (
        "Password must be at least 8 characters\n"
        "The password must contain at least 2 numbers\n"
        "Password must contain at least one capital letter"
    )