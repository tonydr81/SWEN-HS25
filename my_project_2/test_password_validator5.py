from password_validator5 import is_password_valid


def test_password_with_no_special_character():
    result = is_password_valid("Abc12word")
    assert result == "Password must contain at least one special character"


def test_password_with_special_character():
    result = is_password_valid("Abc12word!")
    assert result is True


def test_password_with_multiple_errors():
    result = is_password_valid("abc12")
    expected = (
        "Password must be at least 8 characters\n"
        "Password must contain at least one capital letter\n"
        "Password must contain at least one special character"
    )
    assert result == expected