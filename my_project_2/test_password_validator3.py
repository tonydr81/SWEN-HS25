from password_validator3 import is_password_valid


def test_password_with_multiple_errors():
    result = is_password_valid("somepw")
    assert result == "Password must be at least 8 characters\nThe password must contain at least 2 numbers"


def test_password_with_only_length_error():
    result = is_password_valid("abc12")
    assert result == "Password must be at least 8 characters"


def test_password_with_only_number_error():
    result = is_password_valid("longword")
    assert result == "The password must contain at least 2 numbers"


def test_password_valid():
    result = is_password_valid("long12word")
    assert result is True