from password_validator2 import is_password_valid


def test_password_with_less_than_two_numbers():
    result = is_password_valid("abc1xyz")
    assert result == "The password must contain at least 2 numbers"


def test_password_with_two_or_more_numbers():
    result = is_password_valid("pass12word")
    assert result is True