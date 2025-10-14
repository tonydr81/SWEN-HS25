from password_validator1 import is_password_valid

def test_password_too_short():
    result = is_password_valid("abc123")
    assert result == "Password must be at least 8 characters"

def test_password_long_enough():
    result = is_password_valid("abc12345")
    assert result == True