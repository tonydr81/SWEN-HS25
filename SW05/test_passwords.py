from passwords import check_password_length

def test_password_too_short():
    assert check_password_length("abc") == False

def test_password_long_enough():
    assert check_password_length("abcdefgh") == True