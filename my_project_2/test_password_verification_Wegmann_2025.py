from password_verification_Wegmann_2025 import *


def test_correct_password_is_accepted():
    assert is_password_valid("abD!23f2*") == True


def test_no_numbers_is_rejected():
    assert (
        is_password_valid("afs&asllak*!!Asd")
        == "Password must contain at least 2 numbers"
    )


def test_short_passwords_are_rejected():
    assert is_password_valid("aA34+") == "Password must be at least 8 characters"


def test_rejection_messages_are_combined():
    assert (
        is_password_valid("somePw+")
        == "Password must be at least 8 characters\nPassword must contain at least 2 numbers"
    )


def test_missing_capital_letter_is_rejected():
    assert (
        is_password_valid("abd!23f2*")
        == "Password must contain at least one capital letter"
    )


def test_missing_special_character_is_rejected():
    assert (
        is_password_valid("abde23f2A")
        == "Password must contain at least one special character"
    )
