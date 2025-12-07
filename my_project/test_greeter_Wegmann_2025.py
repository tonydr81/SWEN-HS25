from greeter_Wegmann_2025 import *


# Requirement 1
def test_wenn_Bob_übergeben_wird_grüsst_die_Funktion_Bob():
    assert greet("Bob") == "Hello, Bob."


def test_wenn_Anna_übergeben_wird_grüsst_die_Funktion_Anna():
    assert greet("Anna") == "Hello, Anna."


# Requirement 2
def test_wird_ein_None_übergeben_grüsst_die_Funktion_generisch():
    assert greet(None) == "Hello, my friend."


# Requirement 3
def test_bei_einem_namen_mit_grossbuchstaben_grüsst_die_funktion_mit_schreien():
    assert greet("BOB") == "HELLO, BOB!"


# Requirement 4
def test_liste_mit_zwei_namen_wird_mit_and_getrennt_ausgegeben():
    assert greet(["Jill", "Jane"]) == "Hello, Jill and Jane."


# Requirement 5
def test_liste_mit_mehreren_namen_wird_mit_kommas_und_einem_and_am_ende_ausgegeben():
    assert greet(["Amy", "Brian", "Charlotte"]) == "Hello, Amy, Brian, and Charlotte."


# Requirement 6
def test_trenne_namens_liste_in_schreien_und_nicht_schreien_auf():
    assert (
        greet(["Amy", "BRIAN", "Charlotte"])
        == "Hello, Amy and Charlotte. AND HELLO BRIAN!"
    )


def test_komplett_durchmischte_namen_werden_auch_korrekt_begrüsst():
    assert (
        greet(["Amy", "BRIAN", "Charlotte", "ADRIAN"])
        == "Hello, Amy and Charlotte. AND HELLO BRIAN AND ADRIAN!"
    )


def test_mit_je_drei_namen_wird_auch_korrekt_begrüsst():
    assert (
        greet(["Amy", "BRIAN", "Charlotte", "ADRIAN", "John", "JOAN"])
        == "Hello, Amy, Charlotte, and John. AND HELLO BRIAN, ADRIAN, AND JOAN!"
    )


# Requirement 7
def test_kommas_in_namen_werden_auch_als_listen_behandelt():
    assert greet(["Bob", "Charlie, Dianne"]) == "Hello, Bob, Charlie, and Dianne."


# Requirement 8
def test_escaped_kommas_in_namen_werden_nicht_aufgetrennt():
    assert greet(["Bob", '"Charlie, Dianne"']) == "Hello, Bob and Charlie, Dianne."
