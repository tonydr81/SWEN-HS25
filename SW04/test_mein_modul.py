from mein_modul import meine_funktion

def test_meine_funktion_macht_abc():
    resultat = meine_funktion(1, 2, 3)
    assert resultat == 42

def test_meine_funktion_macht_xyz():
    resultat = meine_funktion("a", "b")
    assert resultat == "hello"