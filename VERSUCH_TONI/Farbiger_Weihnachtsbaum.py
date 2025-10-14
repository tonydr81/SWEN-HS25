# Aufgabenstellung (Teil 4 - Farbiger Weihnachtsbaum)

import random
from rich.console import Console

console = Console()                                         # Erzeuge ein Console-Objekt aus rich, um farbigen Text in der Konsole auszugeben

def zeichne_schicht(zeile, hoehe):                          # Berechnet die Anzahl Sterne und Leerzeichen für eine Zeile der Schicht (oberer Teil des Baumes)
    sterne = 2 * zeile - 1
    leerzeichen = hoehe - zeile
    zeile_inhalt = ["*"] * sterne                           # Erstelle eine Liste mit Sternen
    
    if sterne > 1:                                          # wähle zufällig eine Stelle aus
        position = random.randint(0, sterne - 1)
        zeile_inhalt[position] = "O"
    
    farbige_zeile = "".join(                                 # farbige Ausgabe Grün und hell Rot
        f"[green]{z}[/green]" if z == "*" else f"[bright_red]{z}[/bright_red]"
        for z in zeile_inhalt
    )
    
    console.print(" " * leerzeichen + farbige_zeile)

def zeichne_stamm(hoehe, stammhoehe):                       # Stammhöhe veränderbar
    leerzeichen = hoehe - 1
    for _ in range(stammhoehe):
        console.print(" " * leerzeichen + "[red]x[/red]")   # Stamm in rot

def zeichne_weihnachts_baum(hoehe, stammhoehe):             # Zeichne den Weihnachtsbaum
    for zeile in range(1, hoehe + 1):
        zeichne_schicht(zeile, hoehe)
    
    zeichne_stamm(hoehe, stammhoehe)                        # Zeichne den Stamm

zeichne_weihnachts_baum(10, 3)                               # Aufruf Höhe und Stammhöhe
