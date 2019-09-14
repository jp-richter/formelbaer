# Tex2PNG

Extraction of Mathematical equations out of tex files/archives and conversion into PNGs. 

## Aus der Console aufrufen

### Eigene Dateien
Das Modul ```tex2png.py``` muss zwingend mit folgendem Argument geöffnet werden:
1. Pfad(absolut/relativ) zur Tex-Datei oder Ordner
2. Options

#### Options
##### ```-ts <name>```
Gib einen Subordner von Target an, ansonsten wird ```default``` Wert verwendet.
##### ```-io <name>```
Gib einen Subordner für die Bilder im Target an, ansonsten wird ```defaultimage``` gewählt.
##### ```-d```
Gibt an, ob die Targetsubordner direkt nach dem Kopieren gelöscht werden sollen.

#### Wichtig
Das Script sollte nur aus dem Ordner tex2png heraus aufgerufen werden, da sonst das Template nicht gefunden werden kann.
Der Targetordner wird diesem Ordner dann erstellt.

#### Beispiele
Die folgenden Befehle könnt ihr zum Test ausführen, da der Ordner und die Tex-Dokumente im Projekt existieren.

```
python3 tex2png.py sample
```

```
python3 tex2png.py sample/sample.tex
```

```
python3 tex2png.py sample/sample.tex -io imageoutput -ts testdatei
```


#### Zusätzliche Infos

Jeglicher Output befindet sich dann in diesem Ordner, dazu zählen:
1. template-with-macros.tex: Template, welches für alle Formeln des Dokuments bestimmt ist
2. 0xx.pdf: PDF Dokument jeder Formel numeriert nach ihrem erscheinen im Dokument
3. 0xx.png: PNG der Formel
4. pdflatex.log/0xx.log: Das sind von pdflatex erstellte log Dateien
5. 0xx.aux: Von pdflatex erstellte Datei
6. ghostscript.log: Log von Ghostscript

Relevant für den Endnutzer ist hierbei nur 3 relevant


### Archiv
Das Modul ```tex2png.py``` mit Argumenten öffnen:
1. ```archive``` Schlüsselwort
2. Options

#### Options
##### ```-p```
Gibt an, ob prints ausgegeben werden sollen.

##### ```-d```
Gibt an, ob die bestehenden Imagefiles gelöscht werden sollen.
