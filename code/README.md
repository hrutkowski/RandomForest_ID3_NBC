# Las z NBC


## Opis projektu:
Las losowy z naiwnym klasyfikatorem bayesowskim (NBC) w zadaniu klasyfikacji. Postępujemy tak jak przy tworzeniu lasu losowego, tylko co drugi klasyfikator w lesie to NBC. Jeden z klasyfikatorów (NBC lub drzewo ID3) może pochodzić z istniejącej implementacji. Przed rozpoczęciem realizacji projektu proszę zapoznać się z zawartością [strony](http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html).

## Twórcy projektu:

  💠 Hubert Rutkowski
  
  💠 Adam Szumada

## Wybieranie badań / eksperymentów do wykonania:
Badania uruchamia się poprzez odpowiednią konfigurację pliku main.py.
Należy w nim podać pojedynczo lub w kolejce doświadczenia, które chcemy wykonać.
Do każdego doświdczenia należy w argumencie wpisać ciąg znaków odpowiadający danemu zbiorowi danych.
Możliwe opcje:

    'corona' -> corona.csd
    'divorce' -> divorce.csv
    'glass' -> glass.csv
    'letter' -> letter-recognition.csv
    'loan_approval -> loan_approval.csv

# Potrzebne moduły:
    scikit-learn
    pandas
    matplotlib
    seaborn
    openpyxl

# Uruchamianie badań (programu):
    python main.py