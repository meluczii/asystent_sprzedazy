# Optymalizator Sprzedaży Ubrań

Aplikacja Flask do:
- generowania opisów ofert i hashtagów przez Claude,
- importu CSV,
- analizy sprzedaży przez Pandas,
- wyliczania Top 5 produktów, średniej wartości i miesięcznych sum,
- generowania narracyjnego podsumowania AI,
- eksportu raportu do HTML,
- walidacji plików i długości danych,
- własnego limitu zapytań AI (10 na sesję).

## Uruchomienie

1. Utwórz środowisko:
   `python -m venv venv`

2. Aktywuj je w Windows:
   `venv\Scripts\activate`

3. Zainstaluj biblioteki:
   `pip install -r requirements.txt`

4. W `.env` wpisz prawidłowy `ANTHROPIC_API_KEY`.

5. Uruchom:
   `python app.py`

6. Otwórz:
   `http://127.0.0.1:5000`

## Format CSV

Aplikacja nie wymaga jednej sztywnej nazwy kolumn. Rozpoznaje m.in.:
`produkt/nazwa`, `marka`, `rozmiar`, `kolor`, `stan`, `materiał`,
`cena/wartość`, `data`, `ilość`.

Dla pełnego raportu sprzedaży najlepiej użyć np.:
`produkt,cena,data,ilość`.

## Uwaga

Nie umieszczaj prawdziwego klucza API w repozytorium Git. Plik `.env` jest dodany do `.gitignore`.
