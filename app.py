import io
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

MAX_AI_REQUESTS = 10
MAX_TEXT_LENGTH = 2000
ALLOWED_EXTENSIONS = {".csv"}

# Column aliases make the CSV importer tolerant of different Polish/English headers.
ALIASES = {
    "product": ["produkt", "nazwa", "nazwa produktu", "product", "item", "title"],
    "brand": ["marka", "brand"],
    "size": ["rozmiar", "size"],
    "color": ["kolor", "color"],
    "condition": ["stan", "condition"],
    "material": ["materiał", "material"],
    "category": ["kategoria", "category", "typ"],
    "price": ["cena", "price", "wartość", "wartosc", "kwota", "amount"],
    "date": ["data", "date", "data sprzedaży", "sale date", "order date"],
    "quantity": ["ilość", "ilosc", "quantity", "sztuki", "qty"],
}


def ai_requests_used():
    return int(session.get("ai_requests", 0))


def consume_ai_request():
    used = ai_requests_used()
    if used >= MAX_AI_REQUESTS:
        return False
    session["ai_requests"] = used + 1
    session.modified = True
    return True


def reset_ai_limit():
    session["ai_requests"] = 0
    session.modified = True


def clean_column(name):
    return re.sub(r"\s+", " ", str(name).strip().lower())


def find_column(df, aliases):
    normalized = {clean_column(c): c for c in df.columns}
    for alias in aliases:
        if clean_column(alias) in normalized:
            return normalized[clean_column(alias)]
    return None


def read_csv_file(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("Wybierz plik CSV.")

    suffix = Path(file_storage.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Dozwolone są tylko pliki CSV.")

    raw = file_storage.read()
    if len(raw) > app.config["MAX_CONTENT_LENGTH"]:
        raise ValueError("Plik jest za duży. Maksymalny rozmiar to 5 MB.")

    if not raw:
        raise ValueError("Plik CSV jest pusty.")

    # Try common encodings used by spreadsheet exports.
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "latin1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError("Nie udało się odczytać kodowania pliku CSV.") from last_error

    try:
        # sep=None lets pandas detect comma/semicolon/tab in simple CSV files.
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Nieprawidłowy format CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV nie zawiera żadnych rekordów.")
    if len(df.columns) < 1:
        raise ValueError("CSV nie zawiera kolumn.")

    return df


def normalize_sales_data(df):
    """Return useful normalized fields without requiring one rigid CSV schema."""
    result = pd.DataFrame(index=df.index)

    product_col = find_column(df, ALIASES["product"])
    price_col = find_column(df, ALIASES["price"])
    date_col = find_column(df, ALIASES["date"])
    quantity_col = find_column(df, ALIASES["quantity"])

    result["product"] = (
        df[product_col].fillna("Nieznany produkt").astype(str).str.strip()
        if product_col else pd.Series(["Nieznany produkt"] * len(df), index=df.index)
    )

    if price_col:
        result["price"] = (
            df[price_col].astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
            .replace("", pd.NA)
            .astype(float)
        )
    else:
        result["price"] = pd.NA

    if quantity_col:
        result["quantity"] = (
            pd.to_numeric(df[quantity_col], errors="coerce")
            .fillna(1)
            .clip(lower=0)
        )
    else:
        result["quantity"] = 1

    if date_col:
        result["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        result["date"] = pd.NaT

    result["revenue"] = result["price"] * result["quantity"]
    return result


def calculate_report(df):
    sales = normalize_sales_data(df)
    valid_revenue = sales["revenue"].dropna()

    if valid_revenue.empty:
        total_sales = None
        average_order = None
    else:
        total_sales = float(valid_revenue.sum())
        average_order = float(valid_revenue.mean())

    top = (
        sales.groupby("product", dropna=False)["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    top5 = [{"product": str(k), "quantity": float(v)} for k, v in top.items()]

    monthly = []
    dated = sales.dropna(subset=["date"]).copy()
    if not dated.empty:
        dated["month"] = dated["date"].dt.to_period("M").astype(str)
        sums = dated.groupby("month")["revenue"].sum().sort_index()
        monthly = [{"month": m, "value": float(v)} for m, v in sums.items()]

    return {
        "rows": len(df),
        "columns": [str(c) for c in df.columns],
        "total_sales": total_sales,
        "average_order": average_order,
        "top5": top5,
        "monthly": monthly,
    }


def dataframe_context(df, report):
    # Keep the prompt small: send a compact sample plus deterministic Python statistics.
    sample = df.head(20).fillna("").astype(str).to_dict(orient="records")
    return {
        "statistics": report,
        "sample_rows": sample,
    }


def call_claude(prompt, max_tokens=900):
    if not consume_ai_request():
        raise RuntimeError(
            f"Osiągnięto limit {MAX_AI_REQUESTS} zapytań AI w tej sesji."
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "moj_klucz":
        raise RuntimeError(
            "Brak prawidłowego klucza ANTHROPIC_API_KEY w pliku .env."
        )
    if Anthropic is None:
        raise RuntimeError("Brak biblioteki anthropic. Uruchom: pip install -r requirements.txt")

    client = Anthropic(api_key=api_key)
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


@app.context_processor
def inject_globals():
    return {
        "ai_used": ai_requests_used(),
        "ai_limit": MAX_AI_REQUESTS,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    description = request.form.get("description", "").strip()
    style = request.form.get("style", "Minimalistyczny")

    if not description:
        flash("Wpisz opis produktu.", "error")
        return redirect(url_for("index"))

    if len(description) > MAX_TEXT_LENGTH:
        flash(f"Opis jest za długi. Maksymalnie {MAX_TEXT_LENGTH} znaków.", "error")
        return redirect(url_for("index"))

    prompt = f"""
Jesteś pomocnikiem osoby sprzedającej odzież online.
Na podstawie danych produktu przygotuj atrakcyjną, ale prawdziwą ofertę.
Nie wymyślaj cech, których nie podano.

Styl: {style}
Dane produktu:
{description}

Zwróć:
1. Tytuł oferty (maks. 70 znaków)
2. Krótki opis (3-6 zdań)
3. 8-12 trafnych hashtagów
Nie używaj nazw platform sprzedażowych w treści.
"""
    try:
        result = call_claude(prompt, max_tokens=700)
    except RuntimeError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    return render_template("index.html", generated=result, submitted=description)


@app.route("/analiza", methods=["GET", "POST"])
def analiza():
    if request.method == "GET":
        return render_template("analysis.html")

    try:
        df = read_csv_file(request.files.get("csv_file"))
        report = calculate_report(df)

        # AI receives computed statistics + a small sample, not the whole file.
        context = dataframe_context(df, report)
        prompt = f"""
Jesteś analitykiem sprzedaży odzieży.
Napisz krótkie, konkretne narracyjne podsumowanie raportu na podstawie
statystyk policzonych przez Pythona. Nie zmieniaj liczb i nie wymyślaj danych.
Dane:
{context}

Podsumowanie ma mieć:
- 1 akapit ogólny,
- 2-4 najważniejsze obserwacje,
- krótką sugestię, na czym warto skupić sprzedaż.
"""
        summary = call_claude(prompt, max_tokens=800)
        report["summary"] = summary
        report["generated_at"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        report["filename"] = request.files["csv_file"].filename

        session["last_report"] = report
        session.modified = True
        return render_template("analysis.html", report=report)

    except (ValueError, RuntimeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("analiza"))


@app.route("/raport.html")
def raport_html():
    report = session.get("last_report")
    if not report:
        flash("Najpierw wykonaj analizę pliku CSV.", "error")
        return redirect(url_for("analiza"))

    html = render_template("report.html", report=report)
    filename = f"raport_sprzedazy_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    return send_file(
        io.BytesIO(html.encode("utf-8")),
        mimetype="text/html",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/reset-limit", methods=["POST"])
def reset_limit():
    # Useful during local development/testing.
    reset_ai_limit()
    flash("Licznik zapytań został wyzerowany.", "success")
    return redirect(request.referrer or url_for("index"))


@app.errorhandler(413)
def too_large(_error):
    flash("Plik jest za duży. Maksymalny rozmiar to 5 MB.", "error")
    return redirect(url_for("analiza"))


if __name__ == "__main__":
    app.run(debug=True)
