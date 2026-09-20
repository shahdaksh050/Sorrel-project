"""Column-role vocabulary (src.core.vocab): non-English synonyms, English
regression table, and whole-token negatives."""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.profiler import (
    has_identifier_name_hint,
    is_lat_name,
    is_lon_name,
    profile_dataframe,
)
from src.core.vocab import (
    ROLE_TOKENS,
    foreign_tokens,
    has_role,
    name_tokens,
    resolve_foreign,
    role_tokens,
)
from src.tools.basket import _detect_columns as basket_columns
from src.tools.elasticity import _detect_columns as elasticity_columns
from src.tools.equity import _is_binary_outcome_name, _is_pay, _is_protected


def test_name_tokens_strips_accents_and_splits_camel_case():
    assert name_tokens("Quantité") == ["quantite"]
    assert name_tokens("fechaPedido") == ["fecha", "pedido"]
    assert name_tokens("Straße-Nr") == ["strasse", "nr"]
    assert name_tokens("heartRate") == ["heart", "rate"]


@pytest.mark.parametrize("role", sorted(ROLE_TOKENS))
def test_every_token_is_already_normalised(role):
    for token in role_tokens(role):
        if role == "price" and "_" in token:
            continue  # elasticity matches "unit_price" on a joined bigram
        assert name_tokens(token) == [token], (role, token)


@pytest.mark.parametrize(
    "name, role",
    [
        ("precio_unitario", "price"),
        ("Preis", "price"),
        ("prix_ht", "price"),
        ("preço", "price"),
        ("cantidad", "quantity"),
        ("quantité", "quantity"),
        ("Menge", "quantity"),
        ("quantidade_vendida", "quantity"),
        ("fecha_pedido", "date"),
        ("Datum", "date"),
        ("Geschlecht", "protected_attribute"),
        ("sexo", "protected_attribute"),
        ("Nationalität", "protected_attribute"),
        ("Gehalt", "pay"),
        ("sueldo_mensual", "pay"),
        ("salaire", "pay"),
        ("Lohn", "pay"),
        ("pedido_id", "order"),
        ("Bestellung", "order"),
        ("commande", "order"),
        ("producto", "item"),
        ("Artikel", "item"),
        ("produit", "item"),
        ("duración", "duration"),
        ("Dauer", "duration"),
        ("durée", "duration"),
        ("evento", "event"),
        ("Ereignis", "event"),
        ("événement", "event"),
        ("hora", "time_of_day"),
        ("Uhrzeit", "time_of_day"),
        ("latitud", "geo_lat"),
        ("Längengrad", "geo_lon"),
        ("Dosis", "dose_x_strong"),
        ("Temperatur", "dose_x_strong"),
        ("Zeit", "dose_x_weak"),
        ("Patient", "entity_group"),
        ("Kunde", "entity"),
        ("cliente", "entity"),
        ("rechazado", "adverse_outcome"),
        ("contratado", "binary_outcome"),
        ("abgelehnt", "binary_outcome"),
    ],
)
def test_foreign_names_are_recognised(name, role):
    assert has_role(name, role)


@pytest.mark.parametrize(
    "name, role",
    [
        ("customer_id", "id"), ("zipCode", "id"), ("row_index", "id"), ("sku", "id"),
        ("start_lat", "geo_lat"), ("Longitude", "geo_lon"), ("lng", "geo_lon"),
        ("survival_time", "duration"), ("tenure", "duration"),
        ("churned", "event"), ("status", "event"), ("death", "event"),
        ("unit_price", "price"), ("fare", "price"), ("price", "price"),
        ("qty", "quantity"), ("units_sold", "quantity"), ("orders", "quantity"),
        ("invoice_no", "order"), ("basket", "order"), ("session_id", "order"),
        ("product_name", "item"), ("category", "item"), ("SKU", "item"),
        ("subject", "entity_group"), ("batchId", "entity_group"), ("plate", "entity_group"),
        ("gender", "protected_attribute"), ("Ethnicity", "protected_attribute"),
        ("salary", "pay"), ("annual_income", "pay"), ("bonus", "pay"),
        ("hired", "binary_outcome"), ("attrition", "binary_outcome"),
        ("dose", "dose_x_strong"), ("wavelength", "dose_x_strong"), ("ph", "dose_x_strong"),
        ("time", "dose_x_weak"), ("age", "dose_x_weak"),
        ("date", "date"), ("timestamp", "date"), ("hour", "time_of_day"),
    ],
)
def test_english_regression(name, role):
    assert has_role(name, role)


@pytest.mark.parametrize(
    "name, role",
    [
        ("paid", "id"), ("valid", "id"), ("casino", "id"),
        ("inflation", "geo_lat"), ("latency", "geo_lat"), ("long_term_debt", "geo_lon"),
        ("population", "geo_lat"), ("population", "pay"), ("population", "quantity"),
        ("uv_index", "quantity"), ("data_quality", "date"), ("data", "date"),
        ("update", "date"), ("latency", "date"), ("paid", "pay"), ("paid", "price"),
        ("payload", "pay"), ("payment_id", "pay"), ("sexy", "protected_attribute"),
        ("essex", "protected_attribute"), ("holder", "entity_group"),
        ("horas_trabajadas", "time_of_day"), ("hourly", "time_of_day"),
        ("tempo", "duration"), ("dia", "duration"), ("tag", "duration"),
    ],
)
def test_negatives_never_substring_match(name, role):
    assert not has_role(name, role)


def test_uv_index_is_not_an_identifier_but_row_index_is():
    assert has_identifier_name_hint("row_index")
    assert not has_identifier_name_hint("uv_index")
    assert has_identifier_name_hint("identificador")
    assert has_identifier_name_hint("Kundennummer") is False  # compound word: no whole token
    assert has_identifier_name_hint("kunde_nummer")


def test_last_token_only():
    assert has_role("customer_id_old", "id")
    assert not has_role("customer_id_old", "id", last_token_only=True)
    assert has_role("customer_id", "id", last_token_only=True)


def test_geo_helpers_accept_foreign_names():
    assert is_lat_name("latitud") and is_lon_name("Längengrad")
    assert is_lat_name("start_lat") and not is_lat_name("latency")
    assert is_lon_name("long") and not is_lon_name("long_term_debt")


def test_english_ordering_is_preserved_and_synonyms_come_last():
    assert role_tokens("price")[:6] == ("price", "unit_price", "unitprice", "cost_per", "rate", "fare")
    assert role_tokens("quantity")[:7] == ("quantity", "qty", "units", "volume", "sold", "orders", "count")
    assert not set(foreign_tokens("price")) & set(role_tokens("price")[:6])


def test_ambiguous_words_are_omitted():
    for token in ("data", "tempo", "dia", "tag", "genre", "prime", "charge", "lot", "stuck"):
        assert not any(token in ROLE_TOKENS[role] for role in ROLE_TOKENS), token


def test_resolve_foreign_only_uses_synonyms():
    assert resolve_foreign(["a", "Gehalt", "b"], "pay") == "Gehalt"
    assert resolve_foreign(["a", "Gehalt"], "pay", exclude={"Gehalt"}) is None
    assert resolve_foreign(["salary", "bonus"], "pay") is None  # English is the tool's own search


def test_equity_flags_foreign_names():
    assert _is_protected("Geschlecht") and _is_protected("altersgruppe") and _is_protected("grupo_edad")
    assert _is_pay("Gehalt") and _is_pay("salario")
    assert _is_binary_outcome_name("abgelehnt") and _is_binary_outcome_name("Aprobado")
    assert _is_protected("gender_pay_gap")
    assert not _is_pay("paid")


def _orders(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame({
        "pedido": [f"P{i // 4}" for i in range(n)],
        "producto": [f"item{i % 7}" for i in range(n)],
        "fecha_pedido": pd.date_range("2024-01-01", periods=n, freq="h"),
    })


def test_basket_detects_spanish_columns():
    df = _orders()
    assert basket_columns(profile_dataframe(df)) == ("pedido", "producto")


def test_elasticity_detects_foreign_price_and_quantity():
    n = 120
    df = pd.DataFrame({
        "precio_unitario": [10 + (i % 9) * 0.5 for i in range(n)],
        "cantidad": [50 - (i % 9) * 2 for i in range(n)],
        "region": ["a", "b", "c"] * (n // 3),
    })
    assert elasticity_columns(profile_dataframe(df)) == ("precio_unitario", "cantidad")


def test_profiler_unit_hints_follow_foreign_price():
    df = pd.DataFrame({"precio": [9.5 + (i % 11) for i in range(60)], "cantidad": [(i * 7) % 40 + 1 for i in range(60)]})
    cols = {c.name: c for c in profile_dataframe(df).columns}
    assert cols["precio"].unit_hint == "currency"
    assert cols["precio"].aggregation == "mean"
    assert cols["cantidad"].aggregation == "sum"
