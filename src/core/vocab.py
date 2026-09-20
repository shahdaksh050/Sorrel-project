"""One home for column-name role vocabulary.

Every tool used to carry its own English token tuples ("id", "price",
"salary"...). They live here instead, each with Spanish / French / German /
Portuguese synonyms appended, so `precio_unitario`, `Geschlecht` or
`quantité` are recognised like their English counterparts.

Rules the vocabulary keeps:
  * Whole-token matching only (see `name_tokens`) — never substring, so
    "paid"/"inflation"/"latency" never hit "id"/"lat"/"nfl...".
  * Each role's tuple is ORDERED: the original English tokens first, in their
    original order (elasticity/basket rank by index), synonyms appended after.
  * Every token is stored accent-stripped and lowercase, exactly as
    `name_tokens` emits it.
  * Ambiguous words are left out on purpose: "data" (pt date; an English word),
    "tempo" (pt time; a music measure), "dia" (pt day; diameter), "tag",
    "genre", "prime", "lot", "charge", "probe", "stuck", "handicap".
Each tool keeps its own token *derivation* (plural stripping, bigrams); this
module owns only the word lists.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def name_tokens(name: object) -> list[str]:
    """Column name -> lowercase, accent-free tokens, split on camelCase and any
    non-alphanumeric ("heartRate", "heart-rate" -> heart, rate; "Quantité" ->
    quantite; "Straße" -> strasse)."""
    text = unicodedata.normalize("NFKD", str(name).replace("ß", "ss"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in _NON_ALNUM.split(_CAMEL_BOUNDARY.sub("_", text).lower()) if t]


# role -> (english tokens in their original order, synonyms appended).
_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "id": (
        ("id", "uuid", "guid", "index", "key", "code", "number", "no",
         "zip", "zipcode", "postcode", "phone", "sku"),
        ("identificador", "identifiant", "kennung", "nummer", "numero", "clave", "cle",
         "schluessel", "schlussel", "codigo"),
    ),
    "coded_key": (
        ("id", "ids", "uuid", "guid", "key", "code", "zip", "zipcode", "postal", "postcode"),
        ("identificador", "identifiant", "kennung", "clave", "cle", "schluessel", "schlussel",
         "codigo", "plz"),
    ),
    "geo_lat": (("lat", "latitude"), ("latitud", "breitengrad")),
    "geo_lon": (("lon", "lng", "longitude"), ("longitud", "langengrad", "laengengrad")),
    "duration": (
        ("time", "duration", "tenure", "day", "month", "week", "survival", "followup",
         "lifetime", "elapsed"),
        ("duracion", "tiempo", "antiguedad", "dias", "mes", "semana", "supervivencia", "seguimiento",
         "duree", "temps", "anciennete", "jour", "mois", "semaine", "survie", "suivi",
         "dauer", "zeit", "betriebszugehorigkeit", "tage", "monat", "woche", "uberlebenszeit",
         "duracao", "sobrevivencia", "seguimento"),
    ),
    "event": (
        ("event", "status", "churn", "churned", "censor", "censored", "died", "death", "dead",
         "deceased", "attrition", "failure", "failed", "dropout", "converted", "relapse", "left",
         "terminated"),
        ("evento", "estado", "fallecido", "muerte", "abandono", "censurado", "fallo",
         "evenement", "statut", "deces", "defaillance",
         "ereignis", "ausfall", "abwanderung", "zensiert",
         "situacao", "obito", "falha", "cancelamento"),
    ),
    "price": (
        ("price", "unit_price", "unitprice", "cost_per", "rate", "fare"),
        ("precio", "tarifa", "prix", "tarif", "preis", "preco"),
    ),
    "quantity": (
        ("quantity", "qty", "units", "volume", "sold", "orders", "count"),
        ("cantidad", "unidades", "vendido", "quantite", "unites", "vendu",
         "menge", "anzahl", "verkauft", "quantidade", "qtd", "qtde"),
    ),
    "order": (
        ("order", "transaction", "invoice", "basket", "receipt", "cart", "ticket", "session"),
        ("pedido", "orden", "transaccion", "factura", "cesta", "carrito", "recibo", "sesion",
         "commande", "facture", "panier", "recu",
         "bestellung", "auftrag", "transaktion", "rechnung", "warenkorb", "beleg", "sitzung",
         "transacao", "fatura", "carrinho", "sessao"),
    ),
    "item": (
        ("item", "product", "sku", "article", "description", "category", "good"),
        ("articulo", "producto", "descripcion", "categoria", "mercancia",
         "produit", "categorie", "marchandise",
         "artikel", "produkt", "beschreibung", "kategorie", "ware",
         "produto", "artigo", "descricao"),
    ),
    "entity": (
        ("customer", "client", "user", "account", "patient", "member", "employee",
         "subscriber", "sensor", "device", "station", "subject", "participant",
         "player", "team", "store", "shop", "site", "machine", "vehicle", "school",
         "hospital", "company", "firm", "ticker", "symbol", "product", "sku",
         "household", "person", "respondent", "meter", "asset", "fund", "branch",
         "clinic", "farm", "animal"),
        ("cliente", "usuario", "paciente", "empleado", "dispositivo", "tienda", "producto",
         "empresa", "persona", "cuenta", "escuela",
         "utilisateur", "employe", "magasin", "produit", "entreprise", "personne", "compte", "ecole",
         "kunde", "benutzer", "nutzer", "mitarbeiter", "gerat", "produkt", "unternehmen", "konto", "schule",
         "funcionario", "loja", "produto", "pessoa", "conta", "escola"),
    ),
    "entity_group": (
        ("subject", "patient", "participant", "site", "batch", "plot", "replicate", "cluster",
         "school", "plate", "sample", "station", "unit"),
        ("sujeto", "paciente", "participante", "sitio", "lote", "parcela", "replica", "escuela",
         "estacion", "unidad", "muestra",
         "sujet", "parcelle", "ecole", "unite", "echantillon",
         "proband", "teilnehmer", "standort", "parzelle", "schule", "einheit",
         "sujeito", "escola", "estacao", "unidade", "amostra"),
    ),
    "gender": (
        ("gender", "sex"),
        ("genero", "sexo", "sexe", "geschlecht"),
    ),
    "age": (("age",), ("edad", "alter", "idade")),
    "group": (("group",), ("grupo", "gruppe", "groupe")),
    "protected_attribute": (
        ("gender", "sex", "race", "ethnicity", "ethnic", "agegroup", "disability",
         "nationality", "religion", "marital", "veteran"),
        ("genero", "sexo", "sexe", "geschlecht",
         "raza", "etnia", "discapacidad", "nacionalidad", "civil", "veterano",
         "ethnie", "nationalite", "matrimonial",
         "rasse", "behinderung", "nationalitat", "staatsangehorigkeit", "familienstand",
         "altersgruppe",
         "raca", "deficiencia", "nacionalidade", "religiao"),
    ),
    "pay": (
        ("salary", "pay", "wage", "income", "compensation", "bonus", "earnings", "earning"),
        ("sueldo", "salario", "ingreso", "ingresos", "remuneracion", "compensacion", "bonificacion",
         "salaire", "revenu", "remuneration", "paie",
         "gehalt", "lohn", "einkommen", "verdienst", "vergutung", "bezahlung", "entgelt",
         "renda", "remuneracao", "ordenado"),
    ),
    "adverse_outcome": (
        ("attrition", "terminated", "rejected", "churn", "left", "attrited", "fired",
         "resigned", "quit", "churned"),
        ("rechazado", "desercion", "abandono", "despedido", "renuncia",
         "rejete", "licencie", "demission",
         "abwanderung", "abgelehnt", "gekundigt", "ausgeschieden", "entlassen",
         "rejeitado", "desligado", "demitido"),
    ),
    "binary_outcome": (
        ("hired", "promoted", "attrition", "terminated", "approved", "rejected",
         "selected", "churn", "left"),
        ("contratado", "promovido", "aprobado", "seleccionado",
         "embauche", "promu", "approuve", "selectionne",
         "eingestellt", "befordert", "genehmigt", "ausgewahlt",
         "aprovado", "selecionado",
         # every adverse synonym is also an outcome name (kept in step below)
         ),
    ),
    "dose_x_strong": (
        ("dose", "conc", "concentration", "temperature", "temp", "depth", "pressure", "ph",
         "wavelength", "dilution", "cycle", "generation"),
        ("dosis", "concentracion", "temperatura", "profundidad", "presion", "dilucion", "ciclo",
         "generacion",
         "profondeur", "pression", "cycle",
         "konzentration", "temperatur", "tiefe", "druck", "wellenlange", "verdunnung", "zyklus",
         "profundidade", "pressao", "diluicao", "generacao"),
    ),
    "dose_x_weak": (
        ("time", "age", "day", "hour", "minute", "distance"),
        ("tiempo", "edad", "hora", "minuto", "distancia",
         "temps", "heure", "jour",
         "zeit", "alter", "stunde", "entfernung", "abstand",
         "idade", "minuto"),
    ),
    "date": (
        ("date", "timestamp", "datetime", "time", "day"),
        ("fecha", "horodatage", "datum", "zeitstempel"),
    ),
    "time_of_day": (("hour",), ("hora", "heure", "uhrzeit", "stunde")),
}


def _build() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for role, (english, synonyms) in _SPEC.items():
        extra = _SPEC["adverse_outcome"][1] if role == "binary_outcome" else ()
        out[role] = tuple(dict.fromkeys((*english, *synonyms, *extra)))
    return out


_ROLE_ORDER: dict[str, tuple[str, ...]] = _build()
_ENGLISH: dict[str, frozenset[str]] = {r: frozenset(e) for r, (e, _) in _SPEC.items()}

#: role -> every recognised token (English + synonyms).
ROLE_TOKENS: dict[str, frozenset[str]] = {r: frozenset(t) for r, t in _ROLE_ORDER.items()}


def role_tokens(role: str) -> tuple[str, ...]:
    """The role's tokens in priority order: original English first."""
    return _ROLE_ORDER[role]


def english_tokens(role: str) -> tuple[str, ...]:
    """Only the original English tokens, in their original order."""
    return _SPEC[role][0]


def foreign_tokens(role: str) -> tuple[str, ...]:
    """Only the non-English synonyms, in order."""
    return tuple(t for t in _ROLE_ORDER[role] if t not in _ENGLISH[role])


def has_role(name: object, role: str, last_token_only: bool = False) -> bool:
    """True when a whole token of `name` (only the last one when
    `last_token_only`) belongs to `role`. Never a substring match."""
    tokens = name_tokens(name)
    if not tokens:
        return False
    vocabulary = ROLE_TOKENS[role]
    return tokens[-1] in vocabulary if last_token_only else any(t in vocabulary for t in tokens)


def column_role(profile: Any, name: object, role: str, last_token_only: bool = False) -> bool:
    """True when a validated override (`profile.role_overrides`) assigns `role`
    to `name`, or the name itself carries the role (`has_role`)."""
    overrides = getattr(profile, "role_overrides", None) or {}
    return overrides.get(str(name)) == role or has_role(name, role, last_token_only)


def roles_for(profile: Any, role: str) -> list[str]:
    """Columns holding `role`, validated-override columns first, then name matches."""
    overrides = getattr(profile, "role_overrides", None) or {}
    forced = [c for c, r in overrides.items() if r == role]
    named = [c.name for c in getattr(profile, "columns", ()) if c.name not in forced and has_role(c.name, role)]
    return [*forced, *named]


def resolve_foreign(columns: Iterable[object], role: str, exclude: set[str] | None = None) -> str | None:
    """First column whose name holds a NON-English synonym of `role` as a whole
    token — the fallback after a tool's English candidate search finds nothing,
    so English datasets resolve exactly as before."""
    blocked = exclude or set()
    names = [str(c) for c in columns if str(c) not in blocked]
    tokenised = {n: set(name_tokens(n)) for n in names}
    for token in foreign_tokens(role):
        for n in names:
            if token in tokenised[n]:
                return n
    return None
