"""Timestamped pregame quote collection, independent of historical model artifacts."""
from __future__ import annotations

from hashlib import sha256
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from nfl_model.player_props import PROP_STATS, PropContract
from nfl_model.settlement import american_profit, grade_market, implied_probability

TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
TEAM_NAMES = {name.casefold(): code for code, name in TEAMS.items()}
LIVE_COLUMNS = ["quote_id", "book", "away_team", "home_team", "kickoff", "quoted_at",
                "market", "side", "line", "american_odds", "player", "statistic", "participation", "source"]
GAME_MARKETS = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}
PROP_MARKETS = {"player_pass_yds": "passing_yards", "player_pass_attempts": "attempts",
                "player_pass_tds": "passing_tds", "player_rush_yds": "rushing_yards",
                "player_rush_attempts": "carries", "player_rush_tds": "rushing_tds",
                "player_reception_yds": "receiving_yards", "player_receptions": "receptions",
                "player_reception_tds": "receiving_tds"}
REGIONS = ("us", "us2", "uk", "eu", "au")
MAX_QUOTES = 20_000
EVENT_COLUMNS = ["away_team", "home_team", "kickoff"]
SELECTION_COLUMNS = [*EVENT_COLUMNS, "market", "side", "player", "statistic", "participation"]


def utc_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Timestamps must include a timezone, such as Z or -04:00")
    return stamp.tz_convert("UTC")


def team_code(value):
    text = str(value).strip()
    code = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}.get(text.upper(), text.upper())
    if code in TEAMS:
        return code
    if text.casefold() in TEAM_NAMES:
        return TEAM_NAMES[text.casefold()]
    raise ValueError("Use a recognized NFL team abbreviation or full team name")


def empty_quotes():
    return pd.DataFrame(columns=LIVE_COLUMNS)


def validate_live_quotes(raw, *, now=None):
    """Validate atomically, preserving quote observation time on import/export."""
    now = utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    required = [c for c in LIVE_COLUMNS if c not in ("player", "statistic", "participation", "source")]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    if len(raw) > MAX_QUOTES:
        raise ValueError(f"A session can hold at most {MAX_QUOTES:,} quotes; export and clear it first")
    rows = raw.copy()
    for name in LIVE_COLUMNS:
        if name not in rows:
            rows[name] = ""
    rows = rows[LIVE_COLUMNS]
    for name in ("quote_id", "book", "market", "side", "player", "statistic", "participation", "source"):
        rows[name] = rows[name].fillna("").astype(str).str.strip()
        if rows[name].str.len().gt(200).any():
            raise ValueError(f"{name} must be 200 characters or fewer")
    if rows.quote_id.eq("").any() or rows.book.eq("").any():
        raise ValueError("Each quote needs a quote_id and sportsbook name")
    if rows.quote_id.duplicated().any():
        raise ValueError("quote_id must be unique within an import")
    rows["book"] = rows.book.str.casefold()
    for name in ("market", "side", "statistic", "participation"):
        rows[name] = rows[name].str.lower()
    for name in ("home_team", "away_team"):
        rows[name] = rows[name].map(team_code)
    if rows.home_team.eq(rows.away_team).any():
        raise ValueError("Home and away teams must differ")
    for name in ("kickoff", "quoted_at"):
        rows[name] = pd.to_datetime([utc_timestamp(v) for v in rows[name]], utc=True)
    if rows.quoted_at.gt(now).any():
        raise ValueError("Quote timestamps cannot be in the future")
    if rows.quoted_at.ge(rows.kickoff).any():
        raise ValueError("Only pregame quotes are supported: quoted_at must precede kickoff")
    for name in ("line", "american_odds"):
        rows[name] = pd.to_numeric(rows[name].replace("", np.nan), errors="raise")
    for index, row in rows.iterrows():
        american_profit(row.american_odds)
        if row.market == "player_prop":
            if not row.player or row.statistic not in PROP_STATS:
                raise ValueError("Player props require a player name and supported statistic")
            rule = row.participation or "book_rules"
            if rule not in ("book_rules", "all_candidates", "offense_snap_required"):
                raise ValueError("Unsupported participation rule")
            PropContract(row.statistic, row.line, row.side,
                         "all_candidates" if rule == "book_rules" else rule).validate()
            rows.at[index, "participation"] = rule
        else:
            if row.player or row.statistic or row.participation:
                raise ValueError("Player fields must be empty for game markets")
            # Board spread lines are the selected team's handicap, unlike the grid's home handicap.
            line = None if pd.isna(row.line) else float(row.line)
            home_line = -line if row.market == "spread" and row.side == "away" and line is not None else line
            grade_market(0, 0, market=row.market, side=row.side, line=home_line)
    return rows.reset_index(drop=True)


def merge_quotes(existing, incoming, *, now=None):
    """Idempotent imports; conflicting IDs never silently rewrite a saved quote."""
    incoming = validate_live_quotes(incoming, now=now)
    if existing.empty:
        return incoming
    joined = pd.concat([existing, incoming], ignore_index=True).drop_duplicates()
    if joined.quote_id.duplicated().any():
        raise ValueError("A quote_id already exists with different values; assign a new ID to a new snapshot")
    return validate_live_quotes(joined, now=now)


def quote_board(quotes, *, now=None, stale_minutes=15):
    """Latest observation per book/selection; old main lines cannot remain 'best'."""
    now = utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    if stale_minutes <= 0 or not np.isfinite(stale_minutes):
        raise ValueError("Quote age threshold must be positive")
    rows = quotes.sort_values(["quoted_at", "quote_id"]).drop_duplicates(
        ["book", *SELECTION_COLUMNS], keep="last").copy()
    rows["age_minutes"] = (now - rows.quoted_at).dt.total_seconds() / 60
    rows["status"] = np.where(rows.kickoff.le(now), "Started / past",
                              np.where(rows.age_minutes.gt(stale_minutes), "Stale", "Recent"))
    rows["implied_probability"] = rows.american_odds.map(implied_probability)
    rows["profit_per_unit"] = rows.american_odds.map(american_profit)
    rows["best_same_line"] = False
    # Unknown book-specific prop rules are not comparable across sportsbooks.
    eligible = rows.status.eq("Recent") & rows.participation.ne("book_rules")
    group = rows.loc[eligible].groupby([*SELECTION_COLUMNS, "line"], dropna=False)
    if eligible.any():
        best = group.profit_per_unit.transform("max")
        books = group.book.transform("nunique")
        rows.loc[eligible, "best_same_line"] = rows.loc[eligible, "profit_per_unit"].eq(best) & books.ge(2)
    return rows.sort_values(["kickoff", "market", "book"]).reset_index(drop=True)


class OddsAPIError(ValueError):
    """Safe to show to users: contains neither request URL nor provider response body."""


def _api_request(api_key, endpoint, params):
    if not isinstance(api_key, str) or not api_key.strip():
        raise OddsAPIError("Configure an Odds API key first")
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/" + endpoint
    query = urlencode({**params, "apiKey": api_key.strip(), "dateFormat": "iso"})
    try:
        with urlopen(url + "?" + query, timeout=15) as response:
            body = response.read(5_000_001)
            if len(body) > 5_000_000:
                raise OddsAPIError("The provider response was too large; select fewer markets")
            payload = json.loads(body)
            usage = {name: response.headers.get(f"x-requests-{name}", "unknown")
                     for name in ("remaining", "used", "last")}
    except HTTPError as error:
        messages = {401: "API key was rejected", 403: "This API account cannot access the requested feed",
                    422: "The provider rejected the selected markets", 429: "API quota or rate limit reached"}
        raise OddsAPIError(messages.get(error.code, f"Odds provider returned HTTP {error.code}")) from None
    except (URLError, TimeoutError, OSError):
        raise OddsAPIError("Could not reach the odds provider; check your connection and try again") from None
    except (ValueError, UnicodeError):
        raise OddsAPIError("The odds provider returned an invalid response") from None
    return payload, usage


def normalize_api_quotes(payload, *, now=None):
    """Convert official v4 outcomes, keeping provider update timestamps and spread signs."""
    now = utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    events = [payload] if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise OddsAPIError("The odds provider returned an unexpected event format")
    records, event_choices, skipped = [], [], 0
    try:
        for event in events:
            kickoff = utc_timestamp(event["commence_time"])
            if kickoff <= now:
                continue  # Current models and board contract do not support in-play betting.
            home, away = team_code(event["home_team"]), team_code(event["away_team"])
            if event.get("sport_key") != "americanfootball_nfl":
                raise ValueError("Unexpected sport")
            event_choices.append({"id": str(event["id"]), "home_team": home, "away_team": away,
                                  "kickoff": kickoff.isoformat()})
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    key = market["key"]
                    if key not in GAME_MARKETS and key not in PROP_MARKETS:
                        continue
                    for outcome in market.get("outcomes", []):
                        try:
                            prop = key in PROP_MARKETS
                            side = outcome["name"].lower() if prop or key == "totals" else (
                                "home" if outcome["name"] == event["home_team"] else
                                "away" if outcome["name"] == event["away_team"] else "unknown")
                            record = dict(book=book["key"], home_team=home, away_team=away, kickoff=kickoff,
                                          quoted_at=market.get("last_update") or book.get("last_update"),
                                          market="player_prop" if prop else GAME_MARKETS[key], side=side,
                                          line=outcome.get("point", np.nan), american_odds=outcome["price"],
                                          player=outcome.get("description", "") if prop else "",
                                          statistic=PROP_MARKETS[key] if prop else "",
                                          participation="book_rules" if prop else "", source="the_odds_api")
                            identity = json.dumps(record, sort_keys=True, default=str)
                            record["quote_id"] = "api_" + sha256(identity.encode()).hexdigest()[:24]
                            clean = validate_live_quotes(pd.DataFrame([record]), now=now)
                            records.append(clean.iloc[0].to_dict())
                        except (ValueError, TypeError, KeyError):
                            skipped += 1
    except (ValueError, TypeError, KeyError, AttributeError):
        raise OddsAPIError("The odds provider returned an unsupported event structure") from None
    return pd.DataFrame(records, columns=LIVE_COLUMNS) if records else empty_quotes(), event_choices, skipped


def fetch_odds(api_key, *, markets=("h2h", "spreads", "totals"), region="us", event_id=None):
    """One explicit request, no retries/polling; player props require a selected event."""
    allowed = set(GAME_MARKETS) | (set(PROP_MARKETS) if event_id else set())
    if not markets or not set(markets) <= allowed or region not in REGIONS:
        raise OddsAPIError("Select supported markets and one region; props require an event")
    if event_id is not None and (not str(event_id).isalnum() or len(str(event_id)) > 100):
        raise OddsAPIError("Invalid provider event ID")
    endpoint = f"events/{event_id}/odds" if event_id else "odds"
    payload, usage = _api_request(api_key, endpoint,
                                  {"regions": region, "markets": ",".join(markets), "oddsFormat": "american"})
    quotes, events, skipped = normalize_api_quotes(payload)
    return quotes, events, {**usage, "skipped": skipped, "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()}
