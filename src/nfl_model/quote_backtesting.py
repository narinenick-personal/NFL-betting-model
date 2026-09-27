"""Validate archived prop quotes and grade an explicitly supplied quote cohort."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data_quality import require_columns, require_unique_keys
from nfl_model.player_dataset import OTHER
from nfl_model.player_props import PropContract, price_prop
from nfl_model.settlement import american_profit, implied_probability, unit_profit

QUOTE_COLUMNS = ("quote_id", "game_id", "team", "player_id", "book", "quoted_at", "statistic", "side",
                 "line", "american_odds", "participation", "stake", "closing_at", "closing_line", "closing_odds")


def _aware_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Quote timestamps require an explicit UTC offset or Z")
    return stamp.tz_convert("UTC")


def validate_quotes(quotes: pd.DataFrame, schedules: pd.DataFrame, players: pd.DataFrame, *, fit_end) -> pd.DataFrame:
    require_columns(quotes, list(QUOTE_COLUMNS[:11]), "archived prop quotes")
    rows = quotes.copy()
    if rows.empty:
        raise ValueError("The quote file is empty")
    for column in ("quote_id", "game_id", "team", "player_id", "book", "statistic", "side", "participation"):
        rows[column] = rows[column].astype("string").str.strip().replace("", pd.NA)
        if rows[column].isna().any():
            raise ValueError(f"Missing quote {column}")
    require_unique_keys(rows, ["quote_id"], "quotes")
    require_unique_keys(schedules, ["game_id"], "quote schedules")
    require_unique_keys(players, ["game_id", "team", "player_id"], "quote candidates")
    if rows.player_id.eq(OTHER).any():
        raise ValueError("OTHER cannot be quoted as an individual player")
    rows["quoted_at"] = pd.to_datetime([_aware_timestamp(v) for v in rows.quoted_at], utc=True)
    start = schedules.loc[schedules.game_id.isin(rows.game_id), ["game_id", "season", "gameday", "gametime"]].copy()
    kickoff = pd.to_datetime(start.gameday.astype(str).str[:10] + " " + start.gametime.astype(str), errors="raise")
    start["kickoff"] = kickoff.dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    rows = rows.merge(start[["game_id", "season", "kickoff"]], on="game_id", how="left", validate="many_to_one")
    if rows.kickoff.isna().any() or not rows.season.eq(2025).all():
        raise ValueError("The saved bundle prices completed 2025 historical games only")
    if not rows.quoted_at.lt(rows.kickoff).all():
        raise ValueError("Quotes must precede kickoff")
    # This feature builder does not recreate forecasts made before game day.
    if not rows.quoted_at.dt.tz_convert("America/New_York").dt.date.eq(rows.kickoff.dt.tz_convert("America/New_York").dt.date).all():
        raise ValueError("Replay accepts game-day quotes only; earlier feature vintages are unavailable")
    cutoff = pd.Timestamp(fit_end)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if not rows.quoted_at.gt(cutoff + pd.Timedelta(days=1)).all():
        raise ValueError("Quote overlaps the model fitting period")
    identities = pd.MultiIndex.from_frame(players[["game_id", "team", "player_id"]])
    if not pd.MultiIndex.from_frame(rows[["game_id", "team", "player_id"]]).isin(identities).all():
        raise ValueError("A quoted player is outside the lagged forecast candidates")
    for column in ("line", "american_odds"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    rows["stake"] = pd.to_numeric(rows["stake"], errors="raise") if "stake" in rows else 1.
    if not np.isfinite(rows.stake).all() or not rows.stake.gt(0).all():
        raise ValueError("Stakes must be finite and positive")
    for column in ("closing_at", "closing_line", "closing_odds"):
        if column not in rows:
            rows[column] = np.nan
    closing = rows[["closing_at", "closing_line", "closing_odds"]].notna()
    if (closing.any(axis=1) & ~closing.all(axis=1)).any():
        raise ValueError("Supply closing timestamp, line and odds together, or leave all missing")
    for row in rows.itertuples():
        PropContract(row.statistic, row.line, row.side, row.participation).validate()
        american_profit(row.american_odds)
        if pd.notna(row.closing_at):
            timestamp = _aware_timestamp(row.closing_at)
            if not row.quoted_at <= timestamp < row.kickoff:
                raise ValueError("Closing snapshot must be after entry and before kickoff")
            PropContract(row.statistic, float(row.closing_line), row.side, row.participation).validate()
            american_profit(float(row.closing_odds))
    return rows


def backtest_quotes(quotes, players, simulation_provider):
    """Price/grade every validated supplied quote; no optimized selection or sizing."""
    actuals = players.set_index(["game_id", "team", "player_id"])
    output = []
    for game_id, group in quotes.groupby("game_id", sort=False):
        records = simulation_provider(game_id)
        for row in group.itertuples():
            record = records[row.team]
            candidates, draws = record["candidates"], record["samples"]
            if not candidates.game_id.eq(game_id).all():
                raise ValueError("Simulation provider returned the wrong game")
            matches = np.flatnonzero(candidates.player_id.eq(row.player_id).to_numpy())
            if len(matches) != 1:
                raise ValueError("Quote/player simulation identity mismatch")
            i = matches[0]
            actual = actuals.loc[(game_id, row.team, row.player_id)]
            contract = PropContract(row.statistic, row.line, row.side, row.participation)
            priced = price_prop(draws[row.statistic][:, i], contract, row.american_odds, offense_active=draws["offense_active"][:, i])
            if row.participation == "offense_snap_required" and pd.isna(actual.offense_active):
                raise ValueError(f"Unknown actual offensive participation for quote {row.quote_id}; cannot settle this contract")
            void = row.participation == "offense_snap_required" and actual.offense_active == 0
            value = float(actual[row.statistic])
            grade = 0 if void else int(np.sign(value - row.line) * (1 if row.side == "over" else -1))
            net = 0. if void else float(unit_profit(grade, row.american_odds)) * row.stake
            line_clv, price_clv = np.nan, np.nan
            if pd.notna(row.closing_at):
                line_clv = (float(row.closing_line) - row.line) * (1 if row.side == "over" else -1)
                if row.line == float(row.closing_line):
                    price_clv = 100 * (implied_probability(float(row.closing_odds)) - implied_probability(row.american_odds))
            output.append({**row._asdict(), **priced, "actual": value, "result": "void" if void else {1: "win", -1: "loss", 0: "push"}[grade],
                           "net_profit": net, "expected_net_profit": row.stake * priced["expected_net_profit_per_unit"],
                           "line_clv": line_clv, "price_clv_implied_points": price_clv})
    result = pd.DataFrame(output).drop(columns="Index", errors="ignore")
    summary = {"quotes": len(result), "stake": float(result.stake.sum()), "net_profit": float(result.net_profit.sum()),
               "roi": float(result.net_profit.sum() / result.stake.sum()), "expected_net_profit": float(result.expected_net_profit.sum()),
               "wins": int(result.result.eq("win").sum()), "losses": int(result.result.eq("loss").sum()),
               "pushes": int(result.result.eq("push").sum()), "voids": int(result.result.eq("void").sum()),
               "closing_quotes": int(result.closing_at.notna().sum())}
    return result, summary
