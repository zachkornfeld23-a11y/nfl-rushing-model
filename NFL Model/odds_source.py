from __future__ import annotations

import statistics
import time

import pandas as pd
import requests

from core import NAME_TO_TEAM, norm


BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
BOOKS = ["prizepicks", "draftkings", "fanduel", "betmgm", "betrivers"]


class OddsApiError(RuntimeError):
    pass


class OddsApiRateLimitError(OddsApiError):
    pass


class OddsApiCreditsError(OddsApiError):
    pass


def _safe_error(response):
    """Return API error information without echoing the request URL/API key."""
    try:
        body = response.json()
        code = body.get("error_code") or body.get("code") or ""
        message = body.get("message") or body.get("error") or response.reason
    except Exception:
        code = ""
        message = response.reason or f"HTTP {response.status_code}"
    return str(code), str(message)


def _request(url, params, retries=4):
    """HTTP request with clean 429 handling and exponential-style backoff."""
    retry_delays = [2, 4, 8, 12]

    for attempt in range(retries + 1):
        response = requests.get(url, params=params, timeout=25)

        meta = {
            "remaining": response.headers.get("x-requests-remaining"),
            "used": response.headers.get("x-requests-used"),
            "last": response.headers.get("x-requests-last"),
        }

        if response.ok:
            return response.json(), meta

        code, message = _safe_error(response)

        # Monthly usage exhausted.
        if code == "OUT_OF_USAGE_CREDITS" or "usage credit" in message.lower():
            raise OddsApiCreditsError(
                "The Odds API monthly usage credits are exhausted. "
                f"Credits remaining: {meta.get('remaining') or '0'}."
            )

        # Temporary request-rate limit.
        if response.status_code == 429 or code == "EXCEEDED_FREQ_LIMIT":
            if attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else retry_delays[attempt]
                except (TypeError, ValueError):
                    wait = retry_delays[attempt]

                wait = max(wait, retry_delays[attempt])
                time.sleep(wait)
                continue

            raise OddsApiRateLimitError(
                "The Odds API is temporarily rate-limiting requests. "
                "The dashboard retried automatically but was still limited. "
                "Wait 1–2 minutes and refresh again."
            )

        raise OddsApiError(
            f"The Odds API returned HTTP {response.status_code}"
            + (f" ({code})" if code else "")
            + (f": {message}" if message else "")
        )

    raise OddsApiError("The Odds API request failed.")


def fetch_events(api_key):
    """Fetch current NFL event IDs. app.py caches this result."""
    return _request(
        f"{BASE}/sports/{SPORT}/events",
        {
            "apiKey": api_key,
            "dateFormat": "iso",
        },
    )


def fetch_live_rush_attempts(api_key, schedule, week, events):
    """
    Pull standard player_rush_attempts props event-by-event.

    PrizePicks is the primary selected line.
    If PrizePicks is unavailable, the median of available supported sportsbooks
    is used as the fallback line.
    """
    target_games = set()

    for _, game in schedule[schedule["week"] == week].iterrows():
        target_games.add(
            frozenset([game["home_team"], game["away_team"]])
        )

    wanted_events = []

    for event in events:
        home = NAME_TO_TEAM.get(event.get("home_team"))
        away = NAME_TO_TEAM.get(event.get("away_team"))

        if home and away and frozenset([home, away]) in target_games:
            wanted_events.append(event)

    book_player = {}
    credits_remaining = None
    credits_used = None
    credits_last_refresh = 0

    for index, event in enumerate(wanted_events):
        # Avoid bursting a full slate of event-level prop requests.
        if index > 0:
            time.sleep(1.0)

        data, meta = _request(
            f"{BASE}/sports/{SPORT}/events/{event['id']}/odds",
            {
                "apiKey": api_key,
                "bookmakers": ",".join(BOOKS),
                "markets": "player_rush_attempts",
                "oddsFormat": "american",
                "dateFormat": "iso",
                "includeMultipliers": "true",
            },
        )

        credits_remaining = meta.get("remaining") or credits_remaining
        credits_used = meta.get("used") or credits_used

        try:
            credits_last_refresh += int(meta.get("last") or 0)
        except (TypeError, ValueError):
            pass

        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker.get("key")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_rush_attempts":
                    continue

                for outcome in market.get("outcomes", []):
                    if str(outcome.get("name", "")).lower() != "over":
                        continue

                    player = outcome.get("description") or outcome.get("participant")
                    point = outcome.get("point")

                    if player is None or point is None:
                        continue

                    key = norm(player)
                    book_player.setdefault(key, {})
                    book_player[key].setdefault("display_name", player)
                    book_player[key][book_key] = float(point)

    rows = []

    for player_key, data in book_player.items():
        sportsbook_lines = [
            data.get(book)
            for book in ["draftkings", "fanduel", "betmgm", "betrivers"]
            if data.get(book) is not None
        ]

        consensus = statistics.median(sportsbook_lines) if sportsbook_lines else None
        prizepicks = data.get("prizepicks")

        if prizepicks is not None:
            selected_line = prizepicks
            line_source = "PrizePicks via The Odds API"
        else:
            selected_line = consensus
            line_source = "Sportsbook median via The Odds API"

        rows.append(
            {
                "player_key": player_key,
                "player": data["display_name"],
                "selected_line": selected_line,
                "line_source": line_source,
                "prizepicks": prizepicks,
                "sportsbook_consensus": consensus,
                "draftkings": data.get("draftkings"),
                "fanduel": data.get("fanduel"),
                "betmgm": data.get("betmgm"),
                "betrivers": data.get("betrivers"),
            }
        )

    return pd.DataFrame(rows), {
        "events_queried": len(wanted_events),
        "credits_last_refresh": credits_last_refresh,
        "credits_remaining": credits_remaining,
        "credits_used": credits_used,
    }
