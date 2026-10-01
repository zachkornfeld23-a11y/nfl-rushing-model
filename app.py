from pathlib import Path
import os
import pandas as pd
import streamlit as st

from core import *
from odds_source import (
    fetch_events,
    fetch_live_rush_attempts,
    OddsApiRateLimitError,
    OddsApiCreditsError,
    OddsApiError,
)
from pipeline import maybe_rollover


APP = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Rushing Attempts Model",
    page_icon="🏈",
    layout="wide",
)


def api_key():
    env_key = os.environ.get("ODDS_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets["ODDS_API_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def cached_events(_api_key):
    """Cache current NFL event IDs for 10 minutes."""
    return fetch_events(_api_key)


@st.cache_data(ttl=600, show_spinner=False)
def cached_live_lines(_api_key, season, week):
    """
    Cache a full live prop-board pull for 10 minutes.

    This prevents ordinary Streamlit reruns from repeatedly spending API calls.
    """
    schedule = load_schedule(season)
    events, event_meta = cached_events(_api_key)

    live, quota = fetch_live_rush_attempts(
        _api_key,
        schedule,
        week,
        events,
    )

    if quota.get("credits_remaining") is None:
        quota["credits_remaining"] = event_meta.get("remaining")

    if quota.get("credits_used") is None:
        quota["credits_used"] = event_meta.get("used")

    return live, quota


state, rolled, rollover_message = maybe_rollover()
season = int(state["season"])
week = int(state["current_week"])

sched = load_schedule(season)
ranks = week_rankings(APP / "rankings_history.csv", season, week)
prior = history_lines(APP / "historical_lines.csv", season, week - 1)
status = game_status_map(sched, week)

st.title("🏈 Rushing Attempts Model")
st.caption(
    f"{season} Week {week} • Original 50/50 matchup model + live PrizePicks market overlay"
)

if rolled:
    st.success(rollover_message)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Bet Recommendations", "Full Ranking", "Results & History", "Methodology"]
)

with st.sidebar:
    st.header("Live market")

    refresh = st.selectbox(
        "Auto refresh",
        ["15 minutes", "30 minutes", "60 minutes", "Off"],
        index=1,
    )

    st.caption(
        "Live player props are cached for 10 minutes. Event-level requests are "
        "spaced out and automatically retried if the API rate-limits them."
    )

    if st.button("Refresh live lines now", use_container_width=True):
        cached_live_lines.clear()
        cached_events.clear()
        st.rerun()

    st.markdown(f"**Model week:** {week}")

run_every = {
    "15 minutes": "15m",
    "30 minutes": "30m",
    "60 minutes": "60m",
    "Off": None,
}[refresh]


@st.fragment(run_every=run_every)
def live_tables():
    key = api_key()
    live = pd.DataFrame()
    quota = {}

    if key:
        try:
            live, quota = cached_live_lines(key, season, week)

            if not live.empty:
                upsert_lines(
                    APP / "historical_lines.csv",
                    season,
                    week,
                    ranks,
                    live,
                )

        except OddsApiRateLimitError as exc:
            st.warning(str(exc))

        except OddsApiCreditsError as exc:
            st.error(str(exc))

        except OddsApiError as exc:
            st.error(f"Odds API error: {exc}")

        except Exception as exc:
            st.error(
                "Unexpected live-data error. The stored model/history is still intact."
            )
            print("Unexpected live-data error:", repr(exc))

    else:
        st.warning(
            "Add ODDS_API_KEY to .streamlit/secrets.toml or your environment "
            "to enable live lines."
        )

    current = history_lines(APP / "historical_lines.csv", season, week)
    board = build_board(ranks, current, prior, status)
    rec = recommendations(board)

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Week", week)
        c2.metric("Top-10 lines posted", f"{len(rec)}/10")
        c3.metric("API events queried", quota.get("events_queried", "—"))
        c4.metric("API credits remaining", quota.get("credits_remaining", "—"))

        st.subheader("Top 10 bet recommendations")
        st.caption(
            "Flat/Down first → +0.5–1.5 → +2 or more; OG matchup rank breaks ties."
        )

        cols = [
            "Bet Rank",
            "Rank",
            "Player",
            "Team",
            "Opponent",
            "OG Score",
            "Last Week Line",
            "Current Line",
            "Line Move",
            "Market Move",
            "Game Status",
        ]

        st.dataframe(
            rec[cols],
            use_container_width=True,
            hide_index=True,
        )

        if not live.empty:
            with st.expander("Live book-by-book lines"):
                show = live[
                    [
                        "player",
                        "prizepicks",
                        "sportsbook_consensus",
                        "draftkings",
                        "fanduel",
                        "betmgm",
                        "betrivers",
                        "line_source",
                    ]
                ].copy()

                st.dataframe(
                    show,
                    use_container_width=True,
                    hide_index=True,
                )

    with tab2:
        st.subheader("Full ranking")
        st.caption("Expand any player below the table to audit the exact OG Score calculation.")

        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Player-by-player OG Score calculations")

        # The ranking-history row contains the football-model inputs and
        # intermediate percentile values used to create the OG Score.
        explain = ranks.sort_values("rank").copy()

        for _, r in explain.iterrows():
            rank_num = int(r["rank"])
            player = str(r["player"])
            team = str(r["team"])
            opponent = str(r["opponent"])
            score = float(r["og_score"])

            label = f"#{rank_num} {player} — {team} vs {opponent} — OG Score {score:.2f}"

            with st.expander(label):
                if opponent == "BYE":
                    st.info("This player has a bye, so no matchup score is calculated.")
                    continue

                prior_share = float(r["prior_share"])
                latest_share = float(r["latest_share"])
                share_change = float(r["share_change"])
                opp_carries = float(r["opp_rb_carries_pg"])

                # Prefer stored exact intermediates. Fallback recomputation keeps
                # older ranking-history files compatible.
                if "trend_rank" in r.index and pd.notna(r["trend_rank"]):
                    trend_rank = float(r["trend_rank"])
                else:
                    trend_rank = float(
                        explain["share_change"].rank(method="average", ascending=True).loc[r.name]
                    )

                if "trend_percentile" in r.index and pd.notna(r["trend_percentile"]):
                    trend_pct = float(r["trend_percentile"])
                else:
                    n_tmp = len(explain[explain["opponent"] != "BYE"])
                    trend_pct = (trend_rank - 0.5) / n_tmp

                if "defense_rank" in r.index and pd.notna(r["defense_rank"]):
                    defense_rank = float(r["defense_rank"])
                else:
                    defense_rank = float(
                        explain["opp_rb_carries_pg"].rank(method="average", ascending=True).loc[r.name]
                    )

                if "defense_percentile" in r.index and pd.notna(r["defense_percentile"]):
                    defense_pct = float(r["defense_percentile"])
                else:
                    n_tmp = len(explain[explain["opponent"] != "BYE"])
                    defense_pct = (defense_rank - 0.5) / n_tmp

                if "rank_population" in r.index and pd.notna(r["rank_population"]):
                    n = int(float(r["rank_population"]))
                else:
                    n = int(len(explain[explain["opponent"] != "BYE"]))

                trend_points = 50.0 * trend_pct
                defense_points = 50.0 * defense_pct
                rebuilt_score = trend_points + defense_points

                a, b, c, d = st.columns(4)
                a.metric("Previous-week RB share", f"{prior_share:.2%}")
                b.metric("Latest-week RB share", f"{latest_share:.2%}")
                c.metric("Share change", f"{share_change:+.2%}")
                d.metric("Opponent RB carries/game", f"{opp_carries:.2f}")

                st.markdown("#### Usage-trend side")
                st.code(
                    f"""Previous share = {prior_share:.6f} ({prior_share:.2%})
Latest share   = {latest_share:.6f} ({latest_share:.2%})

Share Change = Latest Share - Previous Share
             = {latest_share:.6f} - {prior_share:.6f}
             = {share_change:.6f} ({share_change:+.2%})

Midpoint percentile rank:
Trend Rank       = {trend_rank:.2f} out of {n}
Trend Percentile = (Trend Rank - 0.5) / N
                 = ({trend_rank:.2f} - 0.5) / {n}
                 = {trend_pct:.6f} ({trend_pct:.2%})

50% weighted contribution
= 50 × {trend_pct:.6f}
= {trend_points:.4f} OG Score points"""
                )

                st.markdown("#### Opponent-volume side")
                st.code(
                    f"""Opponent = {opponent}
Opponent RB Carries/Game = {opp_carries:.4f}

Midpoint percentile rank:
Defense Rank       = {defense_rank:.2f} out of {n}
Defense Percentile = (Defense Rank - 0.5) / N
                   = ({defense_rank:.2f} - 0.5) / {n}
                   = {defense_pct:.6f} ({defense_pct:.2%})

50% weighted contribution
= 50 × {defense_pct:.6f}
= {defense_points:.4f} OG Score points"""
                )

                st.markdown("#### Final OG Score")
                st.code(
                    f"""OG Score
= 100 × (0.50 × Trend Percentile + 0.50 × Defense Percentile)

= 100 × (0.50 × {trend_pct:.6f} + 0.50 × {defense_pct:.6f})
= {trend_points:.4f} + {defense_points:.4f}
= {rebuilt_score:.4f}"""
                )

                delta = rebuilt_score - score
                if abs(delta) < 1e-8:
                    st.success(f"Reconstructed score matches stored OG Score: {score:.4f}")
                else:
                    st.warning(
                        f"Reconstructed score {rebuilt_score:.4f} differs from stored score "
                        f"{score:.4f} by {delta:+.6f}."
                    )


live_tables()


with tab3:
    results = read_csv(APP / "results_history.csv")

    if not results.empty:
        st.subheader("Stored results")
        st.dataframe(
            results.sort_values(
                ["season", "week", "og_rank"],
                na_position="last",
                ascending=[False, False, True],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Stored weekly rankings")
    ranking_history = read_csv(APP / "rankings_history.csv")

    st.dataframe(
        ranking_history.sort_values(
            ["season", "week", "rank"],
            ascending=[False, False, True],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download line history",
        (APP / "historical_lines.csv").read_bytes(),
        "historical_lines.csv",
        "text/csv",
    )


with tab4:
    st.header("Methodology")

    st.markdown(
        """
### 1. Live rushing-attempt line
**Primary source:** PrizePicks standard `player_rush_attempts` market supplied by **The Odds API**.

The app requests the NFL `player_rush_attempts` market event-by-event from:
- PrizePicks
- DraftKings
- FanDuel
- BetMGM
- BetRivers

**Selected Current Line**
1. Use the live **PrizePicks standard line** when available.
2. If PrizePicks is unavailable, use the median of the available sportsbook lines.

The live board is cached for 10 minutes. Event-level prop requests are spaced by one
second and HTTP 429 responses are retried using 2-, 4-, 8-, and 12-second delays.

### 2. Previous-week line
The previous-week line is read from `historical_lines.csv`.

Each successful refresh stores the latest line seen for the current week. When the model
advances to the next week, that stored line becomes the prior-week comparison.

### 3. Market movement
`Line Move = Current Week Line − Previous Week Line`

Buckets:
- **Flat / Down:** move ≤ 0
- **+0.5 to +1.5:** 0 < move ≤ 1.5
- **+2 or more:** move > 1.5

### 4. Backfield usage
Weekly player carries come from **nflverse weekly player stats**.

`RB Share = Player Carries ÷ Total Carries by players classified as RB`

QB and WR rushes are excluded from the backfield denominator.

### 5. Usage trend
`Share Change = Latest Completed Week Share − Previous Week Share`

The tracked back is the team's latest-week RB carry leader; two-week carries break ties.

`Trend Percentile = (average rank − 0.5) ÷ N`

### 6. Opponent rushing opportunity
`Opponent RB Carries/Game = average RB carries allowed per completed game`

Only completed weeks before the target week are used.

### 7. Original matchup score
`OG Score = 50% × Trend Percentile + 50% × Opponent RB-Volume Percentile`

Displayed on a 0–100 scale.

Tier cutoffs:
- A+ ≥ 85
- A ≥ 75
- B+ ≥ 65
- B ≥ 55
- C < 55

### 8. Bet recommendation ordering
Only the **OG Top 10** are eligible.

Among Top-10 players with a current line and an upcoming game:
1. Flat / Down
2. +0.5 to +1.5
3. +2 or more
4. Within each bucket, preserve OG matchup rank

### 9. Actual results
After games finish, actual player carries come from **nflverse weekly player stats**.

Stored result fields include actual carries, final stored line, Over/Under/Push,
OG rank/score, market bucket, and bet rank.

### 10. Automatic week rollover
On dashboard refresh, the pipeline checks the nflverse schedule.

The model advances only when:
1. every regular-season game in the current week has a final score, and
2. nflverse weekly player stats contain the completed week's games.

It then:
1. stores the completed week's results,
2. grades the stored line,
3. builds the next week's 50/50 matchup ranking,
4. writes it to `rankings_history.csv`,
5. increments `state.json`,
6. begins collecting the new week's live lines.


### 11. Player calculation audit
The **Full Ranking** tab includes an expandable calculation for every player.

Each expander shows:
- previous-week RB share
- latest-week RB share
- share change
- usage-trend average rank
- usage-trend midpoint percentile
- opponent RB carries/game
- defense-volume average rank
- defense-volume midpoint percentile
- each 50% weighted contribution
- reconstructed OG Score

This lets the stored OG Score be independently rebuilt from its intermediate values.

### Data sources
- **The Odds API:** PrizePicks + sportsbook rushing-attempt lines
- **nflverse schedules:** game completion / rollover check
- **nflverse weekly player stats:** carries and grading
- **Local CSV history:** prior lines, rankings, and results
"""
    )

    st.info(
        "Week 4 is seeded from the original project workbook. "
        "Week 5 onward is generated automatically with the same 50/50 methodology."
    )

st.caption(
    "Local CSV history persists on your Mac. Live API data is cached for 10 minutes."
)
