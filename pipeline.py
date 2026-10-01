
from pathlib import Path
from core import *
APP=Path(__file__).resolve().parent

def maybe_rollover():
    state=read_state(APP/"state.json")
    season=int(state["season"]); week=int(state["current_week"])
    sched=load_schedule(season)
    if not week_complete(sched,week):
        return state, False, "Week still in progress."
    stats=load_player_stats(season)
    if not stats_ready(stats,sched,week):
        return state, False, "Games are final; waiting for nflverse player stats."
    archive_results(APP/"results_history.csv",APP/"rankings_history.csv",APP/"historical_lines.csv",stats,season,week)
    next_week=week+1
    if week_rankings(APP/"rankings_history.csv",season,next_week).empty:
        nr=generate_rankings(stats,sched,season,next_week)
        upsert_rankings(APP/"rankings_history.csv",nr)
    state["current_week"]=next_week
    write_state(APP/"state.json",state)
    return state, True, f"Archived Week {week} and advanced to Week {next_week}."
