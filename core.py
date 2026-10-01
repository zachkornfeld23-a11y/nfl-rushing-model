
from __future__ import annotations
import json, os, re, statistics, unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo
import datetime as dt
import pandas as pd
import requests

NFL_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
NFL_SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

TEAM_NAMES = {
"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
"CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
"DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
"HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
"LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","MIA":"Miami Dolphins",
"MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
"NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks",
"SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WSH":"Washington Commanders"
}
NAME_TO_TEAM = {v:k for k,v in TEAM_NAMES.items()}
SUFFIXES={"jr","sr","ii","iii","iv","v"}

def norm(name):
    x=unicodedata.normalize("NFKD",str(name))
    x="".join(c for c in x if not unicodedata.combining(c)).lower().replace("’","'")
    x=re.sub(r"[^a-z0-9\s]"," ",x)
    return " ".join(p for p in x.split() if p not in SUFFIXES)

def read_state(path):
    return json.loads(Path(path).read_text())

def write_state(path, state):
    Path(path).write_text(json.dumps(state,indent=2))

def read_csv(path, cols=None):
    p=Path(path)
    if not p.exists():
        return pd.DataFrame(columns=cols or [])
    return pd.read_csv(p)

def load_schedule(season):
    df=pd.read_csv(NFL_SCHEDULE_URL, low_memory=False)
    return df[(df["season"]==season) & (df["game_type"]=="REG")].copy()

def load_player_stats(season):
    df=pd.read_csv(NFL_STATS_URL, low_memory=False)
    return df[(df["season"]==season) & (df["season_type"]=="REG")].copy()

def week_complete(schedule, week):
    g=schedule[schedule["week"]==week]
    return (not g.empty) and g["home_score"].notna().all() and g["away_score"].notna().all()

def stats_ready(stats, schedule, week):
    g=schedule[schedule["week"]==week]
    s=stats[stats["week"]==week]
    if g.empty or s.empty: return False
    if "game_id" in s.columns:
        return s["game_id"].nunique() >= len(g)
    return s["team"].nunique() >= min(32, 2*len(g))

def midpoint_pct(series):
    ranks=series.rank(method="average", ascending=True)
    n=len(series)
    return (ranks-0.5)/n

def generate_rankings(stats, schedule, season, target_week):
    latest=target_week-1
    prior=target_week-2
    use=stats[stats["week"].between(1,latest)].copy()
    if "position_group" in use.columns:
        rb=use[use["position_group"].astype(str).str.upper()=="RB"].copy()
    else:
        rb=use[use["position"].astype(str).str.upper()=="RB"].copy()
    if rb.empty: raise RuntimeError("No RB rows found in nflverse player stats.")
    if "carries" not in rb.columns: raise RuntimeError("nflverse player stats missing carries column.")

    team_week=rb.groupby(["team","week"],as_index=False)["carries"].sum().rename(columns={"carries":"team_rb_carries"})
    rb=rb.merge(team_week,on=["team","week"],how="left")
    rb["share"]=rb["carries"]/rb["team_rb_carries"].replace(0,pd.NA)

    # Track each team's latest-week carry leader; two-week carries break ties.
    latest_rows=rb[rb["week"]==latest].copy()
    two=rb[rb["week"].isin([prior,latest])].groupby(["team","player_display_name"],as_index=False)["carries"].sum().rename(columns={"carries":"two_week_carries"})
    latest_rows=latest_rows.merge(two,on=["team","player_display_name"],how="left")
    latest_rows=latest_rows.sort_values(["team","carries","two_week_carries"],ascending=[True,False,False])
    leaders=latest_rows.groupby("team",as_index=False).first()

    shares=rb.pivot_table(index=["team","player_display_name"],columns="week",values="share",aggfunc="first").reset_index()
    leaders=leaders[["team","player_display_name","carries"]].merge(shares,on=["team","player_display_name"],how="left")
    leaders["prior_share"]=leaders.get(prior,0).fillna(0) if prior in leaders.columns else 0.0
    leaders["latest_share"]=leaders.get(latest,0).fillna(0) if latest in leaders.columns else 0.0
    leaders["share_change"]=leaders["latest_share"]-leaders["prior_share"]
    leaders["share_change_rank"]=leaders["share_change"].rank(method="average", ascending=True)
    leaders["trend_pct"]=midpoint_pct(leaders["share_change"])

    # RB carries allowed per game by defense through latest completed week.
    allowed=rb.groupby(["opponent_team","week"],as_index=False)["carries"].sum()
    defense=allowed.groupby("opponent_team",as_index=False)["carries"].mean().rename(columns={"opponent_team":"opponent","carries":"opp_rb_carries_pg"})
    defense["defense_rank"]=defense["opp_rb_carries_pg"].rank(method="average", ascending=True)
    defense["def_pct"]=midpoint_pct(defense["opp_rb_carries_pg"])

    opp={}
    games=schedule[schedule["week"]==target_week]
    for _,g in games.iterrows():
        opp[g["home_team"]]=g["away_team"]; opp[g["away_team"]]=g["home_team"]

    out=[]
    for _,r in leaders.iterrows():
        team=r["team"]; o=opp.get(team,"BYE")
        if o=="BYE":
            defv=None; defp=None; score=-1
        else:
            d=defense[defense["opponent"]==o]
            defv=float(d["opp_rb_carries_pg"].iloc[0]) if not d.empty else None
            defp=float(d["def_pct"].iloc[0]) if not d.empty else 0.5
            score=100*(0.5*float(r["trend_pct"])+0.5*defp)
        out.append({
            "season":season,
            "week":target_week,
            "player":r["player_display_name"],
            "team":team,
            "opponent":o,
            "og_score":score,
            "prior_share":float(r["prior_share"]),
            "latest_share":float(r["latest_share"]),
            "share_change":float(r["share_change"]),
            "trend_rank":float(r["share_change_rank"]) if "share_change_rank" in r and pd.notna(r["share_change_rank"]) else None,
            "trend_percentile":float(r["trend_pct"]) if pd.notna(r["trend_pct"]) else None,
            "opp_rb_carries_pg":defv,
            "defense_rank":float(d["defense_rank"].iloc[0]) if o!="BYE" and not d.empty and "defense_rank" in d.columns else None,
            "defense_percentile":defp,
            "rank_population":int(len(leaders)),
            "trend_contribution":50*float(r["trend_pct"]) if pd.notna(r["trend_pct"]) else None,
            "defense_contribution":50*defp if o!="BYE" else None,
            "source":"Auto-generated from nflverse weekly player stats"
        })
    df=pd.DataFrame(out).sort_values(["og_score","latest_share"],ascending=[False,False]).reset_index(drop=True)
    df["rank"]=range(1,len(df)+1)
    def tier(x,opp):
        if opp=="BYE": return "BYE"
        if x>=85:return "A+"
        if x>=75:return "A"
        if x>=65:return "B+"
        if x>=55:return "B"
        return "C"
    df["tier"]=[tier(x,o) for x,o in zip(df["og_score"],df["opponent"])]
    return df[[
        "season","week","rank","player","team","opponent","og_score","tier",
        "prior_share","latest_share","share_change",
        "trend_rank","trend_percentile",
        "opp_rb_carries_pg","defense_rank","defense_percentile",
        "rank_population","trend_contribution","defense_contribution",
        "source"
    ]]

def upsert_rankings(path, df):
    old=read_csv(path)
    if not old.empty:
        old=old[~((old["season"]==int(df["season"].iloc[0]))&(old["week"]==int(df["week"].iloc[0])))]
    pd.concat([old,df],ignore_index=True).to_csv(path,index=False)

def week_rankings(path,season,week):
    df=read_csv(path)
    return df[(df["season"]==season)&(df["week"]==week)].sort_values("rank").copy()

def history_lines(path,season,week):
    df=read_csv(path)
    if df.empty:return {}
    s=df[(df["season"]==season)&(df["week"]==week)]
    return {norm(r["player"]):float(r["line"]) for _,r in s.iterrows() if pd.notna(r["line"])}

def upsert_lines(path, season, week, ranking_df, live_df):
    old=read_csv(path)
    rows=[]
    now=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for _,r in ranking_df.iterrows():
        m=live_df[live_df["player_key"]==norm(r["player"])]
        if m.empty: continue
        x=m.iloc[0]
        rows.append({
            "season":season,"week":week,"player":r["player"],"team":r["team"],
            "line":x["selected_line"],"line_source":x["line_source"],
            "prizepicks_line":x.get("prizepicks"),"sportsbook_consensus":x.get("sportsbook_consensus"),
            "draftkings":x.get("draftkings"),"fanduel":x.get("fanduel"),"betmgm":x.get("betmgm"),"betrivers":x.get("betrivers"),
            "saved_at_utc":now
        })
    if rows:
        incoming=pd.DataFrame(rows)
        keys={(season,week,norm(x["player"])) for x in rows}
        if not old.empty:
            keep=[]
            for _,r in old.iterrows():
                keep.append((int(r["season"]),int(r["week"]),norm(r["player"])) not in keys)
            old=old.loc[keep]
        pd.concat([old,incoming],ignore_index=True).to_csv(path,index=False)

def game_status_map(schedule, week):
    now=dt.datetime.now(dt.timezone.utc)
    out={}
    for _,g in schedule[schedule["week"]==week].iterrows():
        final=pd.notna(g["home_score"]) and pd.notna(g["away_score"])
        if final: status="Final"
        else:
            try:
                local=dt.datetime.fromisoformat(f'{g["gameday"]}T{g["gametime"]}').replace(tzinfo=ZoneInfo("America/New_York"))
                status="Started" if now>=local.astimezone(dt.timezone.utc) else "Upcoming"
            except Exception: status="Upcoming"
        out[g["home_team"]]=status; out[g["away_team"]]=status
    return out

def build_board(rankings, current_map, prior_map, status_map):
    rows=[]
    for _,r in rankings.iterrows():
        k=norm(r["player"]); cur=current_map.get(k); prior=prior_map.get(k)
        move=None if cur is None or prior is None else cur-prior
        if cur is None: bucket="No current line"
        elif prior is None: bucket="No prior line"
        elif move<=0: bucket="Flat / Down"
        elif move<=1.5: bucket="+0.5 to +1.5"
        else: bucket="+2 or more"
        rows.append({
            "Rank":int(r["rank"]),"Player":r["player"],"Team":r["team"],"Opponent":r["opponent"],
            "OG Score":float(r["og_score"]),"Tier":r["tier"],"Last Week Line":prior,"Current Line":cur,
            "Line Move":move,"Market Move":bucket,"Game Status":status_map.get(r["team"],"BYE" if r["opponent"]=="BYE" else "Upcoming")
        })
    return pd.DataFrame(rows)

def recommendations(board):
    order={"Flat / Down":0,"+0.5 to +1.5":1,"+2 or more":2,"No prior line":3,"No current line":4}
    x=board[(board["Rank"]<=10)&board["Current Line"].notna()&(board["Game Status"]=="Upcoming")].copy()
    x["_o"]=x["Market Move"].map(order).fillna(9)
    x=x.sort_values(["_o","Rank"]).drop(columns="_o").reset_index(drop=True)
    x.insert(0,"Bet Rank",range(1,len(x)+1))
    return x

def archive_results(results_path, rankings_path, lines_path, stats, season, week):
    existing=read_csv(results_path)
    if not existing.empty and ((existing["season"]==season)&(existing["week"]==week)&existing["og_rank"].notna()).any():
        return
    ranks=week_rankings(rankings_path,season,week)
    prior=history_lines(lines_path,season,week-1)
    current=history_lines(lines_path,season,week)
    # recommendation snapshot from final stored lines
    board=build_board(ranks,current,prior,{t:"Final" for t in ranks["team"]})
    rec=recommendations(board.assign(**{"Game Status":"Upcoming"}))
    betrank={norm(r["Player"]):int(r["Bet Rank"]) for _,r in rec.iterrows()}
    bucket={norm(r["Player"]):r["Market Move"] for _,r in board.iterrows()}
    s=stats[stats["week"]==week]
    rows=[]
    for _,r in ranks.iterrows():
        k=norm(r["player"])
        m=s[s["player_display_name"].map(norm)==k]
        actual=float(m["carries"].sum()) if not m.empty else None
        line=current.get(k)
        rows.append({
            "season":season,"week":week,"player":r["player"],"team":r["team"],"actual_carries":actual,
            "final_line":line,"actual_minus_line":None if actual is None or line is None else actual-line,
            "over_result":"" if actual is None or line is None else ("Over" if actual>line else "Under" if actual<line else "Push"),
            "og_rank":int(r["rank"]),"og_score":float(r["og_score"]),"bet_rank":betrank.get(k),
            "market_bucket":bucket.get(k,""),"source":"nflverse weekly player stats"
        })
    if not existing.empty:
        existing=existing[~((existing["season"]==season)&(existing["week"]==week))]
    pd.concat([existing,pd.DataFrame(rows)],ignore_index=True).to_csv(results_path,index=False)
