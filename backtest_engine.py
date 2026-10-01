"""Historical rushing-attempt research. No future weeks enter feature generation."""
from __future__ import annotations
from dataclasses import dataclass
import io, json, math, hashlib, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from core import generate_rankings, norm, NAME_TO_TEAM, NFL_STATS_URL, NFL_SCHEDULE_URL
from odds_source import _request, BASE, SPORT, OddsApiError

VERSION = 1
BOOKS = ['prizepicks', 'draftkings', 'fanduel', 'betmgm', 'betrivers']
ALIASES = {'LA': 'LAR', 'WAS': 'WSH', 'JAC': 'JAX'}
LINE_COLUMNS = ['season','week','game_id','event_id','player_key','player','book','line','over_price','under_price','snapshot','requested_at','market_updated']


def normalize(df):
    df = df.copy()
    for col in ['team','opponent_team','home_team','away_team']:
        if col in df: df[col] = df[col].replace(ALIASES)
    return df


def iso(ts):
    return pd.Timestamp(ts).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(text); tmp.replace(path)


def get_football(season, cache_dir):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, url in [('schedule', NFL_SCHEDULE_URL), (f'stats_{season}', NFL_STATS_URL.format(season=season))]:
        p = cache_dir / f'{name}.csv'
        if not p.exists() or time.time() - p.stat().st_mtime > 21600:
            try:
                r = requests.get(url, timeout=60)
                if not r.ok: raise RuntimeError(f'{name}: data source returned HTTP {r.status_code}')
                frame = pd.read_csv(io.StringIO(r.text), low_memory=False)
                if 'season' not in frame: raise RuntimeError(f'{name}: invalid football data')
                atomic_text(p, r.text)
            except Exception:
                if not p.exists(): raise RuntimeError(f'Could not load {name}. Try again when nflverse data is available.') from None
        frames.append(normalize(pd.read_csv(p, low_memory=False)))
    schedule, stats = frames
    schedule = schedule[(schedule.season == season) & (schedule.game_type == 'REG')].copy()
    stats = stats[(stats.season == season) & (stats.season_type == 'REG')].copy()
    schedule['kickoff'] = pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str), errors='coerce').dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    return schedule, stats


def completed_weeks(schedule, stats):
    out=[]
    for week, games in schedule.groupby('week'):
        s=stats[stats.week==week]
        ready = set(games.game_id).issubset(set(s.game_id)) if 'game_id' in s else set(games.home_team).union(games.away_team).issubset(set(s.team))
        if games[['home_score','away_score','kickoff']].notna().all().all() and ready:
            out.append(int(week))
    return sorted(out)


def features(stats, schedule, season, week):
    # The exact current model is retained for comparison, including its latest-calendar-week leader rule.
    stats=normalize(stats); schedule=normalize(schedule)
    hist=stats[(stats.season==season)&(stats.week<week)&(stats.season_type=='REG')].copy()
    if hist.empty: return pd.DataFrame()
    original=generate_rankings(hist, schedule, season, week)
    original=original[original.opponent!='BYE'].copy()
    original['player_key']=original.player.map(norm)
    og={(r.team,r.player_key):(r['rank'],r.og_score) for _,r in original.iterrows()}
    pos='position_group' if 'position_group' in hist else 'position'
    rb=hist[hist[pos].astype(str).str.upper()=='RB'].copy()
    rb['player_key']=rb.player_display_name.map(norm)
    teamweeks=rb.groupby(['team','week']).carries.sum()
    defense=rb.groupby(['opponent_team','week']).carries.sum().groupby('opponent_team').mean()
    # Match original defense percentiles over all defenses, rather than only this week's opponents.
    defense_pct=(defense.rank(method='average')-.5)/len(defense)
    games=schedule[schedule.week==week]
    rows=[]
    for _,g in games.iterrows():
        for team, opponent in [(g.home_team,g.away_team),(g.away_team,g.home_team)]:
            t=rb[rb.team==team]
            weeks=sorted(t.week.unique())[-3:]
            if len(weeks)<2 or opponent not in defense: continue
            recent=t[t.week.isin(weeks)]
            last=t[t.week==weeks[-1]].copy()
            totals=recent.groupby('player_key').carries.sum()
            last['recent_carries']=last.player_key.map(totals)
            last=last.sort_values(['carries','recent_carries','player_key'],ascending=[False,False,True])
            if last.empty: continue
            leader=last.iloc[0]; key=leader.player_key
            carries=[float(t[(t.week==w)&(t.player_key==key)].carries.sum()) for w in weeks]
            shares=[c/float(teamweeks.loc[(team,w)]) if float(teamweeks.loc[(team,w)])>0 else 0. for c,w in zip(carries,weeks)]
            jump=shares[-1]-shares[-2]
            slope=(shares[-1]-shares[0])/(len(shares)-1)
            trend3=.5*jump+.5*slope
            rank, score=og.get((team,key),(np.nan,np.nan))
            team_volume=float(np.mean([teamweeks.loc[(team,w)] for w in weeks]))
            weighted_share=float(np.average(shares,weights=np.arange(1,len(shares)+1)))
            projected=weighted_share*(.5*team_volume+.5*float(defense.loc[opponent]))
            rows.append(dict(season=season,week=week,game_id=g.game_id,player=leader.player_display_name,
                player_key=key,player_id=leader.get('player_id',''),team=team,opponent=opponent,
                original_rank=rank,original_score=score,latest_share=shares[-1],share_jump=jump,
                trend3=trend3,history_games=len(weeks),defense_volume=float(defense.loc[opponent]),
                defense_pct=float(defense_pct.loc[opponent]),projected_carries=projected,
                recent_carries=float(np.average(carries,weights=np.arange(1,len(carries)+1)))))
    df=pd.DataFrame(rows)
    if df.empty: return df
    for col in ['share_jump','trend3']:
        df[col+'_pct']=(df[col].rank(method='average')-.5)/len(df)
    return df


class HistoricalClient:
    """Disk cache excludes API keys. Budget bounds new request estimates; reruns reuse responses."""
    def __init__(self,key,cache_dir,budget=4000):
        self.key=key; self.root=Path(cache_dir)/'odds'; self.root.mkdir(parents=True,exist_ok=True)
        self.budget=int(budget); self.spent=0; self.reserved=0; self.remaining=None; self.hits=0

    def fetch(self,path,params,cost):
        digest=hashlib.sha256(json.dumps([path,params],sort_keys=True).encode()).hexdigest()
        p=self.root/(digest+'.json')
        if p.exists():
            self.hits+=1; return json.loads(p.read_text())
        if self.reserved+cost>self.budget: raise RuntimeError('Credit budget reached. Cached progress is safe; raise the budget and resume.')
        if not self.key: raise RuntimeError('Set ODDS_API_KEY in Streamlit Secrets to download historical lines.')
        self.reserved+=cost
        time.sleep(.25)
        try:
            data,meta=_request(BASE+path,dict(params,apiKey=self.key))
        except requests.RequestException:
            raise RuntimeError('The Odds API connection failed. Cached progress is safe. Retry to resume.') from None
        except OddsApiError as exc:
            # The provider error must never expose a credential.
            raise RuntimeError(str(exc).replace(self.key,'[redacted]')) from None
        self.spent+=int(meta.get('last') or cost); self.remaining=meta.get('remaining')
        atomic_text(p,json.dumps(data))
        return data


def parse_event(payload, game, cutoff, season, week):
    snapshot=pd.to_datetime(payload.get('timestamp'),utc=True,errors='coerce')
    if pd.isna(snapshot) or snapshot>cutoff: raise RuntimeError('Rejected a historical response dated after the decision cutoff.')
    event=payload.get('data',{})
    kickoff=pd.to_datetime(event.get('commence_time'),utc=True,errors='coerce')
    if pd.isna(kickoff) or cutoff>=kickoff: return []
    if NAME_TO_TEAM.get(event.get('home_team'))!=game.home_team or NAME_TO_TEAM.get(event.get('away_team'))!=game.away_team:
        raise RuntimeError('Historical event team mismatch.')
    out=[]
    for book in event.get('bookmakers',[]):
        for market in book.get('markets',[]):
            if market.get('key')!='player_rush_attempts': continue
            updated=market.get('last_update') or book.get('last_update')
            ts=pd.to_datetime(updated,utc=True,errors='coerce')
            if pd.notna(ts) and ts>cutoff: continue
            grouped={}
            for x in market.get('outcomes',[]):
                side=str(x.get('name','')).lower(); player=x.get('description')
                if side not in ('over','under') or not player or x.get('point') is None: continue
                pair=(norm(player),float(x['point']))
                grouped.setdefault(pair,{'player':player})[side]=x.get('price')
            # Reject ambiguous multiple standard lines instead of accidentally taking an alternate.
            counts={k:sum(1 for kk,_ in grouped if kk==k) for k,_ in grouped}
            for (key,line),prices in grouped.items():
                if counts[key]!=1 or 'over' not in prices or 'under' not in prices: continue
                out.append(dict(season=season,week=week,game_id=game.game_id,event_id=event.get('id'),
                    player_key=key,player=prices['player'],book=book['key'],line=line,
                    over_price=prices.get('over'),under_price=prices.get('under'),snapshot=iso(snapshot),
                    requested_at=iso(cutoff),market_updated=updated))
    return out


def collect_lines(client,schedule,season,weeks,minutes=60,progress=None):
    rows=[]; audit=[]
    total=sum(len(schedule[schedule.week==w]) for w in weeks); done=0
    for week in sorted(weeks):
        games=schedule[schedule.week==week].sort_values('kickoff')
        if games.empty or games.kickoff.isna().any(): continue
        # Same weekly information set for every player. No Sunday lines used to decide Thursday picks.
        cutoff=games.kickoff.min()-pd.Timedelta(minutes=minutes)
        payload=client.fetch(f'/historical/sports/{SPORT}/events',{'date':iso(cutoff),'dateFormat':'iso'},1)
        snap=pd.to_datetime(payload.get('timestamp'),utc=True,errors='coerce')
        if pd.isna(snap) or snap>cutoff: raise RuntimeError('Invalid historical events timestamp.')
        events=payload.get('data',[])
        for _,g in games.iterrows():
            candidates=[e for e in events if NAME_TO_TEAM.get(e.get('home_team'))==g.home_team and NAME_TO_TEAM.get(e.get('away_team'))==g.away_team and abs((pd.Timestamp(e['commence_time'])-g.kickoff).total_seconds())<86400]
            status='No matching historical event'; result=[]
            if len(candidates)==1:
                e=candidates[0]
                data=client.fetch(f'/historical/sports/{SPORT}/events/{e["id"]}/odds',{'date':iso(cutoff),'dateFormat':'iso','oddsFormat':'american','markets':'player_rush_attempts','bookmakers':','.join(BOOKS)},10)
                result=parse_event(data,g,cutoff,season,week)
                rows.extend(result); status='Available' if result else 'No standard props at cutoff'
            elif len(candidates)>1: status='Ambiguous event match (excluded)'
            audit.append(dict(season=season,week=week,game_id=g.game_id,cutoff=iso(cutoff),status=status,quotes=len(result)))
            done+=1
            if progress: progress(done/max(total,1),f'{season} Week {week}: {done}/{total} games; {client.spent} new credits')
    return pd.DataFrame(rows,columns=LINE_COLUMNS),pd.DataFrame(audit)


def select_lines(lines,source):
    if lines.empty: return lines.copy()
    keys=['season','week','game_id','player_key']
    x=lines.copy()
    if source!='Priority: PrizePicks, DraftKings, FanDuel, BetMGM, BetRivers':
        x=x[x.book==source]
    x['priority']=x.book.map({b:i for i,b in enumerate(BOOKS)})
    return x.sort_values(keys+['priority']).drop_duplicates(keys).drop(columns='priority')


def add_sma_projection(feature_rows, stats_by_season):
    """Upgrade old saved features using cached stats only. Equal-weight prior 3 team games."""
    teams = {}
    for season, stats in stats_by_season.items():
        s = normalize(stats)
        s = s[(s.season == int(season)) & (s.season_type == 'REG')].copy()
        position = 'position_group' if 'position_group' in s else 'position'
        s = s[s[position].astype(str).str.upper() == 'RB']
        for team, group in s.groupby('team'):
            totals = group.groupby('week').carries.sum(min_count=1)
            player_totals = group.groupby(['week', 'player_id']).carries.sum(min_count=1)
            teams[(int(season), team)] = (totals, player_totals)
    rows = []
    for _, r in feature_rows.iterrows():
        totals, players = teams.get((int(r.season), r.team), (pd.Series(dtype=float), pd.Series(dtype=float)))
        weeks = sorted(w for w in totals.index if w < r.week)[-3:]
        volumes = [float(totals.loc[w]) for w in weeks]
        carries = [float(players.get((w, r.player_id), 0.)) for w in weeks]
        shares = [c/v if pd.notna(v) and v > 0 else np.nan for c, v in zip(carries, volumes)]
        valid = len(weeks) == 3 and all(pd.notna(x) for x in shares)
        mean_share = float(np.mean(shares)) if valid else np.nan
        mean_volume = float(np.mean(volumes)) if valid else np.nan
        item = dict(sma_history_games=len(weeks), sma_weeks=', '.join(str(int(w)) for w in weeks),
                    sma_share=mean_share, sma_team_rb_carries=mean_volume,
                    sma_projected_carries=mean_share*mean_volume,
                    sma_recent_carries=float(np.mean(carries)) if valid else np.nan)
        for i in range(3):
            item[f'sma_share_game_{i+1}'] = shares[i] if i < len(shares) else np.nan
        rows.append(item)
    out = feature_rows.reset_index(drop=True).copy()
    calculated = pd.DataFrame(rows)
    for col in calculated:
        out[col] = calculated[col]
    return out


def build_dataset(feature_rows,lines,stats_by_season,source):
    feature_rows = add_sma_projection(feature_rows, stats_by_season)
    selected=select_lines(lines,source)
    fields=['season','week','game_id','player_key','book','line','over_price','under_price','snapshot','requested_at']
    df=feature_rows.merge(selected.reindex(columns=fields),how='left',on=['season','week','game_id','player_key'],validate='one_to_one')
    # Previous CALENDAR week, SAME bookmaker and player. Byes are not silently assigned zero movement.
    lookup={(int(r.season),int(r.week),r.player_key,r.book):float(r.line) for _,r in lines.iterrows()}
    df['prior_line']=[lookup.get((int(r.season),int(r.week)-1,r.player_key,r.book),np.nan) for _,r in df.iterrows()]
    df['line_move']=df.line-df.prior_line
    df['movement_bucket']=np.select([df.line_move<=0,df.line_move.between(0,1.5,inclusive='right'),df.line_move>1.5],[0,1,2],default=3)
    actual=[]; grading=[]
    for _,r in df.iterrows():
        s=stats_by_season[int(r.season)]
        match=s[(s.week==r.week)&(s.team==r.team)&(s.player_id==r.player_id)]
        if len(match)==1 and pd.notna(match.iloc[0].carries):
            actual.append(float(match.iloc[0].carries)); grading.append('Official stat row')
        else:
            actual.append(np.nan); grading.append('Missing/ambiguous participation; ungraded')
    df['actual_carries']=actual; df['grading_status']=grading
    df['projection_edge']=df.projected_carries-df.line
    df['sma_edge']=df.get('sma_projected_carries', np.nan)-df.line
    return df


@dataclass(frozen=True)
class Strategy:
    name:str
    method:str
    top_n:int
    side:str='Over'
    defense_weight:float=.5
    trend:str='share_jump_pct'
    flat_only:bool=False


def strategies():
    return [
        Strategy('Original 50/50 · Top 5 overs', 'original', 5),
        Strategy('Original 50/50 · Top 10 overs', 'original', 10),
        Strategy('OG Top 10 → line movement · Top 5 overs', 'movement', 5),
        Strategy('OG Top 10 → flat/down only · Up to 5 overs', 'movement', 5, flat_only=True),
        Strategy('75% defense / 25% usage · Top 5 overs', 'weighted', 5,
                 defense_weight=.75, trend='share_jump_pct'),
        Strategy('50/50 with 3-game trend · Top 5 overs', 'weighted', 5,
                 defense_weight=.5, trend='trend3_pct'),
        Strategy('3-game average share × team volume · Positive-edge overs', 'sma', 32),
    ]


def pick(df,strategy):
    x=df.copy()
    if strategy.method in ('original','movement'):
        x=x[x.original_rank.notna() & (x.original_rank<=10)]
        if strategy.method=='original': x=x[x.original_rank<=strategy.top_n]
    x=x[x.line.notna()]
    if strategy.flat_only: x=x[x.line_move<=0]
    if x.empty: return x
    if strategy.method=='movement': cols=['movement_bucket','original_rank','player_key']; ascending=[True,True,True]
    elif strategy.method=='original': cols=['original_rank','player_key']; ascending=[True,True]
    elif strategy.method=='weighted':
        x['selection_score']=100*(strategy.defense_weight*x.defense_pct+(1-strategy.defense_weight)*x[strategy.trend])
        cols=['selection_score','player_key']; ascending=[strategy.side=='Under',True]
    elif strategy.method=='edge':
        x=x[x.projection_edge>0] if strategy.side=='Over' else x[x.projection_edge<0]
        cols=['projection_edge','player_key']; ascending=[strategy.side=='Under',True]
    elif strategy.method=='sma':
        x=x[x.sma_edge>0]
        cols=['sma_edge','player_key']; ascending=[False,True]
    else: cols=['player_key']; ascending=[True]
    # Selection never filters on whether the future outcome is known.
    x=x.sort_values(cols,ascending=ascending).head(strategy.top_n).copy()
    x['side']=strategy.side; x['strategy']=strategy.name
    x['bet_rank']=range(1,len(x)+1)
    return x


def grade(bets,offset=0):
    x=bets.copy()
    if x.empty: return x
    x['bet_line']=x.line+np.where(x.side=='Over',-offset,offset)
    diff=(x.actual_carries-x.bet_line)*np.where(x.side=='Over',1,-1)
    x['result']=np.select([diff>0,diff<0,diff==0],['Win','Loss','Push'],default='Ungraded')
    x['units']=np.select([diff>0,diff<0,diff==0],[1.,-1.,0.],default=np.nan)
    return x


def summarize(bets):
    if bets.empty: return dict(selected=0,graded=0,wins=0,losses=0,pushes=0,ungraded=0,units=0.,roi=np.nan,win_rate=np.nan,win_low=np.nan,win_high=np.nan,max_weekly_drawdown=0.)
    counts=bets.result.value_counts(); w=int(counts.get('Win',0)); l=int(counts.get('Loss',0)); p=int(counts.get('Push',0)); n=w+l
    units=float(bets.units.sum()); graded=n+p; rate=w/n if n else np.nan
    if n:
        z=1.96; mid=(rate+z*z/(2*n))/(1+z*z/n); half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
        low,high=mid-half,mid+half
    else: low=high=np.nan
    weekly=bets.groupby(['season','week']).units.sum().sort_index().cumsum()
    peak=weekly.cummax().clip(lower=0)
    return dict(selected=len(bets),graded=graded,wins=w,losses=l,pushes=p,ungraded=len(bets)-graded,units=units,roi=units/graded if graded else np.nan,win_rate=rate,win_low=low,win_high=high,max_weekly_drawdown=float((peak-weekly).max()) if len(weekly) else 0.)


def evaluate(df,offset=0):
    rows=[]; all_bets=[]
    for strategy in strategies():
        picked=[pick(g,strategy) for _,g in df.groupby(['season','week'],sort=True)]
        bets=grade(pd.concat(picked,ignore_index=True),offset)
        rows.append(dict(strategy=strategy.name,**summarize(bets)))
        if not bets.empty: all_bets.append(bets)
    return pd.DataFrame(rows),pd.concat(all_bets,ignore_index=True) if all_bets else pd.DataFrame()


def leaderboard(bets,min_bets=20):
    rows=[dict(strategy=name,**summarize(g)) for name,g in bets.groupby('strategy')] if not bets.empty else []
    if not rows: return pd.DataFrame()
    table=pd.DataFrame(rows); table['enough_bets']=table.graded>=min_bets
    return table.sort_values(['enough_bets','units','roi','graded','strategy'],ascending=[False,False,False,False,True])


def walk_forward(bets,min_bets=20):
    if bets.empty: return bets.copy()
    selected=[]
    for season,week in sorted(set(zip(bets.season,bets.week))):
        past=bets[(bets.season<season)|((bets.season==season)&(bets.week<week))]
        table=leaderboard(past,min_bets)
        if table.empty or not table.enough_bets.any(): continue
        winner=table[table.enough_bets].iloc[0].strategy
        current=bets[(bets.season==season)&(bets.week==week)&(bets.strategy==winner)].copy()
        current['selected_using']='Completed earlier weeks only'
        selected.append(current)
    return pd.concat(selected,ignore_index=True) if selected else bets.iloc[:0].copy()


def export_cache(root):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in Path(root).rglob('*'):
            if p.is_file() and p.suffix in ('.csv','.json'): z.write(p,p.relative_to(root))
    return out.getvalue()


def restore_cache(blob,root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if sum(x.file_size for x in z.infolist())>150_000_000: raise ValueError('Backup is too large.')
        for item in z.infolist():
            p=(root/item.filename).resolve()
            if not p.is_relative_to(root) or p.suffix not in ('.csv','.json'): raise ValueError('Invalid backup path or file type.')
        for item in z.infolist():
            p=root/item.filename; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(z.read(item))
