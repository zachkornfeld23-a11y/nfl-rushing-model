from pathlib import Path
import json
import pandas as pd
import streamlit as st
from backtest_engine import (get_football, completed_weeks, features, HistoricalClient, collect_lines,
    build_dataset, evaluate, summarize, leaderboard, walk_forward, export_cache, restore_cache, atomic_text, BOOKS)

PRIORITY='Priority: PrizePicks, DraftKings, FanDuel, BetMGM, BetRivers'


def show_metrics(bets):
    m=summarize(bets)
    a,b,c,d,e=st.columns(5)
    a.metric('Net units',f'{m["units"]:+.0f}')
    b.metric('Win–Loss–Push',f'{m["wins"]}–{m["losses"]}–{m["pushes"]}')
    c.metric('Win rate',f'{m["win_rate"]:.1%}' if pd.notna(m['win_rate']) else '—')
    d.metric('Even-odds ROI',f'{m["roi"]:.1%}' if pd.notna(m['roi']) else '—')
    e.metric('Ungraded',m['ungraded'])
    if m['graded']:
        st.caption(f'95% win-rate interval: {m["win_low"]:.1%}–{m["win_high"]:.1%}. Largest drawdown at weekly closes: {m["max_weekly_drawdown"]:.0f} units. Intervals do not adjust for testing many strategies or correlated bets.')


def table(df):
    st.dataframe(df,use_container_width=True,hide_index=True)


def display_leaderboard(df):
    x=df.copy()
    for c in ['roi','win_rate','win_low','win_high']:
        if c in x: x[c]=x[c].map(lambda v:f'{v:.1%}' if pd.notna(v) else '—')
    table(x)


def render_backtest(api_key, app_dir):
    root=Path(app_dir)/'backtest_cache'; root.mkdir(exist_ok=True)
    st.title('Backtest & strategies')
    st.caption('Real historical standard lines • +1 win / −1 loss / 0 push • one unit per graded bet')
    st.info('First download the historical data below. After that, compare every strategy without further Odds API requests. Results are research comparisons, not a claim of future profitability.')
    with st.expander('Download or resume historical data',expanded=not (root/'features.csv').exists()):
        a,b,c=st.columns(3)
        seasons=a.multiselect('Seasons',[2025,2026],default=[2025])
        start,end=b.slider('Evaluation weeks',min_value=3,max_value=18,value=(4,18))
        minutes=c.selectbox('Snapshot before first kickoff each week',[60,180,1440],format_func=lambda x:f'{x//60} hour(s)',index=0)
        st.caption('Every player in a week uses the same pregame cutoff. No later Sunday line is allowed to influence a Thursday pick. Games with no posted standard line at this cutoff are reported as missing. The previous week is also downloaded for movement comparisons.')
        budget=st.number_input('Maximum new credits per run',min_value=11,max_value=50000,value=4000,step=100)
        st.caption('Estimate: up to 10 credits per game plus 1 event-list credit per week; roughly 2,600–3,000 credits for a full 2025 run including the prior-week lines. Cache hits cost zero. Actual usage is shown while running.')
        if st.button('Download history & build backtest',type='primary',disabled=not seasons):
            client=HistoricalClient(api_key,root,budget)
            progress=st.progress(0,text='Loading football data…')
            fs=[]; ls=[]; audits=[]; all_stats=[]; coverage=[]
            try:
                # Obtain and validate all public football inputs before paying for historical lines.
                inputs=[]
                for season in seasons:
                    sched,stats=get_football(season,root/'football')
                    completed=completed_weeks(sched,stats)
                    weeks=[w for w in completed if start<=w<=end]
                    if not weeks:
                        st.warning(f'{season}: no fully completed weeks with available stats in this range.'); continue
                    for w in weeks:
                        fs.append(features(stats,sched,season,w))
                    inputs.append((season,sched,stats,weeks))
                if not inputs or not fs: raise RuntimeError('No completed weeks are available for this selection.')
                for season,sched,stats,weeks in inputs:
                    download_weeks=sorted(set(weeks+[w-1 for w in weeks if w>1]))
                    lines,audit=collect_lines(client,sched,season,download_weeks,minutes,
                        lambda pct,msg:progress.progress(pct,text=msg))
                    ls.append(lines); audits.append(audit); all_stats.append(stats)
                    coverage.append({'season':season,'weeks':weeks})
                outputs={'features':pd.concat(fs,ignore_index=True),'lines':pd.concat(ls,ignore_index=True),
                         'audit':pd.concat(audits,ignore_index=True),'stats':pd.concat(all_stats,ignore_index=True)}
                for name,df in outputs.items(): atomic_text(root/f'{name}.csv',df.to_csv(index=False))
                manifest={'version':1,'coverage':coverage,'minutes_before_week':minutes,'new_credits':client.spent,
                          'cached_requests':client.hits,'credits_remaining':client.remaining,'built_at':pd.Timestamp.now(tz='UTC').isoformat()}
                atomic_text(root/'manifest.json',json.dumps(manifest))
                progress.progress(1,text='Backtest ready')
                st.success(f'Built historical dataset. {client.spent} new credits; {client.hits} cached responses reused.')
            except Exception as exc:
                progress.empty()
                safe=str(exc)
                if api_key: safe=safe.replace(api_key,'[redacted]')
                st.error(safe)
                st.caption('Successfully downloaded raw responses are cached. Click the same button to resume. Any previously completed dataset below stays available.')
    with st.expander('Backup / restore data'):
        st.caption('Streamlit server files can disappear after a reboot or redeploy. Download a backup after the first run, then restore it here to avoid buying the same history again. Backups contain data only, never the API key.')
        if any(root.rglob('*.json')):
            st.download_button('Download backtest backup',export_cache(root),'rushing_backtest_backup.zip','application/zip')
        upload=st.file_uploader('Restore a saved backtest backup',type=['zip'])
        if upload and st.button('Restore backup'):
            try: restore_cache(upload.getvalue(),root); st.rerun()
            except Exception as exc: st.error(str(exc))
    if not (root/'features.csv').exists():
        st.warning('No historical backtest has been run yet. The strategy rankings will appear after the download completes.')
        return
    try:
        f=pd.read_csv(root/'features.csv'); lines=pd.read_csv(root/'lines.csv'); stats=pd.read_csv(root/'stats.csv',low_memory=False)
        manifest=json.loads((root/'manifest.json').read_text()); audit=pd.read_csv(root/'audit.csv')
    except Exception:
        st.error('Saved dataset is incomplete. Resume the download or restore a complete backup.'); return
    st.caption('Saved run: '+', '.join(f'{x["season"]} weeks {min(x["weeks"])}–{max(x["weeks"])}' for x in manifest['coverage'])+f' • snapshot {manifest["minutes_before_week"]} minutes before first kickoff • built {manifest["built_at"][:10]}')
    a,b,c=st.columns([2,2,1])
    source=a.selectbox('Historical line source',[PRIORITY]+BOOKS)
    scenario=b.selectbox('Line scenario',['Standard posted line','Hypothetical: Over −4'])
    minimum=c.number_input('Minimum graded bets',min_value=1,max_value=500,value=20)
    offset=4 if scenario.startswith('Hypothetical') else 0
    if offset:
        st.warning('Hypothetical clearance test: thresholds are shifted by 4 attempts. These are not verified alternate lines or available even-money bets. Units and ROI below are illustrative; do not compare them with standard-line profitability.')
    dataset=build_dataset(f,lines,{int(y):g for y,g in stats.groupby('season')},source)
    summary,bets=evaluate(dataset,offset)
    if bets.empty:
        st.warning('No matched posted lines for this source. Check coverage below or choose another book.')
        table(audit); return
    st.caption(f'{len(summary)} strategy configurations tested. Missing lines are excluded, never replaced with projected lines. Top picks without an official stat row remain ungraded and are not replaced by lower-ranked players.')
    tabs=st.tabs(['Strategy leaderboard','Honest validation','Weekly & bet detail','Data coverage','How it works','3-game projection'])
    with tabs[0]:
        st.subheader('Best historical results')
        st.caption('Descriptive ranking by total units, then ROI. These winners were selected after seeing the results. Check Honest validation before drawing conclusions.')
        lb=leaderboard(bets,int(minimum))
        display_leaderboard(lb)
        st.download_button('Download strategy results',lb.to_csv(index=False),'strategy_results.csv','text/csv')
    with tabs[1]:
        periods=sorted(set(zip(bets.season.astype(int),bets.week.astype(int))))
        if len(periods)<2: st.info('At least two completed weeks are needed for chronological validation.')
        else:
            default=next((i for i,p in enumerate(periods[:-1]) if p==(2025,11)),max(0,len(periods)//2-1))
            split=st.selectbox('Last week used to choose the strategy',periods[:-1],index=default,format_func=lambda p:f'{p[0]} Week {p[1]}')
            train=bets[(bets.season<split[0])|((bets.season==split[0])&(bets.week<=split[1]))]
            test=bets[(bets.season>split[0])|((bets.season==split[0])&(bets.week>split[1]))]
            training=leaderboard(train,int(minimum)); eligible=training[training.enough_bets]
            if eligible.empty: st.info('No strategy has enough graded training bets. Download more weeks or lower the minimum.')
            else:
                winner=eligible.iloc[0].strategy
                st.markdown(f'**Chosen using earlier weeks only:** {winner}')
                st.caption('The same strategy is then held fixed for every later week. Later results did not select this winner. Repeatedly changing this split after viewing results weakens the validation.')
                show_metrics(test[test.strategy==winner])
                with st.expander('Training-only leaderboard'): display_leaderboard(training)
            st.subheader('Week-by-week adaptive selection')
            st.caption('Before each week, choose the qualifying strategy with the most units from completed earlier weeks. No current-week result enters selection. The first weeks are a warm-up period.')
            wf=walk_forward(bets,int(minimum)); show_metrics(wf)
            if not wf.empty:
                table(wf.groupby(['season','week','strategy'],as_index=False).agg(bets=('player','count'),units=('units','sum')))
    with tabs[2]:
        available=set(bets.strategy)
        chosen=st.selectbox('Strategy to inspect',[name for name in summary.strategy if name in available],index=0)
        detail=bets[bets.strategy==chosen].copy(); show_metrics(detail)
        weekly=pd.DataFrame([dict(season=y,week=w,**summarize(g)) for (y,w),g in detail.groupby(['season','week'])])
        weekly['cumulative_units']=weekly.units.cumsum(); weekly['period']=weekly.season.astype(str)+' W'+weekly.week.astype(str)
        st.line_chart(weekly.set_index('period')[['cumulative_units']]); display_leaderboard(weekly)
        options=['All']+[f'{y} W{w}' for y,w in sorted(set(zip(detail.season,detail.week)))]
        selected=st.selectbox('Week',options)
        if selected!='All':
            y,w=selected.split(' W'); detail=detail[(detail.season==int(y))&(detail.week==int(w))]
        cols=['season','week','bet_rank','player','team','opponent','side','book','line','bet_line','prior_line','line_move','original_rank','original_score','projected_carries','sma_share','sma_team_rb_carries','sma_projected_carries','sma_edge','actual_carries','result','units','snapshot']
        table(detail.reindex(columns=cols))
        st.download_button('Download selected bets',detail.to_csv(index=False),'selected_backtest_bets.csv','text/csv')
    with tabs[3]:
        st.metric('Eligible leader rows with a matched line',f'{dataset.line.notna().sum()}/{len(dataset)}')
        table(dataset.assign(has_line=dataset.line.notna(),has_previous=dataset.prior_line.notna(),has_actual=dataset.actual_carries.notna()).groupby(['season','week'],as_index=False).agg(leaders=('player','count'),posted_lines=('has_line','sum'),prior_lines=('has_previous','sum'),official_results=('has_actual','sum')))
        st.subheader('Event download audit'); table(audit)
        st.subheader('Excluded or unresolved rows'); table(dataset[dataset.line.isna()|dataset.actual_carries.isna()])
        st.download_button('Download full feature and outcome dataset',dataset.to_csv(index=False),'backtest_dataset.csv','text/csv')
    with tabs[5]:
        st.subheader('Three-game moving-average projection')
        st.markdown('**Projected carries = average player RB carry share × average team RB carries**, using the same three completed team games. Each game has equal weight; byes are skipped.')
        st.caption('One added strategy: bet the over on every eligible team leader whose projection exceeds the standard posted line. No fixed bet count, movement filter, or tuned edge cutoff. Exactly three prior team games are required. This uses your saved statistics and historical lines: no new API credits.')
        st.code('Average share = (share game 1 + share game 2 + share game 3) / 3\nAverage team RB carries = (team carries game 1 + game 2 + game 3) / 3\nProjected carries = Average share × Average team RB carries\nEdge = Projected carries − Posted line\nBet Over only when Edge > 0; otherwise No bet')
        projection_bets=bets[bets.strategy=='3-game average share × team volume · Positive-edge overs'].copy()
        show_metrics(projection_bets)
        if offset:
            st.warning('The selected hypothetical −4 scenario changes grading only. Projection selections still use the standard posted line. Choose Standard posted line above for the actual line backtest.')
        if not projection_bets.empty:
            weekly_projection=pd.DataFrame([dict(season=y,week=w,**summarize(g)) for (y,w),g in projection_bets.groupby(['season','week'])])
            weekly_projection['cumulative_units']=weekly_projection.units.cumsum()
            weekly_projection['period']=weekly_projection.season.astype(str)+' W'+weekly_projection.week.astype(str)
            st.line_chart(weekly_projection.set_index('period')[['cumulative_units']])
            display_leaderboard(weekly_projection)
        else:
            st.info('No positive-edge selections with the required three games of history for this data source.')
        audit_projection=dataset.copy()
        audit_projection['decision']='No bet: projection at/below line'
        audit_projection.loc[audit_projection.sma_edge>0,'decision']='Bet Over'
        audit_projection.loc[audit_projection.line.isna(),'decision']='No bet: missing line'
        audit_projection.loc[audit_projection.sma_projected_carries.isna(),'decision']='No bet: need 3 valid prior team games'
        pcols=['season','week','player','team','opponent','sma_weeks','sma_share_game_1','sma_share_game_2','sma_share_game_3','sma_share','sma_team_rb_carries','sma_projected_carries','line','sma_edge','decision','actual_carries']
        st.subheader('Projection calculations and every decision')
        st.caption('Share columns are fractions: 0.70 means 70%. The window runs oldest to newest. Missing player rows in a prior team game count as zero carries. No opponent, spread, or injury adjustment is included in this first benchmark. The population remains the pregame team leaders in your saved dataset, not every posted RB.')
        table(audit_projection.reindex(columns=pcols))
        st.download_button('Download projection calculations',audit_projection.reindex(columns=pcols).to_csv(index=False),'three_game_projection.csv','text/csv')
        st.download_button('Download projection bets',projection_bets.to_csv(index=False),'three_game_projection_bets.csv','text/csv')
    with tabs[4]:
        st.markdown('''
**Scoring.** One unit per graded selection: +1 for a win, −1 for a loss, 0 for a push. Win rate excludes pushes; ROI is units / graded bets, including pushes. This is your even-odds convention, not sportsbook or PrizePicks cash profit.

**Original baseline.** Reuses the app's exact original 50/50 ranking function, using only prior weeks. Its original top 10 are fixed before checking lines. Movement strategies reorder only those players: flat/down first, +0.5 to +1.5 next, +2 or more last; OG rank breaks ties. Missing prior lines go last. The flat/down-only strategy excludes them. An original top-five strategy never backfills a missing line with number six.

**Six original over-only strategies.** Original 50/50 top 5; original 50/50 top 10; original Top 10 reordered by line movement, bet 5; original Top 10 with flat/down lines only, bet up to 5; 75% defense / 25% latest usage change, bet 5; and 50/50 with the three-game usage trend, bet 5.

**Alternative rankings.** The 75/25 and three-game-trend strategies use the full weekly leader pool. Leaders and shares use each team's recent completed games, so a bye does not remove its back. Original baselines preserve the old app's calendar-week behavior for a faithful comparison.

**Three-game trend.** 50% latest share change + 50% average share change per game from the first to last of up to three prior team games. A big latest jump still matters. Missing player rows in a prior team game count as zero carries, not zero team games. Week 3 has only two games of history.

**Defense.** Average RB carries allowed per completed game before the target week; quarterbacks are excluded. This measures rushing volume allowed, not yards-per-carry efficiency.

**Added three-game projection.** Equal-weight mean of the last three team-game RB carry shares, multiplied by equal-weight mean of team RB carries in those same games. Select all positive-edge overs in the saved leader pool. Require three games; skip byes; use only weeks before the target. It automatically reconstructs these inputs from the stats in an existing backup. This is a seventh strategy.

**Reference projection.** The bet-detail table still shows projected carries for context. This older reference projection is not used to select bets. The new three-game projection appears in separate columns and is not a probability.

**Historical lines.** One fixed timestamp per week, before its first game; standard `player_rush_attempts` only. Default source uses a fixed book priority, not the most favorable line. Previous-week movement uses the same player and same book; byes have no previous-week line. Only quotes containing both Over and Under outcomes are retained. Alternate or ambiguous multiple lines are excluded. No live, final, synthetic, or postgame line fills missing history.

**Outcomes.** Match official rows by season, week, team, and player ID. No official row means ungraded, not an assumed loss or zero carries. Book-specific injury/void rules are not modeled. Injury adjustments are not tested because timestamped injury inputs are not part of this dataset. Official historical statistics may include later corrections.

**Validation.** The descriptive leaderboard optimizes on all selected results. The held-out result selects one strategy using only the earlier period and evaluates it unchanged later. The adaptive comparison reselects each week using only earlier results. Even with a small number of strategies, choosing the best full-sample result creates selection bias; check the later-week validation.

**Sources.** [The Odds API historical documentation](https://the-odds-api.com/liveapi/guides/v4/#get-historical-event-odds), [nflverse weekly player statistics](https://github.com/nflverse/nflverse-data/releases/tag/stats_player), and [nflverse schedules](https://github.com/nflverse/nfldata/blob/master/data/games.csv).
''')
