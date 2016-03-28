from pattern import web
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dateutil import parser
import os
import os.path

def extract_id(i):
    return i.split('/')[-1].split('.')[0]

proxies = {
#  'http': 'http://10.201.96.145:80',
#  'https': 'http://10.201.96.145:80',
}

data = [['team1_name', 'team2_name', 'team1_id', 'team2_id', 'ground_name', 'ground_id', 'match_id', 'date']]
year = datetime.utcnow().year
r = requests.get('http://stats.espncricinfo.com/sl/engine/records/team/match_results.html?class=3;id=%d;type=year'%year,
                proxies=proxies)
dom = web.Element(r.text)
tab = dom.by_tag('table.engineTable')[0].by_tag('tr.data1')

for t in tab:
    t = t.by_tag('td')
    if web.plaintext(t[2].content) != 'no result':
        t1 = web.plaintext(t[0].content)
        t2 = web.plaintext(t[1].content)
        t1_id = extract_id(t[0].by_tag('a')[0].attrs['href'])
        t2_id = extract_id(t[1].by_tag('a')[0].attrs['href'])
        ground_name = web.plaintext(t[4].content)
        ground_id = extract_id(t[4].by_tag('a')[0].attrs['href'])
        match_id = extract_id(t[-1].by_tag('a')[0].attrs['href'])
        date = web.plaintext(t[-2].content)
        data.append([t1, t2, t1_id, t2_id, ground_name, ground_id, match_id, date])

def f(i):
    if '-' in i:
        i = i.split(',')[0].split('-')[0] + i.split(',')[1]
    return parser.parse(i)

df = pd.DataFrame(data[1:], columns = data[0])
df['date'] = df['date'].apply(f)

orig_df = pd.read_csv('all_t20i_05-16.csv')
del orig_df['Unnamed: 0']
orig_df = orig_df.append(df).drop_duplicates(subset=['match_id'])
orig_df.to_csv('all_t20i_05-16.csv')

df = pd.read_csv('all_t20i_05-16.csv')
matches = df.sort('date', ascending = False).match_id.values
matches = [str(i) for i in matches]

n = len(matches)

for idx, game_id in enumerate(matches):
    if not os.path.exists('cleaned_data/%s_summary.csv'%game_id):
        try:
            r1 = requests.get('http://www.espncricinfo.com/ci/engine/match/gfx/%s.json?inning=1;template=wagon'%(game_id),
                             proxies=proxies)
            r2 = requests.get('http://www.espncricinfo.com/ci/engine/match/gfx/%s.json?inning=2;template=wagon'%(game_id),
                             proxies=proxies)
            #print r1.status_code
            #print r2.status_code
            with open('data/'+game_id+'-wagon-inning-1.json', 'w') as outfile:
                json.dump(r1.json(), outfile)

            with open('data/'+str(game_id)+'-wagon-inning-2.json', 'w') as outfile:
                json.dump(r2.json(), outfile)

            r = requests.get('http://www.espncricinfo.com/ci/engine/match/gfx/%s.json?template=pie_wickets'%(game_id),
                            proxies=proxies)
            with open('data/'+game_id+'-pie-wickets.json', 'w') as outfile:
                json.dump(r.json(), outfile)

            try:
                r = requests.get('http://www.espncricinfo.com/ci/engine/match/%s.html?view=hawkeye'%game_id)
                dom = web.Element(r.content)
                data_id = dom.by_id('matchId').attr.get('data-contentid')
                r = requests.get('http://dynamic.pulselive.com/dynamic/cricinfo/%s/uds.json.jgz'%data_id)
                with open('data/'+game_id+'_ball_details.txt', 'wb') as f:
                    f.write(r.content)
            except:
                pass
            print 'Success', game_id, idx, n
        except:
            print 'Failed', game_id, idx, n
    else:
        print 'Already Exists', game_id, idx, n

all_matches = pd.read_csv('all_t20i_05-16.csv')
all_matches = all_matches.sort('date', ascending = False)
matches = all_matches.match_id.values

def f(i):
    return i['bat'], i['bowl'], i['o_u'], i['ovr'], i['r'], i['r_t'], i['x'], i['y'], i['z']

import os.path
failed_ids = []

for idx, match_id in enumerate(matches):
    #if not os.path.exists('cleaned_data/%s_summary.csv'%match_id):
    try:
        wicket_data = json.load(open('data/%s-pie-wickets.json'%(match_id)))
        df = pd.io.json.read_json('data/%s-wagon-inning-1.json'%(match_id), dtype=False)    
        df['inning'] = 1
        df['batting_team'] = wicket_data['t1']['n']
        df['bowling_team'] = wicket_data['t2']['n']

        df2 = pd.io.json.read_json('data/%s-wagon-inning-2.json'%(match_id), dtype=False)
        df2['inning'] = 2
        df2['batting_team'] = wicket_data['t2']['n']
        df2['bowling_team'] = wicket_data['t1']['n']

        df['batsman'], df['bowler'], df['ball_num'], df['ovr'], df['runs_batter'], df['runs_w_extras'], df['x'], df['y'], df['z'] = zip(*df.runs.apply(f))
        df2['batsman'], df2['bowler'], df2['ball_num'], df2['ovr'], df2['runs_batter'], df2['runs_w_extras'], df2['x'], df2['y'], df2['z'] = zip(*df2.runs.apply(f))

        for param in ['ball_num', 'ovr', 'runs_batter', 'runs_w_extras', 'x', 'y', 'z']:
            df[param] = df[param].astype(float)
            df2[param] = df2[param].astype(float)

        from collections import defaultdict
        dismissals = defaultdict(dict)

        for i in wicket_data['t1']['w']:
            dismissals[float(i['ovr'])] = {'inning': 1, 'how': i['how'], 'ovr': float(i['ovr']), 'batsman': i['out'], 'bowler': i['outby']}

        for i in wicket_data['t2']['w']:
            dismissals[float(i['ovr'])] = {'inning': 2, 'how': i['how'], 'ovr': float(i['ovr']), 'batsman': i['out'], 'bowler': i['outby']}

        def add_wickets(inning, ovr, batsman, extras, unique, dismissals = dismissals):
            #Note: a run-out on a no-ball or wide will be double-counted
            if ovr in dismissals and dismissals[ovr]['inning'] == inning:
                if unique == 'True':
                    return True, dismissals[ovr]['how'], dismissals[ovr]['batsman']
                else:
                    if dismissals[ovr]['how'] != 'run out':
                        if extras == 0:
                            return True, dismissals[ovr]['how'], dismissals[ovr]['batsman']
                        else:
                            return False, np.nan, np.nan
                    else:
                        return True, dismissals[ovr]['how'], dismissals[ovr]['batsman'] #think of a better way later
            return False, np.nan, np.nan

        val_count_1 = df.ovr.value_counts()
        val_count_2 = df2.ovr.value_counts()

        def get_unique_1(i):
            if val_count_1[i] == 1:
                return True
            return False

        def get_unique_2(i):
            if val_count_2[i] == 1:
                return True
            return False

        df['ovr_unique'] = df.ovr.apply(get_unique_1)
        df2['ovr_unique'] = df2.ovr.apply(get_unique_2)
        df['extras'] = df['runs_w_extras'] - df['runs_batter']
        df2['extras'] = df2['runs_w_extras'] - df2['runs_batter']
        df['wicket'], df['wicket_method'], df['who_out'] = zip(*df.apply(lambda row: add_wickets(row['inning'], row['ovr'], row['batsman'], row['extras'], row['ovr_unique']), axis=1))
        df2['wicket'], df2['wicket_method'], df2['who_out'] = zip(*df2.apply(lambda row: add_wickets(row['inning'], row['ovr'], row['batsman'], row['extras'], row['ovr_unique']), axis=1))
        df['match_id'] = match_id
        df2['match_id'] = match_id

        df['cumul_runs'] = df['runs_w_extras'].cumsum()
        df['cumul_wickets'] = df['wicket'].cumsum()
        df['cumul_balls'] = df['ovr'].apply(lambda x: 6*int(x) + int(10*x)%10)
        df2['cumul_runs'] = df2['runs_w_extras'].cumsum()
        df2['cumul_wickets'] = df2['wicket'].cumsum()
        df2['cumul_balls'] = df2['ovr'].apply(lambda x: 6*int(x) + int(10*x)%10)

        player_db = {}
        inn_1_batting_order = {}
        inn_2_batting_order = {}
        for idx__, d in enumerate(wicket_data['t1']['p']):
            player_db.update(d)
            inn_1_batting_order[d.keys()[0]] = idx__+1
        for idx__, d in enumerate(wicket_data['t2']['p']):
            player_db.update(d)
            inn_2_batting_order[d.keys()[0]] = idx__+1

        def get_name(i):
            return player_db.get(i, None)

        df['batsman_name'] = df['batsman'].apply(get_name)
        df['bowler_name'] = df['bowler'].apply(get_name)
        df['who_out'] = df['who_out'].apply(get_name)
        df['batting_order'] = df['batsman'].apply(inn_1_batting_order.get)

        df2['batsman_name'] = df2['batsman'].apply(get_name)
        df2['bowler_name'] = df2['bowler'].apply(get_name)
        df2['who_out'] = df2['who_out'].apply(get_name)
        df2['batting_order'] = df2['batsman'].apply(inn_2_batting_order.get)

        df = df.append(df2)    
        df = df.drop('runs', axis=1)
        df = df.reset_index(drop=True)

        try:
            with open('data/%s_ball_details.txt'%(match_id), 'r') as fil:
                a = eval(fil.read()[8:-2])
            l = []
            for i in a[0]['data']:
                temp = i.values()[0].split(',')
                l.append({'non_striker': temp[1], 'ball_speed': temp[3], 'landing_y': temp[4], 'landing_x': temp[5],
                  'bat_right_handed': temp[6], 'ended_x': temp[7], 'ended_y': temp[8],
                  'control': int(temp[9] == 'N'), 'extras_type': temp[19]})
            df2 = pd.DataFrame(l)
            if len(df2) == len(df):
                df = df.join(df2)
                cols = ['inning', 'batting_team', 'bowling_team', 'batsman', 'bowler', 'batsman_name', 'batting_order'
                'non_striker', 'bowler_name', 'bat_right_handed', 'ovr', 'runs_batter', 'runs_w_extras', 'extras',
                'x', 'y', 'z', 'landing_x', 'landing_y', 'ended_x', 'ended_y', 'ball_speed', 'cumul_runs',
                'cumul_wickets',  'cumul_balls', 'wicket', 'wicket_method', 'who_out', 'control', 'extras_type']
            else:
                print 'discrepancy', match_id
                cols = ['inning', 'batting_team', 'bowling_team', 'batsman', 'bowler', 'batsman_name', 'batting_order',
                'bowler_name', 'ovr', 'runs_batter', 'runs_w_extras', 'extras',
                'x', 'y', 'z', 'cumul_runs', 'cumul_wickets', 'cumul_balls', 'wicket', 'wicket_method', 'who_out']
        except:
            cols = ['inning', 'batting_team', 'bowling_team', 'batsman', 'bowler', 'batsman_name', 'batting_order',
                'bowler_name', 'ovr', 'runs_batter', 'runs_w_extras', 'extras',
                'x', 'y', 'z', 'cumul_runs', 'cumul_wickets', 'cumul_balls', 'wicket', 'wicket_method', 'who_out']
        df = df[cols]
        df.to_csv('cleaned_data/%s_summary.csv'%(match_id))
        print 'success', idx, len(matches)
    except:
        print 'failed', idx, match_id, len(matches)
        failed_ids.append(int(match_id))
    #else:
    #    print 'already exists', idx, len(matches)

matches = pd.read_csv('all_t20i_05-16.csv')
match_date = matches.set_index('match_id').date.to_dict()
files = sorted(match_date.items(), key=lambda value: value[1])

def get_req_rr(row):
    if 'cumul_balls' in row:
        runs_to_get = row['target'] - row['cumul_runs']
        balls_remaining = 120 - row['cumul_balls']
        if balls_remaining > 0 and runs_to_get >= 0:
            return 6.*runs_to_get/balls_remaining
    return None

all_data = pd.DataFrame()
for idx, fname in enumerate(files):
    if os.path.exists('cleaned_data/%d_summary.csv'%fname[0]):
        ind_game = pd.read_csv('cleaned_data/%d_summary.csv'%fname[0])#.sort_values('inning').sort_values('ovr')
        ind_game['date'] = fname[1]
        t1_score = ind_game[ind_game.inning == 1].runs_w_extras.sum()
        ind_game['target'] = ind_game.inning.apply(lambda x: t1_score if x == 2 else None)
        ind_game['current_run_rate'] = ind_game.apply(lambda row: 6.*row['cumul_runs']/row['cumul_balls'], axis=1)
        ind_game['required_run_rate'] = ind_game.apply(lambda row: get_req_rr(row) if row['inning'] == 2 else None, axis=1)
        ind_game['ovr_range'] = ind_game.ovr.apply(lambda x: 'first_6' if x < 6 else 'middle_9' if x < 15 else 'last_5')
        all_data = all_data.append(ind_game)
    if idx%50 == 0:
        print idx, len(files)
all_data['balls'] = 1
all_data = all_data.reset_index(drop=True)
cols = all_data.columns.tolist()
all_data = all_data[cols[1:]]
all_data['year'] = all_data['date'].apply(lambda x: x.split('-')[0])

batsman_team_dict = all_data.drop_duplicates(subset = ['batsman_name']).set_index('batsman_name')['batting_team'].to_dict()
bowler_team_dict = all_data.drop_duplicates(subset = ['bowler_name']).set_index('bowler_name')['bowling_team'].to_dict()

major_teams = ['Australia', 'New Zealand', 'England', 'South Africa', 'West Indies', 'Sri Lanka',
               'Pakistan', 'India', 'Bangladesh']

team_innings = pd.DataFrame()
team_innings['opposition'] = all_data.groupby(['batting_team', 'date'])['bowling_team'].apply(lambda x: x.tolist()[0])
team_innings['inning'] = all_data.groupby(['batting_team', 'date'])['inning'].apply(lambda x: x.tolist()[0])
team_innings['opposition_score'] = all_data.groupby(['bowling_team', 'date'])['runs_w_extras'].agg('sum')
team_innings['runs_scored'] = all_data.groupby(['batting_team', 'date'])['runs_w_extras'].agg('sum')
team_innings['win'] = team_innings['runs_scored'] > team_innings['opposition_score']
team_innings['num_out'] = all_data.groupby(['batting_team', 'date'])['wicket'].agg('sum')
team_innings['balls_faced'] = all_data.groupby(['batting_team', 'date'])['ovr'].nunique()
team_innings['num_fours'] = all_data.groupby(['batting_team', 'date'])['runs_batter'].apply(lambda x: sum([i == 4 for i in x.tolist()]))
team_innings['num_sixes'] = all_data.groupby(['batting_team', 'date'])['runs_batter'].apply(lambda x: sum([i == 6 for i in x.tolist()]))
team_innings['num_dots'] = all_data.groupby(['batting_team', 'date'])['runs_batter'].apply(lambda x: sum([i == 0 for i in x.tolist()]))
team_innings = team_innings.reset_index()
team_innings['year'] = team_innings['date'].apply(lambda x: int(x.split('-')[0]))
team_innings['year'] = team_innings['year'].apply(lambda x: '05-07' if x <= 2007 else '08-10' if x <= 2010 else '11-13' if x <=2013 else '14-16')
team_innings['run_rate'] = 6.*team_innings.apply(lambda row: row['runs_scored']/max(row['balls_faced'], 1), axis=1)

team_year = team_innings.groupby(['batting_team', 'year'])[
    ['win', 'runs_scored', 'balls_faced', 'num_fours', 'num_sixes','num_dots']].agg('sum')\
    .join(team_innings.groupby(['batting_team', 'year'])['date'].count()).reset_index()
team_year.columns = team_year.columns.tolist()[:-1] + ['num_games']
team_year = team_year[['batting_team', 'year', 'num_games', 'win', 'runs_scored', 'balls_faced',
                            'num_fours', 'num_sixes', 'num_dots']]
team_year['win_percentage'] = 1.*team_year['win']/team_year['num_games']
team_year['prop_runs_sixes'] = 1.*(6.*team_year['num_sixes'])/team_year['runs_scored']
team_year['prop_runs_boundaries'] = 1.*(6.*team_year['num_sixes'] + 4.*team_year['num_fours'])/team_year['runs_scored']
team_year['run_rate'] = 6.*team_year['runs_scored']/team_year['balls_faced']
team_year['dot_prop'] = 1.*team_year['num_dots']/team_year['balls_faced']
team_year['balls_per_six'] = 1.*team_year['balls_faced']/team_year['num_sixes']
team_year['balls_per_four'] = 1.*team_year['balls_faced']/team_year['num_fours']
team_year['balls_per_boundary'] = 1.*team_year['balls_faced']/(team_year['num_fours'] + team_year['num_sixes'])

team_batting_pos = pd.DataFrame()
team_batting_pos['runs_scored'] = all_data.groupby(['batting_team', 'batting_order'])['runs_w_extras'].agg('sum')
team_batting_pos['num_out'] = all_data.groupby(['batting_team', 'batting_order'])['wicket'].agg('sum')
team_batting_pos['balls_faced'] = all_data.groupby(['batting_team', 'batting_order'])['ovr'].count()
team_batting_pos['num_fours'] = all_data.groupby(['batting_team', 'batting_order'])['runs_batter'].apply(lambda x: sum([i == 4 for i in x.tolist()]))
team_batting_pos['num_sixes'] = all_data.groupby(['batting_team', 'batting_order'])['runs_batter'].apply(lambda x: sum([i == 6 for i in x.tolist()]))
team_batting_pos['num_dots'] = all_data.groupby(['batting_team', 'batting_order'])['runs_batter'].apply(lambda x: sum([i == 0 for i in x.tolist()]))
team_batting_pos = team_batting_pos.reset_index()
team_batting_pos['strike_rate'] = 100.*team_batting_pos.apply(lambda row: row['runs_scored']/max(row['balls_faced'], 1), axis=1)
team_batting_pos['average'] = 1.*team_batting_pos.apply(lambda row: row['runs_scored']/max(row['num_out'], 1), axis=1)

batsman_innings = pd.DataFrame()
batsman_innings['inning'] = all_data.groupby(['batsman_name', 'date'])['inning'].apply(lambda x: x.tolist()[0])
batsman_innings['opposition'] = all_data.groupby(['batsman_name', 'date'])['bowling_team'].apply(lambda x: x.tolist()[0])
batsman_innings['runs_scored'] = all_data.groupby(['batsman_name', 'date'])['runs_batter'].agg('sum')
batsman_innings['num_out'] = all_data.groupby(['batsman_name', 'date'])['wicket'].agg('sum')
batsman_innings['balls_faced'] = all_data.groupby(['batsman_name', 'date'])['ovr'].count()
batsman_innings['num_fours'] = all_data.groupby(['batsman_name', 'date'])['runs_batter'].apply(lambda x: sum([i == 4 for i in x.tolist()]))
batsman_innings['num_sixes'] = all_data.groupby(['batsman_name', 'date'])['runs_batter'].apply(lambda x: sum([i == 6 for i in x.tolist()]))
batsman_innings['num_dots'] = all_data.groupby(['batsman_name', 'date'])['runs_batter'].apply(lambda x: sum([i == 0 for i in x.tolist()]))
batsman_innings['batting_order'] = all_data.groupby(['batsman_name', 'date'])['batting_order'].apply(lambda x: x.dropna().values[0])
batsman_innings['wicket_method'] = all_data.groupby(['batsman_name', 'date'])['wicket_method'].apply(lambda x: x.dropna().tolist())
batsman_innings['wicket_method'] = batsman_innings['wicket_method'].apply(lambda x: 'not out' if len(x) == 0 else x[-1])
batsman_innings['wicket_method'] = batsman_innings.apply(lambda row: 'not out' if row['num_out'] == 0 else row['wicket_method'], axis=1)
for i in ['caught', 'not out', 'leg before wicket', 'bowled', 'hit wicket',
       'run out', 'stumped', 'retired not out (hurt)']:
    batsman_innings[i] = batsman_innings['wicket_method'] == i
batsman_innings = batsman_innings.reset_index()
batsman_innings['team'] = batsman_innings['batsman_name'].apply(batsman_team_dict.get)
batsman_innings['year'] = batsman_innings['date'].apply(lambda x: int(x.split('-')[0]))
batsman_innings['year'] = batsman_innings['year'].apply(lambda x: '05-07' if x <= 2007 else '08-10' if x <= 2010 else '11-13' if x <=2013 else '14-16')
batsman_innings['strike_rate'] = 100.*batsman_innings.apply(lambda row: row['runs_scored']/max(row['balls_faced'], 1), axis=1)

batsman_params = ['runs_scored', 'balls_faced', 'num_fours', 'num_sixes', 'num_dots', 'num_out',
                  'caught', 'not out', 'leg before wicket', 'bowled', 'hit wicket',
                  'run out', 'stumped', 'retired not out (hurt)']
batsman_order = batsman_innings.groupby(['batsman_name', 'batting_order'])[batsman_params].agg('sum')\
    .join(batsman_innings.groupby(['batsman_name', 'batting_order'])['date'].nunique())
batsman_order.columns = batsman_order.columns.tolist()[:-1] + ['num_innings']
batsman_order = batsman_order.reset_index()
for i in ['caught', 'not out', 'leg before wicket', 'bowled', 'hit wicket',
       'run out', 'stumped', 'retired not out (hurt)']:
    batsman_order[i] = 1.*batsman_order[i]/batsman_order['num_innings']
batsman_order['team'] = batsman_order['batsman_name'].apply(batsman_team_dict.get)
batsman_order['average'] = batsman_order.apply(lambda row: row['runs_scored']/max(row['num_out'], 1), axis=1)
batsman_order['runs_per_inning'] = batsman_order['runs_scored']/batsman_order['num_innings']
batsman_order['strike_rate'] = 100.*batsman_order.apply(lambda row: row['runs_scored']/max(row['balls_faced'], 1), axis=1)
batsman_order['impact'] = batsman_order['strike_rate']*batsman_order['runs_per_inning']
batsman_order['prop_dot'] = 100.*batsman_order['num_dots']/batsman_order['balls_faced']
batsman_order['balls_per_six'] = batsman_order['balls_faced']/batsman_order['num_sixes']
batsman_order['balls_per_boundary'] = batsman_order['balls_faced']/(batsman_order['num_sixes'] + batsman_order['num_fours'])
batsman_order['prop_run_boundary'] = 100.*(6.*batsman_order['num_sixes'] + 4.*batsman_order['num_fours'])/batsman_order['runs_scored']
batsman_order['prop_run_six'] = 100.*(6.*batsman_order['num_sixes'])/batsman_order['runs_scored']

batsman_params = ['runs_scored', 'balls_faced', 'num_fours', 'num_sixes', 'num_dots', 'num_out',
                  'caught', 'not out', 'leg before wicket', 'bowled', 'hit wicket',
                  'run out', 'stumped', 'retired not out (hurt)']
batsman_year = batsman_innings.groupby(['batsman_name', 'year'])[batsman_params].agg('sum')\
    .join(batsman_innings.groupby(['batsman_name', 'year'])['date'].nunique())
batsman_year.columns = batsman_year.columns.tolist()[:-1] + ['num_innings']
batsman_year = batsman_year.reset_index()
for i in ['caught', 'not out', 'leg before wicket', 'bowled', 'hit wicket',
       'run out', 'stumped', 'retired not out (hurt)']:
    batsman_year[i] = 1.*batsman_year[i]/batsman_year['num_innings']
batsman_year['team'] = batsman_year['batsman_name'].apply(batsman_team_dict.get)
batsman_year['average'] = batsman_year.apply(lambda row: row['runs_scored']/max(row['num_out'], 1), axis=1)
batsman_year['runs_per_inning'] = batsman_year['runs_scored']/batsman_year['num_innings']
batsman_year['strike_rate'] = 100.*batsman_year.apply(lambda row: row['runs_scored']/max(row['balls_faced'], 1), axis=1)
batsman_year['impact'] = batsman_year['strike_rate']*batsman_year['runs_per_inning']
batsman_year['prop_dot'] = 100.*batsman_year['num_dots']/batsman_year['balls_faced']
batsman_year['balls_per_six'] = batsman_year['balls_faced']/batsman_year['num_sixes']
batsman_year['balls_per_boundary'] = batsman_year['balls_faced']/(batsman_year['num_sixes'] + batsman_year['num_fours'])
batsman_year['prop_run_boundary'] = 100.*(6.*batsman_year['num_sixes'] + 4.*batsman_year['num_fours'])/batsman_year['runs_scored']
batsman_year['prop_run_six'] = 100.*(6.*batsman_year['num_sixes'])/batsman_year['runs_scored']

bowler_spells = pd.DataFrame()
bowler_spells['inning'] = all_data.groupby(['bowler_name', 'date'])['inning'].apply(lambda x: x.tolist()[0])
bowler_spells['opposition'] = all_data.groupby(['bowler_name', 'date'])['batting_team'].apply(lambda x: x.tolist()[0])
bowler_spells['runs_conceded'] = all_data.groupby(['bowler_name', 'date'])['runs_batter'].agg('sum')
bowler_spells['wickets_taken'] = all_data.groupby(['bowler_name', 'date'])['wicket'].agg('sum')
bowler_spells['balls_bowled'] = all_data.groupby(['bowler_name', 'date'])['ovr'].count()
bowler_spells['num_dots'] = all_data.groupby(['bowler_name', 'date'])['runs_batter'].apply(lambda x: sum([i == 0 for i in x.tolist()]))
bowler_spells = bowler_spells.reset_index()
bowler_spells['team'] = bowler_spells['bowler_name'].apply(bowler_team_dict.get)
bowler_spells['year'] = bowler_spells['date'].apply(lambda x: int(x.split('-')[0]))
bowler_spells['year'] = bowler_spells['year'].apply(lambda x: '05-07' if x <= 2007 else '08-10' if x <= 2010 else '11-13' if x <=2013 else '14-16')
bowler_spells['economy_rate'] = 6.*bowler_spells.apply(lambda row: row['runs_conceded']/max(row['balls_bowled'], 1), axis=1)

bowler_params = ['runs_conceded', 'balls_bowled', 'wickets_taken', 'num_dots']
bowler_year = bowler_spells.groupby(['bowler_name', 'year'])[bowler_params].agg('sum')\
    .join(bowler_spells.groupby(['bowler_name', 'year'])['date'].nunique())
bowler_year.columns = bowler_year.columns.tolist()[:-1] + ['num_spells']
bowler_year = bowler_year.reset_index()
bowler_year['team'] = bowler_year['bowler_name'].apply(bowler_team_dict.get)
bowler_year['average'] = bowler_year['runs_conceded']/bowler_year['wickets_taken']
bowler_year['wickets_per_spell'] = bowler_year['wickets_taken']/bowler_year['num_spells']
bowler_year['strike_rate'] = bowler_year['balls_bowled']/bowler_year['wickets_taken']
bowler_year['economy_rate'] = 6.*bowler_year['runs_conceded']/bowler_year['balls_bowled']
bowler_year['prop_dot'] = 100.*bowler_year['num_dots']/bowler_year['balls_bowled']

chasing_under_pressure_team = all_data[all_data.required_run_rate > 9].groupby('batting_team')[['runs_w_extras', 'balls', 'wicket']].agg('sum')
chasing_under_pressure_team['run_rate'] = 6.*chasing_under_pressure_team['runs_w_extras']/chasing_under_pressure_team['balls']
chasing_under_pressure_team['average'] = 1.*chasing_under_pressure_team['runs_w_extras']/chasing_under_pressure_team['wicket']

chasing_under_pressure = all_data[all_data.required_run_rate > 9].groupby('batsman_name')[['runs_batter', 'balls', 'wicket']].agg('sum')
chasing_under_pressure['strike_rate'] = 1.*chasing_under_pressure['runs_batter']/chasing_under_pressure['balls']
chasing_under_pressure['average'] = 1.*chasing_under_pressure['runs_batter']/chasing_under_pressure['wicket']

first_inning_team = team_innings[team_innings.inning == 1].groupby('batting_team')[
    ['win', 'runs_scored', 'num_out', 'balls_faced', 'num_fours', 'num_sixes', 'num_dots']].agg('sum')\
    .join(team_innings[team_innings.inning == 1].groupby('batting_team')['date'].count())
first_inning_team.columns = first_inning_team.columns.tolist()[:-1] + ['total_matches']
first_inning_team['average'] = 1.*first_inning_team['runs_scored']/first_inning_team['num_out']
first_inning_team['run_rate'] = 6.*first_inning_team['runs_scored']/first_inning_team['balls_faced']
first_inning_team['balls_per_six'] = 1.*first_inning_team['balls_faced']/first_inning_team['num_sixes']
first_inning_team['balls_per_boundary'] = 1.*first_inning_team['balls_faced']/first_inning_team['num_fours']
first_inning_team['win_rate'] = 1.*first_inning_team['win']/first_inning_team['total_matches']

second_inning_team = team_innings[team_innings.inning == 2].groupby('batting_team')[
    ['win', 'runs_scored', 'num_out', 'balls_faced', 'num_fours', 'num_sixes', 'num_dots']].agg('sum')\
    .join(team_innings[team_innings.inning == 2].groupby('batting_team')['date'].count())
second_inning_team.columns = second_inning_team.columns.tolist()[:-1] + ['total_matches']
second_inning_team['average'] = 1.*second_inning_team['runs_scored']/second_inning_team['num_out']
second_inning_team['run_rate'] = 6.*second_inning_team['runs_scored']/second_inning_team['balls_faced']
second_inning_team['balls_per_six'] = 1.*second_inning_team['balls_faced']/second_inning_team['num_sixes']
second_inning_team['balls_per_boundary'] = 1.*second_inning_team['balls_faced']/second_inning_team['num_fours']
second_inning_team['win_rate'] = 1.*second_inning_team['win']/second_inning_team['total_matches']

first_inning_bat = batsman_innings[batsman_innings.inning == 1].groupby('batsman_name')[
    ['runs_scored', 'num_out', 'balls_faced', 'num_fours', 'num_sixes', 'num_dots', 'caught', 'not out',
     'leg before wicket', 'bowled', 'hit wicket', 'run out', 'stumped', 'retired not out (hurt)']].agg('sum')
first_inning_bat['average'] = 1.*first_inning_bat['runs_scored']/first_inning_bat['num_out']
first_inning_bat['strike_rate'] = 100.*first_inning_bat['runs_scored']/first_inning_bat['balls_faced']
first_inning_bat['balls_per_six'] = 1.*first_inning_bat['balls_faced']/first_inning_bat['num_sixes']
first_inning_bat['balls_per_boundary'] = 1.*first_inning_bat['balls_faced']/first_inning_bat['num_fours']
for dismissal in [u'caught', u'not out', u'leg before wicket', u'bowled', u'hit wicket', u'run out', u'stumped',
       u'retired not out (hurt)']:
    first_inning_bat[dismissal] = 1.*first_inning_bat[dismissal]/first_inning_bat['num_out']

second_inning_bat = batsman_innings[batsman_innings.inning == 2].groupby('batsman_name')[
    ['runs_scored', 'num_out', 'balls_faced', 'num_fours', 'num_sixes', 'num_dots', 'caught', 'not out',
     'leg before wicket', 'bowled', 'hit wicket', 'run out', 'stumped', 'retired not out (hurt)']].agg('sum')
second_inning_bat['average'] = 1.*second_inning_bat['runs_scored']/second_inning_bat['num_out']
second_inning_bat['strike_rate'] = 100.*second_inning_bat['runs_scored']/second_inning_bat['balls_faced']
second_inning_bat['balls_per_six'] = 1.*second_inning_bat['balls_faced']/second_inning_bat['num_sixes']
second_inning_bat['balls_per_boundary'] = 1.*second_inning_bat['balls_faced']/second_inning_bat['num_fours']
for dismissal in [u'caught', u'not out', u'leg before wicket', u'bowled', u'hit wicket', u'run out', u'stumped',
       u'retired not out (hurt)']:
    second_inning_bat[dismissal] = 1.*second_inning_bat[dismissal]/second_inning_bat['num_out']

first_inning_bat = all_data[all_data.inning==1].groupby('batsman_name')[['runs_batter', 'balls', 'wicket']].agg('sum')
second_inning_bat = all_data[all_data.inning==2].groupby('batsman_name')[['runs_batter', 'balls', 'wicket']].agg('sum')

first_inning_bat['average'] = 1.*first_inning_bat['runs_batter']/first_inning_bat['wicket']
first_inning_bat['strike_rate'] = 100.*first_inning_bat['runs_batter']/first_inning_bat['balls']

second_inning_bat['average'] = 1.*second_inning_bat['runs_batter']/second_inning_bat['wicket']
second_inning_bat['strike_rate'] = 100.*second_inning_bat['runs_batter']/second_inning_bat['balls']

first_inning_first_6 = all_data[(all_data.inning==1) & (all_data.ovr_range == 'first_6')].groupby('batsman_name')[['runs_batter', 'balls', 'wicket']].agg('sum')
first_inning_middle_9 = all_data[(all_data.inning==1) & (all_data.ovr_range == 'middle_9')].groupby('batsman_name')[['runs_batter', 'balls', 'wicket']].agg('sum')
first_inning_last_5 = all_data[(all_data.inning==1) & (all_data.ovr_range == 'last_5')].groupby('batsman_name')[['runs_batter', 'balls', 'wicket']].agg('sum')

for i in [first_inning_first_6, first_inning_middle_9, first_inning_last_5]:
    i['strike_rate'] = 100.*i['runs_batter']/i['balls']
    i['average'] = 1.*i['runs_batter']/i['wicket']

team_innings.to_csv('summary_csvs/team_innings.csv', encoding='utf-8')
team_year.to_csv('summary_csvs/team_year.csv', encoding='utf-8')
team_batting_pos.to_csv('summary_csvs/team_batting_pos.csv', encoding='utf-8')
batsman_innings.to_csv('summary_csvs/batsman_innings.csv', encoding='utf-8')
batsman_order.to_csv('summary_csvs/batsman_order.csv', encoding='utf-8')
batsman_year.to_csv('summary_csvs/batsman_year.csv', encoding='utf-8')
chasing_under_pressure_team.to_csv('summary_csvs/chasing_under_pressure_team.csv', encoding='utf-8')
first_inning_team.to_csv('summary_csvs/first_inning_team.csv', encoding='utf-8')
second_inning_team.to_csv('summary_csvs/second_inning_team.csv', encoding='utf-8')
first_inning_bat.to_csv('summary_csvs/first_inning_bat.csv', encoding='utf-8')
second_inning_bat.to_csv('summary_csvs/second_inning_bat.csv', encoding='utf-8')
first_inning_first_6.to_csv('summary_csvs/first_inning_first_6.csv', encoding='utf-8')
first_inning_middle_9.to_csv('summary_csvs/first_inning_middle_9.csv', encoding='utf-8')
first_inning_last_5.to_csv('summary_csvs/first_inning_last_5.csv', encoding='utf-8')
bowler_spells.to_csv('summary_csvs/bowler_spells.csv', encoding='utf-8')
bowler_year.to_csv('summary_csvs/bowler_year.csv', encoding='utf-8')