 # -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np

batsman_innings = pd.read_csv('summary_csvs/batsman_innings.csv')
batsman_order = pd.read_csv('summary_csvs/batsman_order.csv')
batsman_year = pd.read_csv('summary_csvs/batsman_year.csv')
bowler_spells = pd.read_csv('summary_csvs/bowler_spells.csv')
bowler_year = pd.read_csv('summary_csvs/bowler_year.csv')
chasing_under_pressure_team = pd.read_csv('summary_csvs/chasing_under_pressure_team.csv')
first_inning_bat = pd.read_csv('summary_csvs/first_inning_bat.csv')
first_inning_first_6 = pd.read_csv('summary_csvs/first_inning_first_6.csv')
first_inning_last_5 = pd.read_csv('summary_csvs/first_inning_last_5.csv')
first_inning_middle_9 = pd.read_csv('summary_csvs/first_inning_middle_9.csv')
first_inning_team = pd.read_csv('summary_csvs/first_inning_team.csv')
second_inning_bat = pd.read_csv('summary_csvs/second_inning_bat.csv')
second_inning_team = pd.read_csv('summary_csvs/second_inning_team.csv')
team_batting_pos = pd.read_csv('summary_csvs/team_batting_pos.csv')
team_innings = pd.read_csv('summary_csvs/team_innings.csv')
team_year = pd.read_csv('summary_csvs/team_year.csv')

for i in ['num_out', 'runs_scored', 'opposition_score']:
	team_innings[i] = team_innings[i].astype(int)

team_batting_pos['balls_per_six'] = 1.*team_batting_pos['balls_faced']/team_batting_pos['num_sixes']
team_batting_pos['balls_per_boundary'] = 1.*team_batting_pos['balls_faced']/(team_batting_pos['num_sixes'] + team_batting_pos['num_fours'])

app = Flask(__name__, static_folder = 'assets')
@app.route('/assets/<path:path>')
def serve_static(path):
    root_dir = os.path.dirname(os.getcwd())
    return app.send_static_file(os.path.join(root_dir, 'assets', path))

@app.route('/', methods=['GET'])
def input():
    return render_template('index.html')

@app.route('/batsman', methods=['GET'])
def batsman():
    #input contains a simple form where we ask users whose stats they want to see 
    name = request.args.get('name', None)
    name = name.title()
    return_obj = {
        'all_innings': batsman_innings[batsman_innings.batsman_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'batsman_year': batsman_year[batsman_year.batsman_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'performance_by_order': batsman_order[batsman_order.batsman_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'first_inning_bat': first_inning_bat[first_inning_bat.batsman_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'second_inning_bat': second_inning_bat[second_inning_bat.batsman_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'first_inning_first_6': first_inning_bat[first_inning_bat.batsman_name.str.contains(name)].to_dict('records'),
        'first_inning_middle_9': first_inning_middle_9[first_inning_middle_9.batsman_name.str.contains(name)].to_dict('records'),
        'first_inning_last_5': first_inning_last_5[first_inning_last_5.batsman_name.str.contains(name)].to_dict('records'),
        'all_cols': ['batsman_name', 'date', 'opposition', 'inning', 'batting_order', 'wicket_method', 'balls_faced',
        'runs_scored', 'num_fours', 'num_sixes', 'strike_rate'],
        'innings_cols': ['batsman_name', 'average', 'strike_rate'],
        'year_cols': ['batsman_name', 'year', 'num_innings', 'strike_rate', 'average',
        'balls_per_six', 'balls_per_boundary', 'prop_dot'],
        'pos_cols': ['batsman_name', 'batting_order', 'num_innings', 'strike_rate', 'average',
        'balls_per_six', 'balls_per_six', 'prop_dot'],
        'over_cols': ['batsman_name', 'average', 'strike_rate']
        }
    if len(return_obj['all_innings']) == 0:
    	return 'not found'
    for i in ['first_inning_bat', 'second_inning_bat', 'first_inning_first_6',
    'first_inning_middle_9', 'first_inning_last_5']:
	    if len(return_obj[i]) == 0:
	    	del return_obj[i]
    return render_template('batsman.html', name=name, dets=return_obj)

@app.route('/bowler', methods=['GET'])
def bowler():
    name = request.args.get('name', None)
    name = name.title()
    return_obj = {
        'all_spells': bowler_spells[bowler_spells.bowler_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'all_cols': ['bowler_name', 'date', 'opposition', 'balls_bowled', 'runs_conceded', 'wickets_taken', 'economy_rate'],
        'bowler_year': bowler_year[bowler_year.bowler_name.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'year_cols': ['bowler_name', 'year', 'num_spells', 'balls_bowled', 'wickets_taken', 'runs_conceded',
        'average', 'strike_rate', 'prop_dot', 'wickets_per_spell']
    }
    return render_template('bowler.html', name=name, dets=return_obj)

@app.route('/team', methods=['GET'])
def team():
    name = request.args.get('name', None)
    name = name.title()
    return_obj = {
        'first_inning_team': first_inning_team[first_inning_team.batting_team.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'second_inning_team': second_inning_team[second_inning_team.batting_team.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'team_innings': team_innings[team_innings.batting_team.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'team_batting_pos': team_batting_pos[team_batting_pos.batting_team.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'team_year': team_year[team_year.batting_team.str.contains(name)].replace(np.inf,0).to_dict('records'),
        'team_innings_cols': ['date', 'opposition', 'inning', 'win', 'num_out', 'balls_faced', 'runs_scored',
        'opposition_score', 'num_dots', 'num_fours', 'num_sixes', 'run_rate'],
        'innings_cols': ['total_matches', 'win_rate', 'run_rate', 'average',
        'balls_per_six', 'balls_per_boundary'],
        'team_year_cols': ['year', 'num_games', 'win_percentage', 'balls_per_six', 'balls_per_boundary',
        'run_rate'],
        'team_pos_cols': ['batting_order', 'strike_rate', 'average', 'balls_per_six', 'balls_per_boundary']
    }
    return render_template('team.html', name=name, dets=return_obj)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)