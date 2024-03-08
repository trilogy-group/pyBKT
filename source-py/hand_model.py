import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import random
import sys

from datetime import datetime
from pyBKT.models import Model
from _collections import defaultdict

pd.set_option('display.max_rows', None)

PARAMS = {
    'prior': 0.55,
    'learns': np.array([0.15]),
    'guesses': np.array([0.1]),
    'slips': np.array([0.1]),
    'forgets': np.array([0.03])
    }

start = datetime.now()
step = start

def mae(true_vals, pred_vals):
  """ Calculates the mean absolute error. """
  return np.mean(np.abs(true_vals - pred_vals))

def timeprint(msg):
    global start, step
    now = datetime.now()
    print(msg, now-step, now-start)
    step = datetime.now()

def get_unique_skills_from_df(df):
    return list(set(df.skill_name))

def get_skills_from_file(data_file):
    return get_unique_skills_from_df(pd.read_csv(data_file))

def get_mastery_prediction_from_predictions(preds_df):
    df_sorted = preds_df.sort_values(by=['user_id', 'skill_name', 'order_id'])
    last_entries = df_sorted.groupby(['user_id', 'skill_name'])['state_predictions'].last()
    user_skill_to_state = last_entries.unstack().apply(lambda x: x.dropna().to_dict(), axis=1).to_dict()
    return user_skill_to_state

def print_params(result_dict):
    print('Params:')
    for param in result_dict['params']:
        val = result_dict['params'][param]
        val = val[0] if isinstance(val, np.ndarray) else val
        print(f'\t{param} = {val}')
        

def print_user_skill_mastery(result_dict):
    for user in sorted(result_dict["user_skill_mastery"]):
        print('User', user)
        for skill in result_dict["user_skill_mastery"][user]:
            print("\tSkill", skill, result_dict["user_skill_mastery"][user][skill])

def print_eval_scores(result_dict):
    print("AUC:\t%f" % result_dict["auc"])
    print("RMSE:\t%f" % result_dict["rmse"])
    print("MAE:\t%f" % result_dict["mae"])

def print_stats(result_dict):
    print_params(result_dict)
    print_user_skill_mastery(result_dict)
    print_eval_scores(result_dict)

def evaluate_params(data_file, params):
    model = Model(seed=42, num_fits=5)
    model.skills = get_skills_from_file(data_file)
    coefs = {}
    for skill in model.skills:
        coefs[skill] = params
    model.coef_ = coefs
    preds_df = model.predict(data_path=data_file)
    user_skill_mastery = get_mastery_prediction_from_predictions(preds_df)
    auc = model.evaluate(data_path=data_file, metric='auc')
    rmse = 0 #model.evaluate(data_path=data_file, metric='rmse')
    mae_error = 0 #model.evaluate(data_path=data_file, metric=mae)
    result = {
        'rmse': rmse,
        'auc': auc,
        'mae': mae_error,
        'params': params,
        'user_skill_mastery': user_skill_mastery
    }
    return result, preds_df
        

def generate_random_params():
    prior = random.uniform(0.01, 0.8)
    learns = random.uniform(0.01, 0.8)
    slips = random.uniform(0.1, min(learns, 0.5))
    guesses = random.uniform(0.1, min(learns, 1-slips, 0.5))
    forgets = random.uniform(0.1, min(guesses, slips))
    return {
        'prior': prior,
        'learns': np.array([learns]),
        'guesses': np.array([slips]),
        'slips': np.array([guesses]),
        'forgets': np.array([forgets])
    }

def save_mastery_plot(i, df):
    plt.figure(figsize=(14, 6))

    for user_id, group in df.groupby('user_id'):
        group = group.sort_values(by='problem_id')
        
        plt.plot(group['problem_id'], group['state_predictions'], label=f'User {user_id}', marker='', linestyle='-')
        
        correct_answers = group[group['correct'] == 1]
        plt.scatter(correct_answers['problem_id'], correct_answers['state_predictions'], color='green', marker='o', s=100, label='_nolegend_')
        
        incorrect_answers = group[group['correct'] == 0]
        plt.scatter(incorrect_answers['problem_id'], incorrect_answers['state_predictions'], color='red', marker='x', s=100, label='_nolegend_')

    plt.xlabel('Problem')
    plt.ylabel('Mastery')
    plt.title('Student Mastery Estimation Over Problems')
    all_problem_ids = df['problem_id'].unique()
    plt.xticks(range(min(all_problem_ids), max(all_problem_ids) + 1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    filename = f'plots/mastery_{i}.png'
    plt.savefig(filename)

    plt.close()

def new_result_better(best_result, new_result):
    return new_result['auc'] > best_result['auc']


def main(data_file):
    best_params = None
    best_result = {'rmse': 0, 'auc': 0, 'mae': 0}
    best_iter = 0
    iterations = 1000
    for i in range(iterations):
        timeprint(f'Param testing iteration {i+1} of {iterations}...')
        if i == 0:
            new_params = PARAMS
        else:
            new_params = generate_random_params()
        new_result, preds_df = evaluate_params(data_file, new_params)
        if new_result_better(best_result, new_result):
            print('New best result!')
            print_stats(new_result)
            best_params = new_params
            best_result = new_result
            best_iter = i+1
        else:
            print_eval_scores(new_result)
        save_mastery_plot(i+1, preds_df)
        timeprint(f'Iteration {i+1} complete.')
    timeprint('All iterations complete!')
    print('\nBest result:')
    print(f'Iteration {best_iter}')
    print_stats(best_result)
    

if __name__ == '__main__':
    data_file = sys.argv[1]
    main(data_file)