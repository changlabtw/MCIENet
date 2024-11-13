"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2024.05.07
Last Update: 2024.05.13
"""
import os
import json
import argparse

import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")

import matplotlib.pyplot as plt
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=str, nargs='+', required=True)
    parser.add_argument('--names', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--folders_gp_name', type=str,  default='Name')

    parser.add_argument('--box_group', nargs='+', type=str, default=None)

    parser.add_argument('--criterion', type=str, default='auPRCs')

    # sort group args table 
    parser.add_argument('--save_metrics', nargs='+', type=str, default=None)
    parser.add_argument('--sort_args_group', nargs='+', type=str, default=None)

    args = parser.parse_args()

    return args

def saveJson(data, path):
    with open(path, 'w', encoding='utf-8') as outfile:  
        json.dump(data, outfile, indent=2, ensure_ascii=False)

def box_plot(df, y, hue, path):
    gp_num = len(df[hue].unique())
    df[hue] = df[hue].astype(str).str.strip(hue)
    try:
        df[hue] = df[hue].astype(int)
    except:
        pass
    df.sort_values(hue, inplace=True)

    # plot
    plt.clf()
    _, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(data=df, x="phase", y=y, ax=ax,
                hue=hue, legend='full', 
                notch=True, palette=sns.color_palette('Set3')[:gp_num])
    plt.tight_layout()
    plt.savefig(path)

def compare_result(results_path:list, output_path:str, names:list, criterion:str, box_group:list[str], folders_gp_name:str, top_n:int = 10) -> pd.DataFrame:

    assert len(names) == len(results_path), 'Parameters --names & --results_path have mismatched lengths'

    df_all = pd.DataFrame()
    for name, path in zip(names, results_path):
        result_csv = os.path.join(path, 'result.csv')
        if not os.path.exists(result_csv):
            call_back = os.system(f"python MCIENet\helper_scripts\sort_exp_result.py --folder {path}")
            # assert call_back == 0, "There is a problem when running script -> MCIENet\helper_scripts\sort_exp_result.py"
        
        df = pd.read_csv(result_csv)
        df['Name'] = name
        df = df[['Name'] + df.columns[:-1].tolist()]

        path = os.path.join(path, 'parms_result.csv')
        if os.path.exists(path):
            df_parm = pd.read_csv(path)
            df_parm.drop('experiment', axis=1, inplace=True)
            df = pd.merge(df, df_parm, how='left')

        df_all = pd.concat([df_all, df])

    df_all.rename(columns={'Name': folders_gp_name}, inplace=True)
    os.makedirs(output_path, exist_ok=True)
    df_all.to_csv(os.path.join(output_path, 'result_all.csv'), index=False)

    # BEST
    best_dt = {}
    for group in names:
        tmp = df_all[df_all[folders_gp_name] == group]
        val_result = tmp[tmp['phase'] == 'val']
        val_result = val_result.sort_values(criterion, ascending=False)
        best = val_result['experiment'].tolist()[0]
        best_exp = tmp[tmp['experiment'] == best]
        best_exp = best_exp.set_index('phase')
        best_dt[group] = best_exp.to_dict('index')

    saveJson(best_dt, os.path.join(output_path, 'best_result_gp.json'))


    # different group(folder + name) compare boxplot
    box_plot(df=df_all, y=criterion, hue=folders_gp_name, 
             path=os.path.join(output_path, 'compare_boxplot.png'))


    # other args compare boxplot
    if box_group != None:
        for i in box_group:
            assert i in df_all.columns, '--box_group not found in result args'

            box_plot(
                df=df_all, y=criterion, hue=i, 
                path=os.path.join(output_path, f'compare_boxplot({i}).png')
                )

            if len(names) > 1:
                for group in names:
                    box_plot(
                        df=df_all[df_all[folders_gp_name] == group], 
                        y=criterion, hue=i, 
                        path=os.path.join(output_path, f'compare_boxplot_{group}({i}).png')
                        )

    if len(df_all[folders_gp_name].unique()) > 1:
        df_all['experiment'] = df_all['experiment'] + '__' + df_all[folders_gp_name].astype(str)

    # Top N
    val_all = df_all[df_all['phase'] == 'val']
    val_all = val_all.sort_values(criterion, ascending=False)

    top_ls = val_all['experiment'].tolist()[:top_n]
    top_df = df_all[df_all['experiment'].isin(top_ls)]

    top_df.to_csv(os.path.join(output_path, f'result_top{top_n}.csv'), index=False)

    return df_all, top_df

def sort_gp_result(df, top_df, sort_args, target_metric, save_metric, output_path:str):


    val_set = df[df['phase'] == 'val']
    test_set = df[df['phase'] == 'test']

    top_n = round(top_df.shape[0]/3)

    result = []
    for arg in sort_args:
        for gp in val_set[arg].unique():
            test_gp = test_set.loc[test_set[arg] == gp]
            val_gp = val_set.loc[val_set[arg] == gp]

            # get best result
            best_experiment = val_gp.sort_values(target_metric, ascending=False)['experiment'].values[0]

            for phase in ['val', 'test']:
                tar_set = test_gp if phase == 'test' else val_gp
                # get statistics of target metric
                avg = tar_set[target_metric].mean()
                mid = tar_set[target_metric].median()

                top_n_rate = (top_df[top_df['experiment'].isin(test_gp['experiment'])].shape[0]/3) / top_n
                
                result_dt = {'arg': arg, 'test_values': str(gp).replace(f'{arg}', ''), 
                            'phase': phase, 'avg': avg, 'median': mid, 
                            'best_experiment': best_experiment,
                            f'top{str(top_n)} rate': top_n_rate
                            }

                # get best metric
                result_df = tar_set[tar_set['experiment'] == best_experiment]
                result_dt.update({i:result_df[i].values[0] for i in save_metric})
                result.append(result_dt)

    pd.DataFrame(result).to_csv(os.path.join(output_path, 'group_sort_result.csv'), index=False)

def main():
    args = get_args()
    df_all, top_df = compare_result(args.folders, args.output, args.names, 
                   args.criterion, args.box_group, args.folders_gp_name)
    
    sort_gp_result(df_all, top_df, args.sort_args_group, args.criterion, args.save_metrics, args.output)

if __name__ == '__main__':
    main()