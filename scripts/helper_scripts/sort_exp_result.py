"""
Author: yen-nan ho
Contact: aaron1aaron2@gmail.com
Create Date: 2023.10.10
Last Update: 2023.11.26
"""

import os
import re
import json
import argparse

import pandas as pd
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    # 輸入
    parser.add_argument('--folder', type=str, default='output/GM12878_CTCF/hyper')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    return args

def paser_experiment(df:pd.DataFrame, col:str='experiment') -> pd.DataFrame:
    par_df = df[col].str.split('_', expand=True)

    new_cols = []
    for i in par_df.columns:
        pat = re.search(r'[a-zA-Z]+', par_df.loc[0, i])
        if pat != None:
            new_cols.append(pat.group(0))
        else:
            new_cols.append(f'parma{i}')


    par_df.columns = new_cols
    for i in new_cols:
        par_df[i] = par_df[i].str.replace(f'^{i}-?', '', regex=True)


        first_item = par_df[i].tolist()[0]
        if first_item.isdigit():
            par_df[i] = par_df[i].astype(int)
        elif first_item.replace('.','',1).isdigit():
            par_df[i] = par_df[i].astype(float)

    return par_df

def sort_result(folder:str, output:str=None, target:str='evaluation.json') -> None:
    # evaluation result
    result_df = pd.DataFrame()
    for path in Path(folder).rglob(target):
        exp_name = path.parent.name
        try:
            data = json.load(open(path))
        except:
            raise ValueError(f'Invalid json file -> {path}')

        tmp = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'phase'})
        tmp = pd.DataFrame([exp_name]*len(tmp), columns=['experiment']).join(tmp)

        result_df = pd.concat([result_df, tmp])

    result_df.reset_index(drop=True, inplace=True)

    par_df = paser_experiment(result_df, 'experiment')

    result_df = par_df.join(result_df)

    if output == None:
        output = os.path.join(folder, 'result.csv')

    result_df.to_csv(output, index=False)
    parm_list = list(Path(folder).rglob('parms_count.json'))

    if len(parm_list) != 0:
        # parms sort
        parms_dt_ls = []
        for path in parm_list:
            exp_name = path.parent.name
            data = json.load(open(path))
            data['experiment'] = exp_name
            parms_dt_ls.append(data)

        parm_df = pd.DataFrame(parms_dt_ls)

        par_df = paser_experiment(parm_df, 'experiment')

        parm_df = par_df.join(parm_df)

        output = output.replace('result.csv', 'parms_result.csv')

        parm_df.to_csv(output, index=False)
    else:
        print(f'Not Found parms_count.json, skip parms_result.csv')
    
def main():
    args = get_args()
    sort_result(args.folder, args.output)

if __name__ == '__main__':
    main()