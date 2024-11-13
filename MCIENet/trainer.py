import os
import time
import datetime

import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from MCIENet.utils.helper_func import log_string, saveJson, pretty_dict
from MCIENet.utils.eval_metrics import get_evaluation

def train_model(args, log, model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, patience, improved_threshold=5):
    # args (config)
    data_extractor = args.model['extractor_model']

    # train phase
    best_loss = float('inf')
    wait = 0
    reuslt_ls = []
    epoch_output_ls = []
    for epoch in range(num_epochs):
        epoch_result_dt = {}
        epoch_output_dt = {}
        for phase in ['train', 'val']:
            start = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0

            Label_list = []
            Pred_list = []
            Output_list = []
            Output_prob_list = []

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                if data_extractor == 'transformer':
                    inputs = item[0].to(args.device).long()
                else:
                    inputs = item[0].to(args.device).float()
                labels = item[1].to(args.device).long()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    loss = criterion(output, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # detached data out of the computing graph to enhance GPU efficiency
                    loss, output, labels = loss.detach(), output.detach(), labels.detach()

                    _, preds = torch.max(output, 1)

                    epoch_loss += loss.item() * len(output)
                    epoch_acc += torch.sum(preds == labels.data)

                Pred_list.extend(preds.tolist())
                Label_list.extend(labels.tolist())
                Output_list.extend(output.tolist())
                Output_prob_list.extend(F.softmax(output, dim=1).tolist())
            
            torch.cuda.empty_cache()

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = (epoch_acc.double() / data_size).tolist()

            epoch_result_dt.update({phase:{'loss':epoch_loss, 'acc':epoch_acc, 'timeuse':time.time() - start}})
            epoch_output_dt.update({phase:{'output': Output_list, 
                                           'output(prob)': Output_prob_list,
                                           'pred': Pred_list,
                                           'label': Label_list}}
                                           )


        log_string(log, '\n{} | epoch: {}/{}, train loss: {:.4f}, val_loss: {:.4f} | training time: {:.1f}s, inference time: {:.1f}s'.format(
                                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                    epoch + 1,
                                    num_epochs, 
                                    epoch_result_dt['train']['loss'],
                                    epoch_result_dt['val']['loss'],
                                    epoch_result_dt['train']['timeuse'], 
                                    epoch_result_dt['val']['timeuse']
                                )
                )

        reuslt_ls.append(epoch_result_dt)
        epoch_output_ls.append(epoch_output_dt)

        scheduler.step()

        if round(epoch_loss, improved_threshold) < round(best_loss, improved_threshold):
            log_string(log, f'-> Val Loss decrease from {best_loss:.6f} to {epoch_loss:.6f}, saving model')

            if args.use_state_dict:
                torch.save(model.state_dict(), args.model_file)
            else:
                torch.save(model, args.model_file)

            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
 
        if (epoch % args.eval_freq) == 0:
            epoch_eval_dt = {'epoch': epoch}
            for phase in epoch_output_dt:
                pred_ls = epoch_output_dt[phase]['pred']
                label_ls = epoch_output_dt[phase]['label']


                result_dt, _ = get_evaluation(pred_ls, label_ls, 
                                                [i[1] for i in epoch_output_dt[phase]['output(prob)']])
                
                epoch_eval_dt.update({f'({phase}){k}':v for k,v in result_dt.items()})

            epoch_eval = pd.DataFrame.from_dict([epoch_eval_dt])

            epoch_eval_path = os.path.join(args.output_folder, 'epoch_eval.csv')
            if os.path.exists(epoch_eval_path):
                epoch_eval.to_csv(epoch_eval_path, index=False, header=None, mode='a')
            else:
                epoch_eval.to_csv(epoch_eval_path, index=False, mode='w')

        if wait >= patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break

    if args.save_epoch_out:
        saveJson(epoch_output_dt, os.path.join(args.output_folder, 'epoch_output.json'))

    return reuslt_ls


def test_model(args, log, model, dataloaders_dict, criterion):
    data_extractor = args.model['extractor_model']

    if args.use_state_dict:
        model.load_state_dict(torch.load(args.model_file))
    else:
        model = torch.load(args.model_file)

    model.eval()
    model = model.to(args.device)

    evaluation_dt = {}
    df_pred = pd.DataFrame()
    with torch.no_grad():
        for phase in ['train', 'val', 'test']:
            start = time.time()

            epoch_loss = 0.0
            epoch_acc = 0

            Pred_list = []
            Label_list = []
            Output_list = []
            Output_prob_list = []

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                if data_extractor == 'transformer':
                    inputs = item[0].to(args.device).long()
                else:
                    inputs = item[0].to(args.device).float()
                labels = item[1].to(args.device).long()

                output = model(inputs)
                loss = criterion(output, labels)
                _, preds = torch.max(output, 1)

                epoch_loss += loss.item() * len(output)
                epoch_acc += torch.sum(preds == labels.data)

                Pred_list.extend(preds.tolist())
                Label_list.extend(labels.tolist())
                Output_list.extend(output.tolist())
                Output_prob_list.extend(F.softmax(output, dim=1).tolist())

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = (epoch_acc.double() / data_size).tolist()

            df = pd.DataFrame(
                {'Phase': [phase]*len(Pred_list), 'Label_idx':Label_list, 'Predict_idx': Pred_list,
                 'Output': [i[1] for i in Output_list], 
                 'Output(prob)': [i[1] for i in Output_prob_list]}
            )
            
            if args.save_pred_result:
                df_pred = pd.concat([df_pred, df], ignore_index=True)

            phase_eval_dt = {'loss':epoch_loss, 'acc':epoch_acc, 'timeuse':time.time() - start} 

            result_dt, _ = get_evaluation(Pred_list, Label_list, df['Output(prob)'].values)

            evaluation_dt.update({phase:{**phase_eval_dt, **result_dt}})

        if args.save_pred_result:
            df_pred['Label'] = df_pred['Label_idx'].apply(lambda x: "no loop" if x == 0 else "loop")
            df_pred['Predict'] = df_pred['Predict_idx'].apply(lambda x: "no loop" if x == 0 else "loop")

            df_pred.to_csv(os.path.join(args.output_folder, 'pred_and_label.csv'), index=False)
        
        saveJson(evaluation_dt, os.path.join(args.output_folder, 'evaluation.json'))

        for phase in evaluation_dt:
            res = evaluation_dt[phase]
            log_string(log, f'[{phase}]\n' + pretty_dict(res, decimal=4))