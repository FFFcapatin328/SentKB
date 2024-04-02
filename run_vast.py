import os
import socket
import nni
from nni.experiment import Experiment

if __name__ == '__main__':
    ''' NNI framework
    experiment = Experiment('local')
    search_space = {
        'batch_size': {'_type': 'choice', '_value': [8, 16]},
        'lr': {'_type': 'loguniform', '_value': [1e-6, 0.1]},
        'l2_reg': {'_type': 'loguniform', '_value': [1e-6, 0.1]},
        'n_layers_freeze': {'_type': 'choice', '_value': [0, 5, 10, 15]},
        'seed': {'_type': 'randint', '_value': [0, 200]},
    }'''

    data = ['vast'][0]
    topic = ''
    # batch_size = 16
    batch_size=16
    epochs = 50
    patience = 10
    # lr = 2e-5
    # l2_reg = 5e-5
    lr = 0.000008142587485342115
    l2_reg = 0.000001032344354072693
    model = ['bert-base'][0]
    wiki_model = ['', 'bert-base'][1]
    # n_layers_freeze = 10
    n_layers_freeze = 5
    n_layers_freeze_wiki = 0
    gpu = '0'
    inference = 0
    senti_flag = True
    graph_flag = True
    summa_flag = True

    if wiki_model == model:
        n_layers_freeze_wiki = n_layers_freeze
    if not wiki_model or wiki_model == model:
        n_layers_freeze_wiki = 0

    os.makedirs('train_logs', exist_ok=True)
    if data != 'vast':
        file_name = f'train_logs/{data}-topic={topic}-lr={lr}-bs={batch_size}.txt'
    else:
        file_name = f'train_logs/{data}-lr={lr}-bs={batch_size}.txt'

    if model != 'bert-base':
        file_name = file_name[:-4] + f'-{model}.txt'
    if n_layers_freeze > 0:
        file_name = file_name[:-4] + f'-n_layers_fz={n_layers_freeze}.txt'
    if wiki_model:
        file_name = file_name[:-4] + f'-wiki={wiki_model}.txt'
    if n_layers_freeze_wiki > 0:
        file_name = file_name[:-4] + f'-n_layers_fz_wiki={n_layers_freeze_wiki}.txt'
    if senti_flag:
        file_name = file_name[:-4] + f'-senti_flag={senti_flag}.txt'
    if summa_flag:
        file_name = file_name[:-4] + f'-summa_flag={summa_flag}.txt'
    if graph_flag:
        file_name = file_name[:-4] + f'-graph_flag={graph_flag}.txt'

    n_gpus = len(gpu.split(','))
    file_name = file_name[:-4] + f'-n_gpus={n_gpus}.txt'
    
    command = f"CUDA_VISIBLE_DEVICES={gpu} " \
              f"python -u src/train.py " \
              f"--data={data} " \
              f"--topic={topic} " \
              f"--model={model} " \
              f"--n_layers_freeze={n_layers_freeze} " \
              f"--n_layers_freeze_wiki={n_layers_freeze_wiki} " \
              f"--batch_size={batch_size} " \
              f"--epochs={epochs} " \
              f"--patience={patience} " \
              f"--lr={lr} " \
              f"--l2_reg={l2_reg} " \
              f"--gpu={gpu} " \
              f"--inference={inference} " \
              f"--wiki_model={wiki_model} " \
              f"--senti_flag={senti_flag} " \
              f"--summa_flag={summa_flag} " \
              f"--graph_flag={graph_flag} " 
    
    ''' NNI framework
    experiment.config.trial_command = command
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space

    experiment.config.tuner.name = 'Random'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 100 # 运行30次实验
    experiment.config.trial_concurrency = 1

    experiment.run(8080)
    input('Press enter to quit')
    experiment.stop()
    '''
    print(command)
    os.system(command)
