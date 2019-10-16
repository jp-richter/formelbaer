import config as cfg
import logging
import shutil
import os


if os.path.exists(cfg.paths_cfg.log):
    os.remove(cfg.paths_cfg.log)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename=cfg.paths_cfg.log)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

generator_loss_sequence = []
discriminator_loss_sequence = []
oracle_loss_sequence = []


def start_loading_data():
    global log 

    log.info('Start loading data..')
    print('Start loading data..')


def finish_loading_data():
    global log 

    log.info('Data successfully loaded.')
    print('Data successfully loaded.')


def start_experiment():
    global log

    print('Starting experiment..')

    log.info('''STARTING EXPERIMENT

        Models Version: 1
        Systems Version: 1

        Total Iterations {}
        Discriminator Steps {}
        Generator Steps {}
        Fixed Sequence Length {}
        Monte Carlo Trials {}
        Batch Size {}

        Generator Hidden Dim {}
        Generator Layers {}
        Generator Dropout {}
        Generator Learning Rate {}
        Generator Baseline {}
        Generator Gamma {}

        Discriminator Dropout {}
        Discriminator Learnrate {}

        Oracle Use {}
        Oracle Sample Size {}

        '''.format(
            cfg.app_cfg.iterations, 
            cfg.app_cfg.d_steps, 
            cfg.app_cfg.g_steps, 
            cfg.app_cfg.seq_length, 
            cfg.app_cfg.montecarlo_trials, 
            cfg.app_cfg.batchsize, 

            cfg.g_cfg.hidden_dim, 
            cfg.g_cfg.layers,  
            cfg.g_cfg.dropout, 
            cfg.g_cfg.learnrate, 
            cfg.g_cfg.baseline, 
            cfg.g_cfg.gamma, 

            cfg.d_cfg.dropout, 
            cfg.d_cfg.learnrate, 

            cfg.app_cfg.oracle, 
            cfg.app_cfg.oracle_samplesize))


def write(iteration, nn_generator, nn_discriminator, nn_oracle, printout=False):
    global log

    g_reward = nn_generator.running_reward / (iteration * cfg.app_cfg.g_steps)
    o_loss = nn_oracle.running_loss / (iteration * cfg.app_cfg.g_steps)
    d_loss = nn_discriminator.running_loss / (iteration * cfg.app_cfg.d_steps)

    entry = '''###
        Iteration {iteration}
        Generator Reward {greward}
        Discriminator Loss {dloss}
        Oracle Loss {oloss}

        '''.format(iteration=iteration, greward=g_reward, dloss=d_loss, oloss=o_loss)

    log.info(entry)

    if printout: print(entry)

    generator_loss_sequence.append(nn_generator.running_reward)
    discriminator_loss_sequence.append(nn_discriminator.running_loss)
    oracle_loss_sequence.append(nn_oracle.running_loss)

    nn_generator.running_reward = 0.0
    nn_discriminator.running_loss = 0.0
    nn_oracle.running_loss = 0.0


def finish_experiment(directory):
    global log, generator_loss_sequence, discriminator_loss_sequence, oracle_loss_sequence

    print('Finishing experiment..')

    log.info('''FINISHING EXPERIMENT

        ''')

    generator_loss_sequence = ', '.join(map(str, generator_loss_sequence))
    discriminator_loss_sequence = ', '.join(map(str, discriminator_loss_sequence))
    oracle_loss_sequence = ', '.join(map(str, oracle_loss_sequence))
    log.info('Generator Loss as Sequence: ' + generator_loss_sequence)
    log.info('Discriminator Loss as Sequence ' + discriminator_loss_sequence)
    log.info('Oracle Loss as Sequence ' + oracle_loss_sequence)

    shutil.copyfile(cfg.paths_cfg.log, directory + '/results.log')
