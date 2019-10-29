import config as cfg
import logging
import shutil
import math


log = None

generator_loss_sequence = []
discriminator_loss_sequence = []
oracle_score_sequence = []


def init():
    global log

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    handler = logging.FileHandler(cfg.paths.log, mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    log.addHandler(handler)


def start_loading_data():
    global log

    if log is None:
        init()

    log.info('Start loading data..')
    print('Start loading data..')


def finish_loading_data():
    global log 

    log.info('Data successfully loaded.')
    print('Data successfully loaded.')


def discriminator_loss(nn_discriminator, epoch, d_epoch):
    global log

    num_samples = cfg.general.size_real_dataset * 2  # real + synthetic
    num_batches = math.ceil(num_samples / cfg.general.batch_size)
    average_loss = nn_discriminator.running_loss / num_batches
    average_acc = nn_discriminator.running_acc / (num_batches * cfg.general.batch_size)  # loss already averaged

    print('Epoch {} Discriminator Epoch {} Average Loss {} Train Acc {}'.format(epoch, d_epoch, average_loss, average_acc))
    log.info('Epoch {} Discriminator Epoch {} Average Loss {} Train Acc {}'.format(epoch, d_epoch, average_loss, average_acc))

    nn_discriminator.running_loss = 0.0
    nn_discriminator.running_acc = 0.0


def generator_reward(nn_policy, epoch):
    global log

    average_reward = nn_policy.running_reward / cfg.general.g_steps

    print('Epoch {} Generator Average Reward {}'.format(epoch, average_reward))
    log.info('Epoch {} Generator Average Reward {}'.format(epoch, average_reward))

    nn_policy.running_reward = 0.0


def start_experiment():
    global log

    if log is None:
        init()

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
        Real Data Sample Size {}
        Recycling {}

        '''.format(
            cfg.general.iterations,
            cfg.general.d_steps,
            cfg.general.g_steps,
            cfg.general.sequence_length,
            cfg.general.montecarlo_trials,
            cfg.general.batch_size,

            cfg.generator.hidden_dim,
            cfg.generator.layers,
            cfg.generator.dropout,
            cfg.generator.learnrate,
            cfg.generator.baseline,
            cfg.generator.gamma,

            cfg.discriminator.dropout,
            cfg.discriminator.learnrate,

            cfg.general.oracle,
            cfg.general.size_real_dataset,
            cfg.general.recycling))


def adversarial(iteration, nn_generator, nn_discriminator, nn_oracle, printout=False):
    global log

    g_reward = nn_generator.running_reward / (iteration * cfg.general.g_steps)
    o_loss = nn_oracle.running_score / (iteration * cfg.general.g_steps)
    d_loss = nn_discriminator.running_loss / (iteration * cfg.general.d_steps)

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
    oracle_score_sequence.append(nn_oracle.running_score)

    nn_generator.running_reward = 0.0
    nn_discriminator.running_loss = 0.0
    nn_oracle.running_score = 0.0


def finish_experiment(directory):
    global log, generator_loss_sequence, discriminator_loss_sequence, oracle_score_sequence

    print('Finishing experiment..')

    log.info('''FINISHING EXPERIMENT

        ''')

    generator_loss_sequence_str = ', '.join(map(str, generator_loss_sequence))
    discriminator_loss_sequence_str = ', '.join(map(str, discriminator_loss_sequence))
    oracle_score_sequence_str = ', '.join(map(str, oracle_score_sequence))
    log.info('Generator Loss as Sequence: ' + generator_loss_sequence_str)
    log.info('Discriminator Loss as Sequence: ' + discriminator_loss_sequence_str)
    log.info('Oracle Loss as Sequence: ' + oracle_score_sequence_str)

    shutil.copyfile(cfg.paths.log, directory + '/results.log')
