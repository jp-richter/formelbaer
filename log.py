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

    num_samples = cfg.general.num_real_samples * 2  # real + synthetic
    num_batches = math.ceil(num_samples / cfg.general.batch_size)
    average_loss = nn_discriminator.running_loss / num_batches
    average_acc = nn_discriminator.running_acc / (num_batches * cfg.general.batch_size)  # loss already averaged

    print('Epoch {} D Epoch {} Loss {} Acc {}'.format(epoch, d_epoch, average_loss, average_acc))
    log.info('Epoch {} D Epoch {} Loss {} Acc {}'.format(epoch, d_epoch, average_loss, average_acc))

    discriminator_loss_sequence.append(average_loss)

    nn_discriminator.running_loss = 0.0
    nn_discriminator.running_acc = 0.0


def generator_loss(nn_policy, epoch):
    global log

    average_loss = nn_policy.running_loss / cfg.general.g_steps

    print('Epoch {} G         Loss {}'.format(epoch, average_loss))
    log.info('Epoch {} G         Loss {}'.format(epoch, average_loss))

    generator_loss_sequence.append(average_loss)

    nn_policy.running_loss = 0.0
    nn_policy.save(cfg.paths.policies + '/' + str(epoch) + '.pt')


def start_experiment():
    global log

    if log is None:
        init()

    print('Starting experiment..')

    log.info('''STARTING EXPERIMENT
    
        Added Multipages

        Total Epochs {}
        Discriminator Steps {}
        Discriminator Epochs {}
        Generator Steps {}
        
        Fixed Sequence Length {}
        Monte Carlo Trials {}
        Batch Size {}
        Batch Size Multiplier {}

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
        

        '''.format(
            cfg.general.total_epochs,
            cfg.general.d_steps,
            cfg.general.d_epochs,
            cfg.general.g_steps,
            cfg.general.sequence_length,
            cfg.general.montecarlo_trials,
            cfg.general.batch_size,
            cfg.general.g_batchsize_multiplier,

            cfg.generator.hidden_dim,
            cfg.generator.layers,
            cfg.generator.dropout,
            cfg.generator.learnrate,
            cfg.generator.baseline,
            cfg.generator.gamma,

            cfg.discriminator.dropout,
            cfg.discriminator.learnrate,

            cfg.general.oracle,
            cfg.general.num_real_samples))


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
    shutil.copytree(cfg.paths.policies, directory + '/policies')
