import config as cfg
import logging
import shutil
import plotter

log = None

generator_loss_sequence = []
generator_reward_sequence = []
generator_prediction_sequence = []
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


def discriminator_loss(nn_discriminator, epoch, d_epoch):
    global log

    average_loss = nn_discriminator.running_loss / max(nn_discriminator.loss_divisor,1)
    average_acc = nn_discriminator.running_acc / max(nn_discriminator.acc_divisor,1)

    msg = '''---
Epoch {} D Epoch {}
Loss {}
Acc {}'''.format(epoch, d_epoch, average_loss, average_acc)

    print(msg)
    log.info(msg)

    discriminator_loss_sequence.append(average_loss)

    nn_discriminator.running_loss = 0.0
    nn_discriminator.loss_divisor = 0
    nn_discriminator.running_acc = 0.0
    nn_discriminator.acc_divisor = 0


def generator_loss(nn_policy, epoch, g_step):
    global log

    average_loss = nn_policy.running_loss / max(nn_policy.loss_divisor,1)
    average_reward = nn_policy.running_reward / max(nn_policy.reward_divisor,1)
    average_prediction = nn_policy.running_prediction / max(nn_policy.prediction_divisor, 1)

    print(average_reward)
    print(average_prediction)

    msg = '''---
Epoch {} G Step {}
Reward {}
Prediction {}
Loss {}'''.format(epoch, g_step, average_reward, average_prediction, average_loss)

    print(msg)
    log.info(msg)

    generator_loss_sequence.append(average_loss)
    generator_prediction_sequence.append(average_prediction)
    generator_reward_sequence.append(average_reward)

    nn_policy.running_loss = 0.0
    nn_policy.loss_divisor = 0
    nn_policy.running_reward = 0.0
    nn_policy.reward_divisor = 0
    nn_policy.running_prediction = 0.0
    nn_policy.prediction_divisor = 0
    nn_policy.save(cfg.paths.policies + '/' + str(epoch) + '.pt')


def start_experiment():
    global log

    if log is None:
        init()

    print('Starting experiment..')

    log.info('''STARTING EXPERIMENT
    
        0.1 Added Multipages
        
        0.2 Resetting D Weights After Each Step
            Added Bias Initialization
            
        0.3 Set Batchsize to cores
        
        0.4 Switched to update policy after each step in a sequence
        

        Total Epochs {}
        Discriminator Steps {}
        Discriminator Epochs {}
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
        

        '''.format(
            cfg.general.total_epochs,
            cfg.general.d_steps,
            cfg.general.d_epochs,
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
            cfg.general.num_real_samples))


def finish_experiment(directory):
    global log, generator_loss_sequence, discriminator_loss_sequence, oracle_score_sequence

    print('Finishing experiment..')

    log.info('''FINISHING EXPERIMENT

        ''')

    generator_loss_sequence_str = ','.join(map(str, generator_loss_sequence))
    generator_reward_sequence_str = ','.join(map(str, generator_reward_sequence))
    generator_prediction_sequence_str = ','.join(map(str, generator_prediction_sequence))
    discriminator_loss_sequence_str = ','.join(map(str, discriminator_loss_sequence))
    oracle_score_sequence_str = ','.join(map(str, oracle_score_sequence))

    log.info('Generator Loss as Sequence: ' + generator_loss_sequence_str)
    log.info('Generator Reward as Sequence: ' + generator_reward_sequence_str)
    log.info('Generator Prediction as Sequence: ' + generator_prediction_sequence_str)
    log.info('Discriminator Loss as Sequence: ' + discriminator_loss_sequence_str)
    log.info('Oracle Loss as Sequence: ' + oracle_score_sequence_str)

    shutil.copyfile(cfg.paths.log, directory + '/results.log')
    shutil.copytree(cfg.paths.policies, directory + '/policies')

    try:
        plotter.plot(directory + '/results.log')
    except:
        print('Failed to plot the results.')
        log.info('Failed to plot the results.')
