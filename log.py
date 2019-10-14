import constants as const

import logging
import shutil

log = None

# making parsing a bit easier
generator_loss_sequence = []
discriminator_loss_sequence = []

def start_experiment():

    logging.basicConfig(level=logging.INFO, filename=const.FILE_LOG)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    log.info('''STARTING EXPERIMENT

    	''')

    log.info('''Models Version: 1
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

	    '''.format(
	        const.ADVERSARIAL_ITERATIONS, 
	        const.ADVERSARIAL_DISCRIMINATOR_STEPS, 
	        const.ADVERSARIAL_GENERATOR_STEPS, 
	        const.ADVERSARIAL_SEQUENCE_LENGTH, 
	        const.ADVERSARIAL_MONTECARLO_TRIALS, 
	        const.ADVERSARIAL_BATCHSIZE, 
	        const.GENERATOR_HIDDEN_DIM, 
	        const.GENERATOR_LAYERS, 
	        const.GENERATOR_DROPOUT, 
	        const.GENERATOR_BASELINE,
	        const.GENERATOR_GAMMA, 
	        const.DISCRIMINATOR_DROPOUT, 
	        const.DISCRIMINATOR_LEARNRATE))


def log(iteration, g_steps, d_steps, nn_generator, nn_discriminator):

    g_reward = -1 * nn_generator.running_reward / (iteration * g_steps)
    d_loss = discriminator.running_loss / (iteration * d_steps)

    log.info('''###
        Iteration {iteration}
        Generator Reward {greward}
        Discriminator Loss {dloss}

        '''.format(iteration=iteration, greward=g_reward, dloss=d_loss))

    generator_loss_sequence.append(generator.running_reward)
    discriminator_loss_sequence.append(discriminator.running_loss)

    generator.running_reward = 0.0
    discriminator.running_loss = 0.0


def finish_experiment(directory):

	log.info('''FINISHING EXPERIMENT

		''')

    generator_loss_sequence = ', '.join(map(str, generator_loss_sequence))
    discriminator_loss_sequence = ', '.join(map(str, discriminator_loss_sequence))
    log.info('Generator Loss as Sequence: ' + generator_loss_sequence)
    log.info('Discriminator Loss as Sequence ' + discriminator_loss_sequence)

    shutil.copyfile(c.FILE_LOG, directory + '/results.log')
