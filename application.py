import generator
import discriminator
import torch
import tokens
import tree
import math
import datetime
import shutil
import os
import converter
import logging
import datetime

import constants as c
from pathlib import Path


log = None


def save_experiment():

    # all results go here
    folder = constants.DIRECTORY_APPLICATION + '/'+ str(datetime.datetime.now())
    os.makedirs(folder)

    # save 100 example pngs
    pngs = folder + '/pngs'
    os.makedirs(pngs)

    generator.update_rollout()
    examples = generator.rollout(sequence_length, batchsize=100)

    converter.convert(examples, pngs)
    converter.cleanup(pngs)

    # save model parameters
    generator.save_parameters(folder)
    discriminator.save_parameters(folder)

    # save result and parameter logs
    shutil.copyfile(c.FILE_RESULT_LOG, folder + '/results.log')
    shutil.copyfile(c.FILE_PARAMETERS_LOG, folder + '/parameters.log')


def log_results(iteration):
    global log

    if log is None:
        logging.basicConfig(level=logging.INFO, filename=c.FILE_RESULT_LOG)
        log = logging.getLogger(__name__)
        log.setLevel(logging.INFO)

    greward = -1 * generator.running_reward / (iteration * c.ADVERSARIAL_GENERATOR_STEPS)
    dloss = discriminator.running_loss / (iteration * c.ADVERSARIAL_DISCRIMINATOR_STEPS)

    log.info('''###
        Iteration {iteration}
        Generator Reward {greward}
        Discriminator Loss {dloss}
        ###'''.format(iteration=iteration, greward=greward, dloss=dloss))

    generator.running_reward = 0.0
    discriminator.running_loss = 0.0


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def discriminator_training():

    samples = generator.rollout()
    half = samples[:len(samples)//2]
    converter.convert_to_png(half)
    discriminator.train()
    clear(c.DIRECTORY_GENERATED_DATA)


def generator_training():

    for current_length in range(c.ADVERSARIAL_SEQUENCE_LENGTH):

        batch, hidden = generator.step()
        state_action_values = torch.empty([c.ADVERSARIAL_BATCHSIZE,0])

        for _ in range(c.ADVERSARIAL_MONTECARLO_TRIALS):

            samples = generator.rollout(batch, hidden)
            converter.convert_to_png(samples)

            single_episode = discriminator.rewards()
            single_episode = single_episode[:,None]
            state_action_values = torch.cat([state_action_values, single_episode], dim=1)

            clear(c.DIRECTORY_GENERATED_DATA)

        state_action_values = torch.mean(state_action_values, dim=1)
        generator.feedback(state_action_values)

    generator.update_policy()


def adversarial_training():

    for iteration in range(c.ADVERSARIAL_ITERATIONS):

        for _ in range(c.ADVERSARIAL_DISCRIMINATOR_STEPS):
            discriminator_training()

        for _ in range(c.ADVERSARIAL_GENERATOR_STEPS):
            generator_training()

        generator.update_rollout()

        if iteration+1 % 5 == 0:
            log_results(iteration+1)

    save_experiment()

if __name__ == "__main__":
    adversarial_training()
