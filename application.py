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


def save():

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


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def main():

    logging.basicConfig(level=logging.INFO, filename=FILE_RESULT_LOG)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # pre training

    for iteration in range(c.ADVERSARIAL_ITERATIONS):
        for _ in range(c.ADVERSARIAL_DISCRIMINATOR_STEPS):
            
            samples = generator.rollout(c.ADVERSARIAL_SEQUENCE_LENGTH, batchsize=c.ADVERSARIAL_BATCHSIZE)
            converter.convert(samples, c.DIRECTORY_GENERATED_DATA)
            discriminator.train()
            clear(c.DIRECTORY_GENERATED_DATA)

        generator.update_rollout()

        for _ in range(generator_steps):
            
            for current_length in range(c.ADVERSARIAL_SEQUENCE_LENGTH):
                batch, hidden = generator.step(batchsize=c.ADVERSARIAL_BATCHSIZE)
                state_action_values = torch.empty([c.ADVERSARIAL_BATCHSIZE,0])
                missing_length = c.ADVERSARIAL_SEQUENCE_LENGTH - current_length

                for _ in range(c.ADVERSARIAL_MONTECARLO_TRIALS):
                    samples = generator.rollout(missing_length, batch, hidden)
                    converter.convert(samples, c.DIRECTORY_GENERATED_DATA)

                    single_episode = discriminator.rewards(c.DIRECTORY_GENERATED_DATA)
                    single_episode = single_episode[:,None]
                    state_action_values = torch.cat([state_action_values, single_episode], dim=1)

                    clear(c.DIRECTORY_GENERATED_DATA)

                state_action_values = torch.mean(state_action_values, dim=1)
                generator.feedback(state_action_values)

            generator.update_policy()

        if iterations+1 % 5 == 0:

            greward = -1 * generator.running_reward / (iteration+1 * generator_steps)
            dloss = discriminator.running_loss / (iteration+1 * discriminator_steps)

            log.info('''###
                Iteration {iteration}
                Generator Reward {greward}
                Discriminator Loss {dloss}
                ###'''.format(iteration=iteration, greward=greward, dloss=dloss))

            generator.running_reward = 0.0
            discriminator.running_loss = 0.0

    save()

if __name__ == "__main__":
    main()
