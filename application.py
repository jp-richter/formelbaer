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

import constants as c
from pathlib import Path


logging.basicConfig(level=logging.INFO, file='results.log')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

arxiv_data_dir = c.DIRECTORY_ARXIV_DATA
generated_data_dir = c.DIRECTORY_GENERATED_DATA

iterations = c.ADVERSARIAL_ITERATIONS
discriminator_steps = c.ADVERSARIAL_DISCRIMINATOR_STEPS
generator_steps = c.ADVERSARIAL_GENERATOR_STEPS
sequence_length = c.ADVERSARIAL_SEQUENCE_LENGTH
montecarlo_steps = c.ADVERSARIAL_MONTECARLO
batch_size = c.ADVERSARIAL_PREFERRED_BATCH_SIZE


# TODO malus fuer syntaktisch inkorrekte baeume

# trade off: computation time vs batch size effects
# trade off: more generalization with lower batch size but less accurate gradients
# TODO tests: make up for high batch sizes with learning rate?


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def main():

    # pre training

    for iteration in range(iterations):
        for _ in range(discriminator_steps):
            
            samples = generator.rollout(sequence_length, batch_size=batch_size)
            converter.convert(samples, generated_data_dir)
            discriminator.train()
            clear(generated_data_dir)

        generator.update_rollout()

        for _ in range(generator_steps):
            
            for current_length in range(sequence_length):
                batch, hidden = generator.step(batch_size=batch_size)
                state_action_values = torch.empty([batch_size,0])
                missing_length = sequence_length - current_length

                for _ in range(montecarlo_steps):
                    samples = generator.rollout(missing_length, batch, hidden)
                    converter.convert(samples, generated_data_dir)

                    single_episode = discriminator.rewards(generated_data_dir)
                    single_episode = single_episode[:,None]
                    state_action_values = torch.cat([state_action_values, single_episode], dim=1)

                    clear(generated_data_dir)

                state_action_values = torch.mean(state_action_values, dim=1)
                generator.feedback(state_action_values)

            generator.update_policy()

        if iterations % 1 == 0:

            greward = -1 * generator.running_reward / (iteration+1 * generator_steps)
            dloss = discriminator.running_loss / (iteration+1 * discriminator_steps)

            log.info('''###
                Iteration {iteration}
                Generator Reward {greward}
                Discriminator Loss {dloss}
                ###'''.format(iteration=iteration, greward=greward, dloss=dloss))

            generator.running_reward = 0.0
            discriminator.running_loss = 0.0

    generator.update_rollout()
    for _ in range(math.ceil(100 / batch_size)):
        evaluation = generator.rollout(sequence_length, batch_size=batch_size)
        converter.convert(evaluation, generated_data_dir)

    with os.scandir(generated_data_dir) as iterator:
        for entry in iterator:
            if entry.is_file() and not entry.name.endswith('.png'):
                os.remove(entry)

if __name__ == "__main__":
    main()
