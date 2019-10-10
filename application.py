import generator
import discriminator
import torch
import tokens
import tree
import math
import datetime
import shutil
import os

import constants as c

from pathlib import Path
from converter import convert


arxiv_data_dir = c.DIRECTORY_ARXIV_DATA
generated_data_dir = c.DIRECTORY_GENERATED_DATA

iterations = c.ADVERSARIAL_ITERATIONS
discriminator_steps = c.ADVERSARIAL_DISCRIMINATOR_STEPS
generator_steps = c.ADVERSARIAL_GENERATOR_STEPS
sequence_length = c.ADVERSARIAL_SEQUENCE_LENGTH
montecarlo_steps = c.ADVERSARIAL_MONTECARLO
batch_size = c.ADVERSARIAL_PREFERRED_BATCH_SIZE


# TODO malus fuer syntaktisch inkorrekte baeume


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def main():

    # pre training

    for iteration in range(iterations):
        for _ in range(discriminator_steps):
            
            samples = generator.rollout(sequence_length, batch_size=batch_size)
            convert(samples, generated_data_dir)
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
                    convert(samples, generated_data_dir)

                    single_episode = discriminator.rewards(generated_data_dir)
                    single_episode = single_episode[:,None]
                    state_action_values = torch.cat([state_action_values, single_episode], dim=1)

                    clear(generated_data_dir)

                state_action_values = torch.mean(state_action_values, dim=1)
                generator.feedback(state_action_values)

            generator.update_policy()

        if iterations % 1 == 0:

            gloss = generator.running_loss / (iteration+1 * generator_steps)
            dloss = discriminator.running_loss / (iteration+1 * discriminator_steps)

            print('###')
            print('Iteration {}'.format(iteration))
            print('Generator Loss {}'.format(gloss))
            print('Discriminator Loss {}'.format(dloss))
            print('###')

            generator.running_loss = 0.0
            discriminator.running_loss = 0.0


if __name__ == "__main__":
    main()
