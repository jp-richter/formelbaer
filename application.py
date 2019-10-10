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


app_dir = c.DIRECTORY_APPLICATION
arxiv_dir = c.DIRECTORY_ARXIV_DATA
gen_dir = c.DIRECTORY_GENERATED_DATA

iterations = c.ADVERSARIAL_ITERATIONS
discriminator_steps = c.ADVERSARIAL_DISCRIMINATOR_STEPS
generator_steps = c.ADVERSARIAL_GENERATOR_STEPS
sequence_length = c.ADVERSARIAL_SEQUENCE_LENGTH
montecarlo_steps = c.ADVERSARIAL_MONTECARLO
batch_size = c.ADVERSARIAL_PREFERRED_BATCH_SIZE


# TODO malus fuer syntaktisch inkorrekte baeume
# TODO training output
# TODO irgendwie uebersichtlicher machen?


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def main():

    # pre training

    for iteration in range(iterations):
        for _ in range(discriminator_steps):
            
            samples = generator.rollout(sequence_length, batch_size=batch_size)
            convert(samples, gen_dir)
            discriminator.train()
            clear(gen_dir)

        generator.update_rollout()

        for _ in range(generator_steps):
            
            for length in range(sequence_length):
                batch, h = generator.step(batch_size=batch_size)
                rewards = torch.empty([batch_size,0])

                for _ in range(montecarlo_steps):
                    missing = sequence_length - length
                    samples = generator.rollout(missing, batch, h)
                    convert(samples, gen_dir)
                    reward = discriminator.rewards(gen_dir)
                    reward = reward[:,None]
                    rewards = torch.cat([rewards, reward], dim=1)

                    clear(gen_dir)

                rewards = torch.mean(rewards, dim=1)
                generator.feedback(rewards)

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
