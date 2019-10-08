import generator
import discriminator
import torch
import tokens
import tree
import math
import datetime
import shutil
import os

from pathlib import Path
from converter import convert


# TODO malus fuer syntaktisch inkorrekte baeume
# TODO training output
# TODO irgendwie uebersichtlicher machen?


home_dir = str(Path.home()) + '/formelbaer'
arxiv_dir = home_dir + '/arxiv'
gen_dir = home_dir + '/generated'

iterations = 1
discriminator_steps = 1
generator_steps = 1
sequence_length = 2
montecarlo_steps = 1

batch_size = 32


def init():

    if not os.path.exists(home_dir):
        os.makedirs(home_dir)

    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    if not os.path.exists(arxiv_dir):
        raise ValueError('Please save the training set of arxiv .png or .pt files in ' \
            + arxiv_dir + ' or change the ARXIV directory in application.py accordingly.')


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def main():

    init()

    # pre training

    for _ in range(iterations):
        for _ in range(discriminator_steps):
            
            samples = generator.rollout(batch_size, sequence_length)
            convert(samples, gen_dir)
            discriminator.train(gen_dir, batch_size)
            clear(gen_dir)

        generator.updaterollout()

        for _ in range(generator_steps):
            
            for length in range(sequence_length):
                batch, h = generator.step(batch_size)
                rewards = torch.empty([batch.shape[0],0])

                for _ in range(montecarlo_steps):
                    missing = sequence_length - length
                    samples = generator.rollout(batch_size, missing, batch, h)
                    convert(samples, gen_dir)
                    reward = discriminator.rewards(gen_dir, batch_size)
                    rewards = torch.cat([rewards, reward], dim=1)

                    clear(gen_dir)

                    rewards = torch.mean(rewards, dim=1)
                    generator.feedback(rewards)

            generator.updatepolicy()


if __name__ == "__main__":
    main()
