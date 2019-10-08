import generator
import discriminator
import constants
import torch
import tokens
import tree
import math
import datetime
import shutil
import os

from converter import convert


# TODO malus fuer syntaktisch inkorrekte baeume
# TODO training output
# TODO default werte und argumente flexibel machen


def clear(folder):

    shutil.rmtree(folder)
    os.makedirs(folder)


def main():
    # pre training

    for _ in range(constants.ITERATIONS):
        for _ in range(constants.DISCRIMINATOR_STEPS):
            
            samples = generator.rollout()
            convert(samples, folder=constants.GENERATED)
            discriminator.train()
            clear(folder=constants.GENERATED)

        generator.update_rollout()

        for _ in range(constants.GENERATOR_STEPS):
            
            for _ in range(constants.SEQUENCE_LENGTH):
                batch, h = generator.step()
                rewards = torch.empty([batch.shape[0],0])

                for _ in range(constants.MONTECARLO):
                    samples = generator.rollout(batch, h)
                    convert(samples, folder=constants.GENERATED )
                    reward = discriminator.rewards(folder=constants.GENERATED)
                    rewards = torch.cat([rewards, reward], dim=1)

                    clear(folder=constants.GENERATED )

                    rewards = torch.mean(rewards, dim=1)
                    generator.feedback(rewards)

            generator.update_policy()


if __name__ == "__main__":
    main()
