import generator
import rewards
import constants
import torch
import tokens
import tree
import converter
import math
import datetime
import shutil
import os


# TODO malus fuer syntaktisch inkorrekte baeume


batch_size = constants.GRU_BATCH_SIZE
generated = constants.GENERATED 
arxiv = constants.ARXIV 
length = constants.SEQUENCE_LENGTH
monte = constants.MONTECARLO 
iterations = constants.ITERATIONS
gensteps = constants.GENERATOR_STEPS 
dissteps = constants.DISCRIMINATOR_STEPS


def train_discriminator():

    for _ in range(1):
        samples = generator.rollout()
        converter.convert(samples, folder=generated)
    
    rewards.train()

    shutil.rmtree(generated)
    os.makedirs(generated)


def train_generator():

    for _ in range(length):
        batch, h = generator.step()
        reward = torch.empty([batch.shape[0],0])

        for _ in range(monte):
            samples = generator.rollout(batch, h)
            folder = generated + '/'+ str(datetime.datetime.now())[-15:]
            os.makedirs(folder)
            converter.convert(samples, folder)
            reward = torch.cat([reward, rewards.rewards(folder)], dim=1)

        reward = torch.mean(reward, dim=1)
        generator.feedback(reward)

    generator.update_policy()


def main():
    # pre training

    for _ in range(iterations):
        for _ in range(dissteps):
            train_discriminator()

        generator.update_rollout()

        for _ in range(gensteps):
            train_generator()


if __name__ == "__main__":
    main()
