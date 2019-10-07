import generator
import rewards
import constants
import torch
import tokens
import tree
import converter
import math


cycles = constants.ITERATIONS
gsteps = constants.GSTEPS
dsteps = constants.DSTEPS
seqlength = constants.SEQ_LENGTH
mcarlo = constants.MONTECARLO
negatives = constants.NEGATIVE_SAMPLES
dneg_dir = constants.DNEG_DIR
batch_size = constants.GRU_BATCH_SIZE


def train_discriminator():

    for _ in range(math.floor(negatives / batch_size)):
        samples = generator.rollout()
        converter.convert(samples[-1], folder=dneg_dir)
    
    rewards.train()


def train_generator():

    for _ in range(seqlength):
        batch, h = generator.step()
        reward = torch.empty([batch.shape[0],0])

        for _ in range(mcarlo):
            samples = generator.rollout(batch, h)
            folder = converter.convert(samples)
            reward = torch.cat([reward, rewards.rewards(folder)], dim=1)

        reward = torch.mean(reward, dim=1)
        generator.feedback(reward)

    generator.update_policy()


def main():
    # pre training

    for _ in range(cycles):
        for _ in range(dsteps):
            train_discriminator()

        generator.update_rollout()

        for _ in range(gsteps):
            train_generator()


if __name__ == "__main__":
    main()
