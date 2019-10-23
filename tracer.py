import sys
import traceback


class TracePrints(object):

    def __init__(self):

        self.stdout = sys.stdout

    def write(self, s):

        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)


def start():

    sys.stdout = TracePrints()


def stop():

    sys.stdout = sys.__stdout__
