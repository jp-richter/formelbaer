import subprocess
import os
import shutil
import shlex
import datetime
import tree
import multiprocessing
import multiprocessing.sharedctypes
import math
import ctypes
import pathlib
import config as cfg


# preamble preloaded in preamble.fmt, saved in preamble.tex, pdf compression 3
code_directory = pathlib.Path(__file__).resolve().parent
preamble = pathlib.PurePath(code_directory,'preamble.fmt')

# precompile in case pdflatex versions differ
precompile_cmd = 'pdflatex -ini -jobname="preamble" "&pdflatex preamble.tex\\dump"'

try:
    subprocess.run(precompile_cmd, cwd=code_directory, stdout=subprocess.DEVNULL, shell=True)
except Exception as e:
    print(e)

template = '''%&preamble

\\begin{{document}}
\\begin{{minipage}}[c][1cm]{{50cm}}
\\centering{{
    $ {expression} $
}}
\\end{{minipage}}
\\end{{document}}
'''

#
# MULTIPROCESSING WITH RAY
#

# import ray


# num_cpus = multiprocessing.cpu_count()
# ray.init(num_cpus=num_cpus)


# @ray.remote
# def process_with_ray(pid, offset, sequences, directory, file_count):

#     start_index = pid * offset
#     end_index = (pid+1) * offset
#     end_index = min(end_index, len(sequences))

#     for i in range(start_index, end_index):
#         name = str(file_count + i)
#         file = pdflatex(sequences[i], directory, directory + '/' + name + '.tex')
#         file = croppdf(directory, file, name)
#         file = pdf2png(directory, file, name)

#     return True


# def convert_with_ray(sequences, directory):
#     global num_cpus

#     shutil.copyfile(preamble, directory + '/preamble.fmt')

#     trees = tree.batch2tree(sequences)
#     expressions = [tree.latex() for tree in trees]

#     cpus_used = min(len(expressions), num_cpus)
#     offset = math.ceil(num_seqs / cpus_used)
#     file_count = len(os.listdir(directory))

#     # copy to shared memory once instead of copying to each cpu
#     sequences_id = ray.put(expressions)
#     offset_id = ray.put(offset)
#     directory_id = ray.put(directory)
#     file_count_id = ray.put(file_count)

#     # no need for return value but call get for synchronisation
#     ray.get([process_with_ray.remote(
#         pid, 
#         offset_id, 
#         sequences_id, 
#         directory_id, 
#         file_count_id) for pid in range(cpus_used)])


#
# MULTIPROCESSING WITH RAY END
#

#
# MULTIPROCESSING WITH MULTIPROCESSING LIBRARY
#


current_file_count = None
current_directory = None
current_expressions = None


def process_with_multiprocessing(pid):
    global current_directory, current_file_count, current_expressions

    num_seqs = len(current_expressions)
    free_cpus = multiprocessing.cpu_count()
    cpus_used = min(num_seqs, free_cpus)

    offset = math.ceil(num_seqs / cpus_used)
    start_index = pid * offset
    end_index = (pid+1) * offset

    for i in range(start_index, end_index):
        name = str(current_file_count + i)

        if not i < num_seqs:
            break

        file = pdflatex(current_expressions[i], current_directory, current_directory + '/' + name + '.tex')
        file = croppdf(current_directory, file, name)
        file = pdf2png(current_directory, file, name)


def convert_with_multiprocessing(sequences, directory = cfg.paths_cfg.synthetic_data):
    global current_directory, current_file_count, current_expressions

    shutil.copyfile(preamble, directory + '/preamble.fmt')

    trees = tree.batch2tree(sequences)
    current_expressions = [tree.latex() for tree in trees]

    current_file_count = len(os.listdir(directory))
    current_directory = directory
    free_cpus = multiprocessing.cpu_count()
    cpus_used = min(len(current_expressions), free_cpus)

    processes = []

    for cpu in range(cpus_used):
        process = multiprocessing.Process(target=process_with_multiprocessing, args=(cpu,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


#
# MULTIPROCESSING WITH MULTIPROCESSING END
#


def convert_to_png(sequences, directory = cfg.paths_cfg.synthetic_data):

    convert_with_multiprocessing(sequences, directory)


def clean_up(directory):

    with os.scandir(directory) as iterator:
        for entry in iterator:
            if entry.is_file() and not entry.name.endswith('.png'):
                os.remove(entry)


def pdflatex(expr, directory, file):

    with open(file,'w') as f:
        f.write(template.format(expression=expr))

    cmd = ['pdflatex',
        '-interaction=batchmode',
        '-interaction=nonstopmode',
        file]
    #  stdout=subprocess.DEVNULL, 
    try:
        subprocess.run(cmd, cwd=directory, stdout=subprocess.DEVNULL, timeout=2)
    except Exception as e:
        print(e)

    return file[:-3] + 'pdf'


def croppdf(directory, file, expr_id):

    cmd = ('gs '
        '-dUseCropBox '
        '-dSAFER '
        '-dBATCH '
        '-dNOPAUSE '
        '-sDEVICE=pdfwrite '
        '-sOutputFile={}/crop_{}.pdf '
        '-c [/CropBox [550 0 850 100] ' # (x,y) (x',y')
        '-c /PAGES pdfmark '
        '-f {} '
        '> {}')

    cmd = cmd.format(directory,expr_id,file,cfg.paths_cfg.dump)

    try:
        subprocess.run(cmd, cwd=directory, shell=True)
    except Exception as e:
        print(e)

    return directory + '/crop_' + expr_id + '.pdf'


def pdf2png(directory, file, expr_id):

    cmd = ['gs',
        '-dUseCropBox',
        '-dSAFER',
        '-dBATCH',
        '-dNOPAUSE',
        '-sDEVICE=png16m',
        '-r80',
        '-sOutputFile=' + expr_id + '.png',
        file]
    
    try:
        subprocess.run(cmd, cwd=directory, stdout=subprocess.DEVNULL, timeout=2)
    except Exception as e:
        print(e)

    return directory + '/' + expr_id + '.png'
