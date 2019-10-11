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

# preamble preloaded in preambel.fmt
# pdf compression set to 0
code_folder = pathlib.Path(__file__).resolve().parent
preamble = pathlib.PurePath(code_folder,'preamble.fmt')

# precompile in case pdflatex versions differ
precompile_cmd = 'pdflatex -ini -jobname="preamble" "&pdflatex preamble.tex\\dump"'

try:
    subprocess.run(precompile_cmd, cwd=code_folder, stdout=subprocess.DEVNULL, shell=True)
except Exception as e:
    print(e)

# equation environment doesn't work
# template = '''
# %&preambel

# \\begin{{document}}
# \\begin{{minipage}}[c][1cm]{{50cm}}
# \\begin{{equation}}
# {expression}
# \\end{{equation}}
# \\end{{minipage}}
# \\end{{document}}
# '''

template = '''%&preamble

\\begin{{document}}
\\begin{{minipage}}[c][1cm]{{50cm}}
\\centering{{
    $ {expression} $
}}
\\end{{minipage}}
\\end{{document}}
'''

current_start_index = None
current_folder = None


def processing(enumeration):
    global current_folder, current_start_index

    index, expression = enumeration
    index += current_start_index
    
    file = pdflatex(expression, current_folder, current_folder + '/' + str(index) + '.tex')
    file = pdf2png(current_folder, file, str(index))


def convert(sequences, folder):
    global current_folder, current_start_index

    shutil.copyfile(preamble, folder + '/preamble.fmt')

    trees = tree.batch2tree(sequences)
    expressions = [tree.latex() for tree in trees]

    current_start_index = len(os.listdir(folder))
    current_folder = folder
    free_cpus = multiprocessing.cpu_count()

    with multiprocessing.Pool(free_cpus) as pool:
        pool.map(processing, enumerate(expressions))

        pool.close() # don't remove
        pool.join() # don't remove


def pdflatex(expr, folder, file):

    with open(file,'w') as f:
        f.write(template.format(expression=expr))

    cmd = ['pdflatex',
        '-interaction=batchmode',
        '-interaction=nonstopmode',
        file]
    #  stdout=subprocess.DEVNULL, 
    try:
        subprocess.run(cmd, cwd=folder, stdout=subprocess.DEVNULL, timeout=30)
    except Exception as e:
        print(e)

    return file[:-3] + 'pdf'


def croppdf(folder, file, expr_id):

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
        '> /Users/jan/formelbaer/log.txt')

    cmd = cmd.format(folder,expr_id,file)

    try:
        subprocess.run(cmd, cwd=folder, shell=True)
    except Exception as e:
        print(e)

    return folder + '/crop_' + expr_id + '.pdf'


def pdf2png(folder, file, expr_id):

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
        subprocess.run(cmd, cwd=folder, stdout=subprocess.DEVNULL, timeout=30)
    except Exception as e:
        print(e)

    return folder + '/' + expr_id + '.png'
