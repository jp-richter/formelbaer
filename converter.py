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

# preamble preloaded in preambel.fmt
# pdf compression set to 0
code_directory = pathlib.Path(__file__).resolve().parent
preamble = pathlib.PurePath(code_directory,'preamble.fmt')

# precompile in case pdflatex versions differ
precompile_cmd = 'pdflatex -ini -jobname="preamble" "&pdflatex preamble.tex\\dump"'

try:
    subprocess.run(precompile_cmd, cwd=code_directory, stdout=subprocess.DEVNULL, shell=True)
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
current_directory = None
current_expressions = None


def cleanup(directory):

    with os.scandir(directory) as iterator:
        for entry in iterator:
            if entry.is_file() and not entry.name.endswith('.png'):
                os.remove(entry)


# def processing(enumeration):
#     global current_directory, current_start_index

#     index, expression = enumeration
#     index += current_start_index
    
#     file = pdflatex(expression, current_directory, current_directory + '/' + str(index) + '.tex')
#     file = croppdf(current_directory, file, str(index))
#     file = pdf2png(current_directory, file, str(index))


def processing(pid):
    global current_directory, current_start_index, current_expressions

    num_seqs = len(current_expressions)
    free_cpus = multiprocessing.cpu_count()
    cpus_used = min(len(num_seqs), free_cpus)

    offset = num_seqs // cpus_used
    start_index = current_start_index + pid * offset
    next_index = current_start_index + (pid+1) * offset

    for i in range(next_index - start_index):
        index = start_index + i 
        name = str(index)

        file = pdflatex(current_expressions[index], current_directory, current_directory + '/' + name + '.tex')
        file = croppdf(current_directory, file, name)
        file = pdf2png(current_directory, file, name)


def convert_to_png(sequences, directory = cfg.paths_cfg.synthetic_data):
    global current_directory, current_start_index, current_expressions

    shutil.copyfile(preamble, directory + '/preamble.fmt')

    trees = tree.batch2tree(sequences)
    current_expressions = [tree.latex() for tree in trees]

    current_start_index = len(os.listdir(directory))
    current_directory = directory
    free_cpus = multiprocessing.cpu_count()
    cpus_used = min(len(current_expressions), free_cpus)

    for pid in range(cpus_used):
        # pid = multiprocessing.sharedctypes.RawValue(ctypes.c_int, processor)
        p = multiprocessing.Process(target=processing, args=(pid,))
        p.start()
        p.join()

    # with multiprocessing.Pool(free_cpus) as pool:
    #     pool.map(processing, enumerate(expressions))

    #     pool.close() # don't remove
    #     pool.join() # don't remove


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
