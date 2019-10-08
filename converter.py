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

template = '''
\\documentclass{{standalone}}

\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath, amssymb}}

\\begin{{document}}
\\begin{{minipage}}[c][1cm]{{50cm}}
\\begin{{equation}}
{expression}
\\end{{equation}}
\\end{{minipage}}
\\end{{document}}
'''


def convert(sequences, folder):

    trees = tree.batch2tree(sequences)
    expressions = [tree.latex() for tree in trees]
    expr_id = len(os.listdir(folder))

    free = multiprocessing.cpu_count()
    offset = math.ceil(len(expressions) / free)

    def processing(pid):

        start_id = pid.value  * offset
        next_start_id = pid.value * offset + offset

        for id in range(next_start_id - start_id):
            expr_id = start_id + id
            latex = expressions[expr_id]
            file = pdflatex(latex, folder, folder + '/' + str(expr_id) + '.tex')
            # file = croppdf(folder, file, str(expr_id)) -> torchvision.transforms
            file = pdf2png(folder, file, str(expr_id))

            expr_id += 1

    for processor in range(free):
        pid = multiprocessing.sharedctypes.RawValue(ctypes.c_int, processor)
        p = multiprocessing.Process(target=processing, args=(pid,))
        p.start()
        p.join()

    with os.scandir(folder) as iterator:
        for entry in iterator:
            if entry.is_file() and not entry.name.endswith('.png'): 
                os.remove(folder + '/' + entry.name)


def pdflatex(expr, folder, file):

    with open(file,'w') as f:
        f.write(template.format(expression=expr))

    cmd = ['pdflatex',
        '-interaction=batchmode',
        '-interaction=nonstopmode',
        file]

    try:
        subprocess.run(cmd, cwd=folder, stdout=subprocess.DEVNULL, timeout=1)
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
        subprocess.run(cmd, cwd=folder, stdout=subprocess.DEVNULL, timeout=1)
    except Exception as e:
        print(e)

    return folder + '/' + expr_id + '.png'
