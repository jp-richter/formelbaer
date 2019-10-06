import subprocess
import os
import constants
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


def convert(expressions):

    batch_folder = constants.HOME + '/'+ str(datetime.datetime.now())[-15:]
    class_folder = batch_folder + '/generated'

    os.mkdir(batch_folder)
    os.mkdir(class_folder)
    expr_id = 0

    free = multiprocessing.cpu_count()
    offset = math.ceil(len(expressions) / free)

    def processing(pid):

        start_id = pid.value  * offset
        next_start_id = pid.value * offset + offset - 1

        print(pid.value)
        print(offset)
        print(next_start_id-start_id)

        for expr_id in range(next_start_id - start_id):
            latex = expressions[expr_id]
            file = pdflatex(latex, class_folder, class_folder + '/' + str(expr_id) + '.tex')
            file = croppdf(class_folder, file, str(expr_id))
            file = pdf2png(class_folder, file, str(expr_id))

            expr_id += 1

    for processor in range(free):
        pid = multiprocessing.sharedctypes.RawValue(ctypes.c_int, processor)
        p = multiprocessing.Process(target=processing, args=(pid,))
        p.start()
        p.join()

    for file in os.listdir(class_folder): 
        if not file.endswith('.png'): 
            os.remove(class_folder + '/' + file)

    return batch_folder


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

expressions = ['hello', 'helaaaau', 'hellooo', 'halooooo']
convert(expressions)