import subprocess
import os
import constants
import shutil
import shlex
import datetime
import tree

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


def convert(batch):

    folder = constants.HOME + '/'+ datetime.datetime.now.strftime("%d-%m-%Y_%H:%M:%S")
    name = 0

    # TODO in parallel?
    files = []
    for sample in batch:
        tree = tree.parse(sample)
        latex = tree.latex()

        file = pdflatex(latex, folder, folder + '/' + str(sample))
        file = croppdf(folder, file, str(sample))
        file = pdf2png(folder, file, str(name))

        files.append(file)
        name += 1

    return files

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
        '-c [/CropBox [550 0 850 100] '
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

