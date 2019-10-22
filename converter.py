import subprocess
import os
import shutil
import multiprocessing
import math
import pathlib
import config as cfg
import importlib
import tree


# multiprocessing library ray should give better performance, windows is not supported
ray_spec = importlib.util.find_spec("ray")
ray_available = ray_spec is not None
num_cpus = multiprocessing.cpu_count()

if ray_available:
    import ray
    ray.init(num_cpus=num_cpus)

# preamble preloaded in preamble.fmt, saved in preamble.tex, pdf compression set to 3
code_directory = pathlib.Path(__file__).resolve().parent
preamble = pathlib.PurePath(code_directory, 'preamble.fmt')

# using preambles precompiled with different pdflatex versions raises an error
precompile_cmd = 'pdflatex -ini -jobname="preamble" "&pdflatex preamble.tex\\dump" > ' + cfg.paths_cfg.dump
subprocess.run(precompile_cmd, cwd=code_directory, shell=True)

# latex expression template
template = '''%&preamble

\\begin{{document}}
\\begin{{minipage}}[c][1cm]{{50cm}}
\\centering{{
    $ {expression} $
}}
\\end{{minipage}}
\\end{{document}}'''


def conditional_ray_remote(func):

    if not ray_available: return func
    else: return ray.remote(func)


@conditional_ray_remote
def process(pid, offset, sequences, directory, file_count):

    start_index = pid * offset
    end_index = (pid + 1) * offset
    end_index = min(end_index, len(sequences))

    for i in range(start_index, end_index):
        name = str(file_count + i)
        file = pdflatex(sequences[i], directory, directory + '/' + name + '.tex')
        file = croppdf(directory, file, name)
        file = pdf2png(directory, file, name)

    return True


def convert_to_png(sequences, directory=cfg.paths_cfg.synthetic_data):
    global num_cpus

    shutil.copyfile(preamble, directory + '/preamble.fmt')

    trees = tree.batch2tree(sequences)
    expressions = [t.latex() for t in trees]

    num_seqs = len(expressions)
    cpus_used = min(num_seqs, num_cpus)
    offset = math.ceil(num_seqs / cpus_used)
    file_count = len(os.listdir(directory))

    if ray_available:
        def use_ray():
            # copy to shared memory once instead of copying to each cpu
            sequences_id = ray.put(expressions)
            offset_id = ray.put(offset)
            directory_id = ray.put(directory)
            file_count_id = ray.put(file_count)

            # no need for return value but call get for synchronisation
            ray.get([process.remote(
                pid,
                offset_id,
                sequences_id,
                directory_id,
                file_count_id) for pid in range(cpus_used)])

        return use_ray()

    else:
        def use_multiprocessing():
            processes = []

            for pid in range(cpus_used):
                proc = multiprocessing.Process(target=process, args=(pid,offset,sequences,directory, file_count))
                processes.append(proc)
                proc.start()

            for proc in processes:
                proc.join()

        return use_multiprocessing()


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
        '-c [/CropBox [550 0 850 100] '  # (x,y) (x',y')
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
