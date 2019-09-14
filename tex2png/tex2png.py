import sys
sys.path.append('libs/tex2png')

import ExtractEquationsFromArchives as eefa
import converter as cvt
from multiprocessing import Pool
import helpers as hlp
import constant as c
import os
import subprocess
import sys
import datetime as dt
import t2parchive as t2pa
import time as t
from PIL import Image

def convertsingleformula(formulastring):
    foldername = 'TMP'
    formulaname = 'formula'
    formulastring = eefa.convertInputToEquations(formulastring)[0]
    subprocess.run('mkdir -p \"' + c.TARGETFOLDER + '\"', shell=True)
    subprocess.run('mkdir -p \"' + hlp.concatinatepath(c.TARGETFOLDER,foldername) + '\"', shell=True)

    cvt.convertsingleformulatoimage(hlp.concatinatepath(c.ROOTTEX2PNG,c.TARGETFOLDER),formulastring,foldername,formulaname,'')
    image = Image.open(hlp.concatinatepath(c.TARGETFOLDER,foldername,formulaname+c.PNGEXTENSION))
    return image

def convertsinglefile(pathtoequationfile, targetsubfoldername=c.TARGETSUBFOLDER, pathmacrostxt=None):

    macros = ''
    equationstr = hlp.readfromfile(pathtoequationfile)
    (equationsstr, notused) = eefa.inputfromfile(equationstr, macros)
    if pathmacrostxt is not None:
        macrostr = hlp.readfromfile(pathmacrostxt)
        (notused, macros) = eefa.inputfromfile(macrostr, macros)
    root = subprocess.check_output('pwd').decode('utf-8').replace('\n', '')
    pathtotarget = hlp.concatinatepath(root, c.TARGETFOLDER)
    equations = eefa.convertInputToEquations(equationsstr)
    count = 1;
    for eq in equations:
        cleaneq = hlp.cleanequation(eq)
        # print(str(count) + ' Equation' + ': ' + cleaneq)
        cvt.convertsingleformulatoimage(pathtotarget, cleaneq, subfoldername=targetsubfoldername,
                                        formulaname=hlp.getnumberstr(count), macrostr=macros)
        count += 1


PATHTOFOLDER = ''


def initworker(pathtofolder):
    global PATHTOFOLDER
    PATHTOFOLDER = pathtofolder


def convertsinglefolder(pathtofolder):
    if not os.path.isdir(pathtofolder):
        raise NotADirectoryError('Folder behind path doesn\'t exist.')

    time = dt.datetime.now()

    print('Main: Start generating. ' + str(time))
    pool = Pool(c.NMBWORKERS, initializer=initworker(pathtofolder))

    targetssubfolderpath = hlp.concatinatepath(c.TARGETFOLDER, c.TARGETSUBFOLDER)

    if os.path.isdir(c.TARGETFOLDER):
        if os.path.isdir(targetssubfolderpath):
            subprocess.run('rm -rf \"' + hlp.concatinatepath(targetssubfolderpath, "/*") + '\"', shell=True)
    else:
        subprocess.run('mkdir -p \"' + c.TARGETFOLDER + '\"', shell=True)

    pool.map(_workload, os.listdir(pathtofolder))

    print('Main: Runtime: ' + str(dt.datetime.now() - time))


def _workload(file):
    pathtofile = hlp.concatinatepath(PATHTOFOLDER, file)
    if os.path.isdir(pathtofile) or (not os.path.isfile(pathtofile)):
        return
    targetsubsubfolder = hlp.concatinatepath(c.TARGETSUBFOLDER, file)
    targetsubsubfolderpath = hlp.concatinatepath(c.TARGETFOLDER, targetsubsubfolder)

    targetsubsubimagefolderpath = hlp.concatinatepath(c.TARGETFOLDER,c.TARGETSUBFOLDERIMAGE, file)

    convertsinglefile(pathtofile, targetsubsubfolder)
    try:
        hlp.recursivextractintactimages(targetsubsubfolderpath, targetsubsubimagefolderpath)
    except:
        pass

    if c.DELETESET:
        subprocess.run('rm -rf \"' + hlp.concatinatepath(targetsubsubfolderpath) + '\"', shell=True)


def checkargumentsforflags(arguments):
    targetsubfolderimageargument = False
    targetsubfolderargument = False

    for argument in arguments:
        if targetsubfolderimageargument:
            targetsubfolderimageargument = False
            c.TARGETSUBFOLDERIMAGE = argument
            if os.path.isdir(argument):
                print('Argument: Subfolder \'', argument, '\' already exists.')

        elif targetsubfolderargument:
            targetsubfolderargument = False
            c.TARGETSUBFOLDER = argument

        elif argument == c.TARGETSUBFOLDERIMAGEFLAG:
            targetsubfolderimageargument = True
        elif argument == c.TARGETSUBFOLDERFLAG:
            targetsubfolderargument = True
        elif argument == c.PRINTFLAG:
            c.PRINTSET = True
        elif argument == c.DELETEFLAG:
            c.DELETESET = True
        else:
            print('Argument: \'', argument, '\' not recognized.')


def checkarchiveargumentsforflags(arguments):
    for argument in arguments:
        if argument == c.PRINTFLAG:
            c.PRINTSET = True
        elif argument == c.DELETEFLAG:
            c.DELETESET = True
        elif argument == c.COUNTERFLAG:
            c.COUNTERSET = True
        else:
            print('Argument: \'', argument, '\' not recognized.')

# First argument:  absolut or relative path
# Second argument: subfoldername in target
if len(sys.argv) > 1:
    # The Last Argument is the ARCHIVEARGUMENT
    if (sys.argv[1] == c.ARCHIVEARGUMENT):
        arguments = sys.argv[2:len(sys.argv)]
        checkarchiveargumentsforflags(arguments)
        t2pa.main()

    else:

        arguments = sys.argv[2:len(sys.argv)]
        checkargumentsforflags(arguments)

        if os.path.isdir(sys.argv[1]):
            convertsinglefolder(sys.argv[1])
        elif os.path.isfile(sys.argv[1]):
            convertsinglefile(sys.argv[1])
        else:
            print('No valid path for folder or file. Please check your input.')


else:
    print('Please call the function with an argument of a path to a file or folder')
