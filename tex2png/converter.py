import sys
sys.path.append('libs/tex2png')

import subprocess
import constant as c
import helpers as hlp
import os
import time


def copytemplate(targetpath, templatepath, subfoldername, newfilename):
    abstargetsubfolder = hlp.concatinatepath(targetpath, subfoldername)

    hlp.copyfile(templatepath, abstargetsubfolder, newfilename)


def insertintotemplate(subfoldername, templatename, placeholder, string):

    reltemptexpath = hlp.concatinatepath(c.TARGETFOLDER, subfoldername, templatename)
    abstemptexpath = hlp.concatinatepath(c.ROOTTEX2PNG, reltemptexpath)

    try:
        filestr = hlp.readfromfile(abstemptexpath)
        hlp.writefile(abstemptexpath, filestr.replace(placeholder, string))
    except(FileExistsError):
        raise FileExistsError('\"' + c.TEMPLATETEXNAME + '\" file doesn\'t exist in ' + reltemptexpath)


def converttextopdf(pathtofolder, filename, formulaname):
    abspathtofile = hlp.concatinatepath(pathtofolder, filename)
    pdflatexprocess = subprocess.run(
        hlp.changecurrentdirectory(pathtofolder) +
        'pdflatex '
        '-interaction=batchmode '
        '-interaction=nonstopmode '
        '--jobname=' + formulaname + ' \"' +
        abspathtofile + '\" '
        ' > pdflatex' + formulaname + '.log',
        shell=True)
    try:
        pdflatexprocess.check_returncode()
    except:
        return -1


def croppdftoformat(pathtofolder, filename):
    abspathtofile = hlp.concatinatepath(pathtofolder, filename)
    try:
        ghostscriptprocess = subprocess.run(hlp.changecurrentdirectory(pathtofolder) +
                                            'gs '
                                            '-dUseCropBox '
                                            '-dSAFER -dBATCH -dNOPAUSE '
                                            '-sDEVICE=pdfwrite '
                                            '-sOutputFile=' + 'crop' + filename + ' '
                                                                                  '-c \"[/CropBox [550 0 850 100] /PAGES pdfmark\"  ' +
                                            '-f ' + filename +
                                            ' > ghostscript' + filename + 'crop.log'
                                            , shell=True)
        ghostscriptprocess.check_returncode()

    except:
        return -1
    return 0


def convertpdftopng(pathtofolder, filename, formulaname):
    abspathtofile = hlp.concatinatepath(pathtofolder, filename + '.pdf')
    try:
        ghostscriptprocess = subprocess.run(hlp.changecurrentdirectory(pathtofolder) +
                                            'gs '
                                            '-dUseCropBox '
                                            '-dSAFER -dBATCH -dNOPAUSE '
                                            '-sDEVICE=png16m '
                                            '-r80 '
                                            '-sOutputFile=' + formulaname + '.png \"' +
                                            abspathtofile + '\" '
                                            ' > ghostscript' + formulaname + '.log'
                                            , shell=True)
        ghostscriptprocess.check_returncode()

    except:
        return -1
    return 0


def convertsingleformulatoimage(pathtotarget, formulastr, subfoldername, formulaname, macrostr):
    abspathtosubfoldername = hlp.concatinatepath(pathtotarget, subfoldername)
    formulastr = hlp.cleanequation(formulastr)
    copytemplate(hlp.concatinatepath(c.ROOTTEX2PNG, c.TARGETFOLDER),
                 hlp.concatinatepath(c.ROOTTEX2PNG, c.TEMPLATETEXNAME), subfoldername, c.TEMPLATEMACROTEXNAME)
    insertintotemplate(subfoldername, c.TEMPLATEMACROTEXNAME, c.MARCOPH, macrostr)

    insertintotemplate(subfoldername, c.TEMPLATEMACROTEXNAME, c.EQUATIONPH, formulastr)

    returncode = converttextopdf(abspathtosubfoldername, c.TEMPLATEMACROTEXNAME, formulaname)


    if not os.path.isfile(hlp.concatinatepath(abspathtosubfoldername, formulaname + '.pdf')):
        return

    croppdftoformat(abspathtosubfoldername, formulaname + '.pdf')

    # subprocess.run('rm ' + hlp.concatinatepath(abspathtosubfoldername, formulaname+'.pdf'), shell=True)

    returncode = convertpdftopng(abspathtosubfoldername, 'crop' + formulaname, formulaname)
    if (returncode == -1):
        return
