import sys
sys.path.append('libs/tex2png')
sys.path.append('libs/indexfaissrepo')

import subprocess
from multiprocessing import Pool
import multiprocessing as mp
import os
import datetime as dt
import constant as c
import helpers as hlp
import ExtractEquationsFromArchives as eefa
import converter as cvt
import time
import shutil
from PIL import Image
import io


PROCESSID = 'main'
PROCESSTARGETFOLDER = hlp.concatinatepath(c.TARGETFOLDER, PROCESSID)
ARCHIVE = 'none'

EXTRACTEDIMAGESCOUNTER = None;
IMAGESCOUNTER = None;
ARCHIVECOUNTER = None;

def _cleartargetofprocess():
    subprocess.run('rm  -rf ' + hlp.concatinatepath(c.TARGETFOLDER, PROCESSID, '/*'), shell=True)


def processprint(text):
    if c.PRINTSET:
        time = dt.datetime.now().time()
        print('TIME: ' + str(time) + '; ID: ' + PROCESSID + '; ARCHIV: ' + ARCHIVE + ': ' + text)


# For Subprocess: Create folder inside the target folder with own processid as name
def init(imagecounter, extractedimagecounter, archivecounter):
    global EXTRACTEDIMAGESCOUNTER
    global IMAGESCOUNTER
    global ARCHIVECOUNTER

    EXTRACTEDIMAGESCOUNTER = extractedimagecounter
    IMAGESCOUNTER = imagecounter
    ARCHIVECOUNTER = archivecounter

    global PROCESSID
    PROCESSID = str(mp.current_process().pid)
    global PROCESSTARGETFOLDER
    PROCESSTARGETFOLDER = hlp.concatinatepath(c.TARGETFOLDER, PROCESSID)
    subprocess.run('mkdir \"' + PROCESSTARGETFOLDER + '\"', shell=True)


def convertsingleequationsfile(pathtoequationfile, targetsubfoldername, pathmacrostxt=None):
    macros = ''
    equationsstr = hlp.readfromfile(pathtoequationfile)
    #(equationsstr, notused) = eefa.inputfromfile(equationstr, macros)
    if pathmacrostxt is not None:
        macrostr = hlp.readfromfile(pathmacrostxt)

    root = subprocess.check_output('pwd').decode('utf-8').replace('\n', '')
    pathtotarget = hlp.concatinatepath(root, c.TARGETFOLDER)
    equations = hlp.splittexttoequations(equationsstr)
    count = 1;
    for eq in equations:
        cleaneq = hlp.cleanequation(eq)
        cvt.convertsingleformulatoimage(pathtotarget, cleaneq, subfoldername=targetsubfoldername,
                                        formulaname=hlp.getnumberstr(count), macrostr=macrostr)
        count += 1


def convertsinglefolderexcludenoneqtxt(pathtofolder, processsubfoldername):
    if not os.path.isdir(pathtofolder):
        raise NotADirectoryError('Folder behind path doesn\'t exist.')

    pathmacrostxt = hlp.concatinatepath(pathtofolder, c.MACROSTXT)
    if not os.path.isfile(pathmacrostxt):
        pathmacrostxt = None

    eqfiles = hlp.findequationtxt(pathtofolder)
    for file in eqfiles:
        processprint(file)
        pathtofile = hlp.concatinatepath(pathtofolder, file)
        processsubsubfoldername = hlp.concatinatepath(processsubfoldername, file)
        processprint(processsubsubfoldername)
        convertsingleequationsfile(pathtofile, processsubsubfoldername, pathmacrostxt)

def copytoarchivefolder(folder, absolutearchivepath, folderpath, images):
    processprint('FOLDER: ' + folder + ': Copying Images')
    absolutearchivepngfolder = hlp.concatinatepath(absolutearchivepath, c.ARCHIVEPNGFOLDER, folder)
    # if os.path.isdir(absolutearchivepngfolder):
    #    subprocess('rm  -rf ' + hlp.concatinatepath(absolutearchivepngfolder,'/*'), shell=True)
    # else:

    subprocess.run('mkdir -p ' + absolutearchivepngfolder, shell=True)
    for image in images:
        shutil.copyfile(hlp.concatinatepath(folderpath, image),
                        hlp.concatinatepath(absolutearchivepngfolder, image))

def usedatabasefunction(equationsfilename, absolutearchivepath, images, archiveid, pngfolderpath, connection):
    for image in images:
        formulastr=hlp.extractformula(hlp.concatinatepath(absolutearchivepath, equationsfilename), image)
        im = Image.open(hlp.concatinatepath(pngfolderpath, image)).convert(mode='L')
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        c.DATABASEINSERTFUNCTION(arxivid=archiveid, formulastr=formulastr, pilimage=img_byte_arr,
                                 connection=connection)


def updateImageCounter(nmbextractedimages, nmbimagecounter):
    global EXTRACTEDIMAGESCOUNTER
    global IMAGESCOUNTER

    with EXTRACTEDIMAGESCOUNTER.get_lock():
        EXTRACTEDIMAGESCOUNTER.value += nmbextractedimages

    with IMAGESCOUNTER.get_lock():
        IMAGESCOUNTER.value += nmbimagecounter
    with ARCHIVECOUNTER.get_lock():
        ARCHIVECOUNTER.value += 1

    countersave = hlp.concatinatepath(c.TARGETFOLDER, c.COUNTERRESULTFOLDER, PROCESSID + ".txt")
    time = dt.datetime.now()
    subprocess.run('touch '+countersave+'; echo \"' +
                   str(time) +
                   '\nIC: ' + str(IMAGESCOUNTER.value) +
                   '\nEIC: ' + str(EXTRACTEDIMAGESCOUNTER.value) +
                   '\nARCHIVENMB: ' + str(ARCHIVECOUNTER.value) +
                   '\" > '+countersave, shell=True)


def _workload(archive):
    global ARCHIVE
    ARCHIVE = archive
    processprint(': Starting work')
    processprint(': Converting')

    absolutearchivepath = hlp.concatinatepath(c.ROOTARCHIVES, archive)

    try:
        convertsinglefolderexcludenoneqtxt(absolutearchivepath, PROCESSID)
    except:
        processprint('Error while Converting. Abort')
        processprint(': Delete ' + archive + ' in target')
        _cleartargetofprocess()
        processprint(': End for Archive')
        return

    processprint(': Extracting')
    folders = os.listdir(PROCESSTARGETFOLDER)
    connection = c.DATABASECONNECTFUNCTION()
    for folder in folders:
        processprint('FOLDER: ' + folder + ': Extracting')
        folderpath = hlp.concatinatepath(PROCESSTARGETFOLDER, folder)
        if os.path.isdir(folderpath):
            images = hlp.extractintactimages(folderpath)
            if images:
                processprint('FOLDER: ' + folder + ': Extracted ' + str(len(images)) + ' PNGs')
                if c.DATABASEINSERTFUNCTION is None:
                    copytoarchivefolder(folder, absolutearchivepath, folderpath, images)
                else:
                    usedatabasefunction(folder, absolutearchivepath, images, archive, folderpath, connection)

            else:
                processprint('FOLDER: ' + folder + ': No Images produced')

            if c.COUNTERSET:
                updateImageCounter(nmbextractedimages=len(images), nmbimagecounter=hlp.countimages(folderpath))



    processprint(': Delete ' + archive + ' in target')
    _cleartargetofprocess()
    processprint(': End for Archive')


def deletearchivefiles(archive):
    pngfolderpath = hlp.concatinatepath(c.ROOTARCHIVES,archive,c.ARCHIVEPNGFOLDER)
    if (os.path.isdir(pngfolderpath)):
        subprocess.run('rm  -rf ' + pngfolderpath, shell=True)


# For Mainprocess: Create empty target folder and start subprocesses
#if __name__ == '__main__':
def main():
    print('Main: Archive files will be updated.')
    if (os.path.isdir(c.TARGETFOLDER)):
        subprocess.run('rm -rf ' + c.TARGETFOLDER+'/*', shell=True)
    else:
        subprocess.run('mkdir -p \"' + c.TARGETFOLDER + '\"', shell=True)

    imagescounter = mp.Value('L', 0);
    extractedimagescounter = mp.Value('L', 0);
    archivecounter = mp.Value('L', 0);

    pool = Pool(c.NMBWORKERS, initializer=init, initargs=(imagescounter, extractedimagescounter, archivecounter))

    time = dt.datetime.now()



    if c.DELETESET:
        print('Main: Delete old files.')
        pool.map(deletearchivefiles, os.listdir(c.ROOTARCHIVES))

    counterresultfolderpath = hlp.concatinatepath(c.TARGETFOLDER, c.COUNTERRESULTFOLDER)
    counterendresultpath = hlp.concatinatepath(counterresultfolderpath,'endresult.txt')
    if c.COUNTERSET:
        subprocess.run('mkdir -p \"' + counterresultfolderpath + '\"', shell=True)
        subprocess.run('rm -rf ' + counterresultfolderpath + '/*', shell=True)
        subprocess.run('touch '+counterendresultpath+'; echo \"'+str(time)+'\" > '+counterendresultpath,shell=True)

    print('Main: Start generating. ' + str(time))
    pool.map(_workload, os.listdir(c.ROOTARCHIVES))#[slice(3)])

    timenew = dt.datetime.now()
    runtime = str(timenew - time)

    if c.COUNTERSET:
        subprocess.run('touch '+counterendresultpath+'; echo \"'+
                       str(time)+'\n'+str(timenew)+'\n'+runtime +
                       '\nIC: '+str(imagescounter.value) +
                       '\nEIC: '+str(extractedimagescounter.value)+
                       '\" > '+counterendresultpath, shell=True)

        print('Countervalues:'+' Image Counter:',imagescounter.value,' ExtractedImageCounter:',extractedimagescounter.value)


    print('Main: Runtime: ' + runtime)

#c.PRINT=True
#main()
