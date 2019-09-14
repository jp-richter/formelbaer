import sys
sys.path.append('libs/tex2png')

import os
import subprocess
import constant as c
import re
from PIL import Image

# Returns a string with maximal three leading zeros
def getnumberstr(number):
    text = ''
    if number < 10:
        text += '000%i' % number
    elif number < 100:
        text += '00%i' % number
    elif number < 1000:
        text += '0%i' % number
    else:
        text += '%i' % number
    return text


def findoutnumber(imagename):
    imagename = imagename[:-len(c.PNGEXTENSION)]
    return int(imagename)

# Concatinates path segements to a single path
def concatinatepath(*pathsegments):
    iterpathsegments = iter(pathsegments)
    path = next(iterpathsegments)
    for pathsegment in iterpathsegments:
        if (path.endswith('/') and pathsegment.startswith('/')):
            path += pathsegment[1:]
        elif ((not path.endswith('/')) and (not pathsegment.startswith('/'))):
            path += '/' + pathsegment
        else:
            path += pathsegment
    return path


# Returns a string of all the content
def readfromfile(path):
    if not os.path.isfile(path):
        raise FileExistsError('File behind path doesn\'t exist.')
    tempfile = open(path, 'r')
    tempstr = tempfile.read()
    tempfile.close()
    return tempstr


# Replaces in
def replacetext(text, placeholder, string):
    return text.replace(placeholder, string)


def writefile(path, text):
    if not os.path.isfile(path):
        raise FileExistsError('File behind path doesn\'t exist.')
    temp = open(path, 'w')
    temp.write(text)
    temp.close()


# Cleans equation string from newline characters
def cleanequation(equationstr):
    equationstr = equationstr.rstrip()
    return equationstr


# Returns file names in the given folder which end with "_equation.txt"
def findequationtxt(parent):
    equations = []
    if os.path.isdir(parent):
        for file in os.listdir(parent):
            if file.endswith(c.EQUATIONTXTEXTENSION):
                equations.append(file)
    return equations

# Splits a string of equations into single equations in an array
def splittexttoequations(text):
    splitre = re.compile('\$\$\$\$\$' + '[0-9]{4}')
    equations = splitre.split(text)
    # equations = text.split(c.EQTXTHEAD)
    equations.pop(0)
    return equations


def changecurrentdirectory(path):
    return 'cd \"' + path + '\"; '


def copyfile(filepath, pathtocopy, copyname):
    try:
        mkdirprocess = subprocess.run('mkdir -p \"' + pathtocopy + '\"', shell=True)

        cpprocess = subprocess.run('cp \"' + filepath + '\" \"' + concatinatepath(pathtocopy, copyname) + '\"',
                                   shell=True)
        return 0
    except FileExistsError:
        return -1


def extractformula(pathtoequationsfile, imagename):
    equationstxt = readfromfile(pathtoequationsfile)
    imagenumber = findoutnumber(imagename)
    equationsarray = splittexttoequations(equationstxt)
    return equationsarray[imagenumber-1]

def recursivextractintactimages(folderpath, targetpath):
    if not os.path.isdir(folderpath):
        raise NotADirectoryError(targetpath + ' is not an existing Directory.')
    elif not os.path.isdir(targetpath):
        subprocess.run('mkdir -p \"' + targetpath + '\"', shell=True)


    imagearray = extractintactimages(folderpath)
    for image in imagearray:
        imagepath = concatinatepath(folderpath,image)
        copyfile(imagepath, targetpath, image)

    files = os.listdir(folderpath)

    for file in files:
        filepath = concatinatepath(folderpath,file)
        if os.path.isdir(filepath):
            targetsubpath = concatinatepath(targetpath, file)
            subprocess.run('mkdir -p \"' + targetsubpath + '\"', shell=True)
            recursivextractintactimages(filepath, targetsubpath)


def countimages(folderpath):
    if not os.path.isdir(folderpath):
        raise NotADirectoryError('Directory not found.')
    count = 0;
    for file in os.listdir(folderpath):
        absolutepathfromtex2png = concatinatepath(folderpath, file)
        if absolutepathfromtex2png.endswith(c.PNGEXTENSION):
            count+=1
    return count

# Extracts intact PNGs
# Does not work recursivly
def extractintactimages(folderpath):
    if not os.path.isdir(folderpath):
        raise NotADirectoryError('Directory not found.')
    imagearray = []
    for file in os.listdir(folderpath):
        absolutepathfromtex2png = concatinatepath(folderpath, file)
        if _isintactimage(absolutepathfromtex2png):
            imagearray.append(file)
    return imagearray

# Checks if the file is an Image and has the right width and height
def _isintactimage(path):
    if os.path.isfile(path):
        if path.endswith(c.PNGEXTENSION):
            image = Image.open(open(path, 'rb'))
            width, height = image.size
            if width == c.IMAGEWIDTH and height == c.IMAGEHEIGHT:
                extrema = image.convert("L").getextrema()
                if extrema == (255, 255):
                    return False
                return True

    return False
