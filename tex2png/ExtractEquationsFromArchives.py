import sys
sys.path.append('libs/tex2png')

import os
import re
from multiprocessing import Pool
import datetime
import constant as c
import helpers as hlp
import multiprocessing as mp


ARCHIVESWITHFORMULA = None
EXTRACTEDFORMULA = None
NUMBEROFFORMULASPERARCHIVE = None


def init(a, b, c):
    global EXTRACTEDFORMULA
    global ARCHIVESWITHFORMULA
    global NUMBEROFFORMULASPERARCHIVE

    EXTRACTEDFORMULA = b
    ARCHIVESWITHFORMULA = a
    NUMBEROFFORMULASPERARCHIVE = c


def updateImageCounter(nmbarchiveswithformula, nmbextractedformula, nmbformulaperarchive):
    global ARCHIVESWITHFORMULA
    global EXTRACTEDFORMULA
    global NUMBEROFFORMULASPERARCHIVE

    with ARCHIVESWITHFORMULA.get_lock():
        ARCHIVESWITHFORMULA.value += nmbarchiveswithformula
    with EXTRACTEDFORMULA.get_lock():
        EXTRACTEDFORMULA.value += nmbextractedformula
    if nmbformulaperarchive >= 0:
        with NUMBEROFFORMULASPERARCHIVE.get_lock():
            if nmbformulaperarchive > 300:
                NUMBEROFFORMULASPERARCHIVE[301] += 1 #max(NUMBEROFFORMULASPERARCHIVE[701],nmbformulaperarchive)
            elif nmbformulaperarchive >= 0:
                NUMBEROFFORMULASPERARCHIVE[nmbformulaperarchive] += 1

def saveEquations(equations, path):
    text = ''
    count = 1
    for equation in equations:
        numberstr = hlp.getnumberstr(count)
        text += c.EQTXTHEAD + numberstr + '\n' + equation
        count += 1
    try:
        with open(path, 'w+') as file:
            file.write(text)
            # print('Writing file ' + path)
    except IOError as error:
        print('Write equations to %s failed' % path)
        print(error)


def convertInputToEquations(inputString):
    commands = []
    commandsRE = []
    for env in c.MATHENVIROMENTS:
        commands.append(env)
        commandsRE.append(env)
    # Add * to every command (difference between RE commands and "normal" commands
    length = len(commands)
    for i in range(0, length, 1):
        commandsRE.append(commands[i] + "\\*")
        commands.append(commands[i] + "*")

    newLineRE = re.compile(r'\n', re.IGNORECASE)
    results = []
    counts = []
    for index in range(0, len(commands), 1):
        commandRE = commandsRE[index]
        command = commands[index]
        startstring = '\\\\begin\\{%s\\}' % commandRE
        start = re.compile(startstring, re.IGNORECASE)
        endstring = '\\\\end\\{%s\\}' % commandRE
        end = re.compile(endstring, re.IGNORECASE)

        count = 0
        startresults = start.split(inputString)
        for i in range(1, len(startresults), 1):
            count += len(startresults[i - 1]) + len(startstring)
            endResults = end.split(startresults[i])
            if len(endResults) > 0:
                result = endResults[0]
                # result = ('\\begin{%s}\n' % command) + endResults[0] + ('\\end{%s}\n\n\n\n' % command)
                # Replace new line character
                allLines = newLineRE.split(result)
                if len(allLines) > 1:
                    count2 = count
                    for line in allLines:
                        line = line.strip()
                        if len(line) > 0 and (not line == '\n') and (not line == r'\\'):
                            linewithcommand = ('\\begin{%s}' % command) + line + ('\\end{%s}\n' % command)
                            results.append(linewithcommand)
                            counts.append(count2)
                            count2 += len(line)
                else:
                    result = ('\\begin{%s}' % command) + endResults[0] + ('\\end{%s}\n' % command)
                    results.append(result)
                    counts.append(count)
    # Sort results by counts
    for i in range(0, len(results), 1):
        index = i
        for j in range(i + 1, len(results), 1):
            if counts[index] > counts[j]:
                index = j
        swap = results[index]
        results[index] = results[i]
        results[i] = swap
        swap = counts[index]
        counts[index] = counts[i]
        counts[i] = swap
    #print(results)
    return results


def inputfromfile(lines, macros):
    lines = lines.split("\n")
    inputString = ''
    commentedLine = re.compile('%.*')
    macroLine = re.compile('\\\\(newcommand|def).*')
    beginLine = re.compile('\\\\begin.*')
    searchMacro = True
    for line in lines:
        if not commentedLine.match(line):
            if searchMacro and beginLine.match(line):
                searchMacro = False
            if searchMacro and macroLine.match(line):
                macros += line
            elif not searchMacro:
                inputString += line
    return (inputString, macros)


def readfilesinarchive(archive):
    macros = ''
    countFormula = 0
    for filename in os.listdir(archive):
        if filename.endswith(".tex"):
            path = os.path.join(archive, filename)
            # print('Read %s' % path)
            inputstring = ''
            for encoding in ['utf-8', 'cp1252']:
                try:
                    with open(path, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                        # print('%d' % len(lines))
                        (inputstring, macros) = inputfromfile(lines, macros)
                        break
                except:
                    error = 'Wrong encoding ' + encoding
                    # print(error)
            if len(inputstring) > 0:
                # print(path)
                equations = convertInputToEquations(inputstring)
                if len(equations) > 0:
                    countFormula += len(equations)
                    updateImageCounter(0, len(equations), -1)
                    # print("Found %d equations" % len(equations))
                    newname = filename.replace(".tex", c.EQUATIONTXTEXTENSION)
                    savePath = os.path.join(archive, newname)
                    saveEquations(equations, savePath)
    if countFormula > 0:
        updateImageCounter(1, 0, countFormula)
    else:
        updateImageCounter(0, 0, countFormula)
    if countFormula == 2735:
        print(archive)
    if len(macros) > 0:
        macrosPath = os.path.join(archive, c.MACROSTXT)
        try:
            with open(macrosPath, 'w+') as file:
                file.write(macros)
                # print('Writing file ' + path)
        except IOError as error:
            print('Write macros to %s failed' % macrosPath)
            print(error)

COUNTER = 0;
def workload(archive):
    parent = c.ROOTARCHIVES + archive
    if os.path.isdir(parent):
        # print('Read files in ' + parent)
        readfilesinarchive(parent)


if __name__ == '__main__':
    archiveswithformula = mp.Value('i', 0)
    extractedformula = mp.Value('i', 0)
    numberofformularperarchive = mp.Array("i", [0]*302)
    pool = Pool(c.NMBWORKERS, initializer=init, initargs=(archiveswithformula, extractedformula, numberofformularperarchive))
    time = datetime.datetime.now()
    pool.map(workload, os.listdir(c.ROOTARCHIVES))

    print(datetime.datetime.now() - time)

    print('Number archives %i' % len(os.listdir(c.ROOTARCHIVES)))
    print('Archives with formular %i' % archiveswithformula.value)
    print('number of formulars %i' % extractedformula.value)

    for i in range(0, 302):
        print("%i archives with %i formular" % (numberofformularperarchive[i], i))
