import subprocess

ROOTTEX2PNG = subprocess.check_output('pwd').decode('utf-8').replace('\n', '')
# Parent folder of the archives
ROOTARCHIVES = '/home/fpdmws2018/arxiv/tex-files/'
# Number of Processes for Process Pools
NMBWORKERS = 20

##########function pointer############
#
#
DATABASEINSERTFUNCTION = None
#


##########folder/file-names############
#
#
TARGETFOLDER = 'target'
TARGETSUBFOLDERDEFAULT = 'default'
TARGETSUBFOLDERIMAGEDEFAULT = 'defaultimage'
TEMPLATETEXNAME = 'template.tex'
TEMPLATEMACROTEXNAME = 'template-with-macros.tex'
TEMPLATEFORMULATEXNAME = 'template-formula.tex'
ARCHIVEPNGFOLDER = 'Formula-PNGs'
COUNTERRESULTFOLDER = 'INTERMEDIATE_COUNTER_RESULTS'
#

##########flags/arguments################
#
#
ARCHIVEARGUMENT = 'archive'
#
PRINTFLAG = '-p'
PRINTSET = False
#
DELETEFLAG = '-d'
DELETESET = False
#
TARGETSUBFOLDERIMAGEFLAG = '-tsi'
TARGETSUBFOLDERIMAGE = TARGETSUBFOLDERIMAGEDEFAULT
#
TARGETSUBFOLDERFLAG = '-ts'
TARGETSUBFOLDER = TARGETSUBFOLDERDEFAULT
#
COUNTERFLAG = '-c'
COUNTERSET = False
#

##########images#########################
#
#
PNGEXTENSION = '.png'
#
IMAGEHEIGHT = 32
#
IMAGEWIDTH = 333
#

##########Equations#####################
#
# Defines the math enviroments which will be matched
MATHENVIROMENTS = ["equation", "equations", "displaymath", "eqnarray", "align", "multiline", "gather"]
#
# Name extension for the text file
EQUATIONTXTEXTENSION = '_equations.txt'
#
# Headline before every equation
EQTXTHEAD = '$$$$$'
#

##########MACROTXT#######################
#
# Name extension for the text file
MACROSTXT = 'macros.txt'
#

##########latex-formula-template.tex######
#
# Placeholder where the equation will be inserted
EQUATIONPH = '$$$$$'
#
# Placeholder where the definitions(newcommand, def) will be inserted
MARCOPH = '%%%%%'
#
