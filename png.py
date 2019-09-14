import os
import argparse
import subprocess

template = '''
\\begin{{equation}}
{formular}
\\end{{equation}}
'''

def tex(string, path, filename):
	with open(path + '/' + filename + '.tex','w') as f:
		f.write(template.format(formular=string))

def cd():
	cmd = ['cd', os.path.abspath('png.py')[:-6] + '\\tex2png']

	print(os.path.abspath('png.py')[:-7] + '\\tex2png')

	try:
		subprocess.run(cmd, cwd=os.path.abspath('png.py')[:-7] + '\\tex2png', stdout=subprocess.DEVNULL, timeout=1)
	except:
		print('Could not change working directory.')

def tool(path):
	cmd = ['python', 'tex2png.py', path]

	try:
		subprocess.run(cmd, cwd=os.path.abspath('png.py')[:-7] + '\\tex2png', stdout=subprocess.DEVNULL, timeout=1)
	except error as e:
		print(e)

# funktioniert nicht auf windows ??

test = 'mein test string 3 = 5 \\sigma'
path = '~/Desktop'
name = 'test'
desktop = os.path.normpath(os.path.expanduser("~/Desktop"))

tex(test, desktop, name)
tool(desktop + '/' + name + '.tex')
	