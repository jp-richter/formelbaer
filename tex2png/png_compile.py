import os
import subprocess
import tempfile
from multiprocessing.pool import Pool
from PIL import Image

document="""\\documentclass[convert={{density=90,outext=.png,outname={name},subjobname={name}}}]{{standalone}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb}}
{macros}
\\begin{{document}}
\\begin{{minipage}}[c][3cm]{{15cm}}
{equation}
\\end{{minipage}}
\\end{{document}}
"""


def compile_string(formula):
    currentdir = os.getcwd()
    tmp = "tmp"
    filename = "tmp.tex"
    tex = document.format(name="tmp", equation=formula, macros="")
    with open(filename, 'w') as tf:
        tf.write(tex)
        tf.flush()
        tf.close()
        result = subprocess.run(["pdflatex", "--shell-escape", "-interaction=nonstopmode", tmp + ".tex"],
                                cwd=currentdir, universal_newlines=True)
        if result.returncode == 0:
            print("SUCCESS", tf.name)
        else:
            print("FAIL", tf.name)
    return Image.open(tmp + ".png")


def work(data):
	i,l,m,currentdir = data
#	print(data)
	for macros in [1,2,3]:
		with open(os.path.join(currentdir,str(i)+".tex"),"w") as tf:
			# with tempfile.NamedTemporaryFile(mode="w",suffix=".tex",dir=currentdir) as tf:
			def f7(seq):
				seen = set()
				seen_add = seen.add
				return [x for x in seq if not (x in seen or seen_add(x))]
			if macros==1:
				tex = document.format(name=str(i),equation=l,macros=m)
			elif macros==2:
				tex = document.format(name=str(i),equation=l,macros=f7(m))
			else:
				tex = document.format(name=str(i),equation=l,macros="\n".join(["%"+x for x in m.split("\n")]))
			tf.write(tex)
			tf.flush()
			result = subprocess.run(["pdflatex","--shell-escape","-interaction=nonstopmode",str(i)],cwd=currentdir,universal_newlines=True,stdout=subprocess.PIPE)
			if result.returncode==0:
				print("SUCCESS",tf.name)
				break
			else:
				print("FAIL",tf.name)
			tf.close()


def compile_files(root):
    tp = Pool(1)
    macro = False
    m = ""
    with open(root, "r") as f:
        lastresult = None
        for l in f:
            if l.startswith("=="):
                if lastresult is not None:
                    print("wait for it")
                    lastresult.wait()
                file = l.split("==")[1]
                currentdir = os.path.join("", file)
                print(currentdir)
                os.makedirs(currentdir, exist_ok=True)
                i = 1
            elif l.startswith("BEGIN_MACROS"):
                macro = True
                m = ""
            elif l.startswith("END_MACROS"):
                macro = False
            elif macro == True:
                m += l
            else:
                # m = "\n".join(list(set(m.split("\n"))))
                lastresult = tp.apply_async(work, ((i, l, m, currentdir),)).wait()
                # print("starting runner",i,lastresult.get())
                i += 1

    tp.close()
    tp.join()

    print(tp)

if __name__ == "__main__":
    compile_string("\\begin{equation}(a + b)^2\\end{equation}").show()