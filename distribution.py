import re
import sys
import os
import config

REGEX = {}  # (token id, regex)

# 0: TokenInfo('root', 1, 0, '\\sqrt{{{}}}'),
REGEX[0] = r'sqrt'

# 1: TokenInfo('fac', 1, 1, '{}!'),
REGEX[1] = r'!'

# 2: TokenInfo('max', 1, 2, 'max {}'),
REGEX[2] = r'max'

# 3: TokenInfo('min', 1, 3, 'min {}'),
REGEX[3] = r'min'

# 4: TokenInfo('argmax', 1, 4, 'argmax {}'),
REGEX[4] = r'arg(\b)*max'

# 5: TokenInfo('argmin', 1, 5, 'argmin {}'),
REGEX[5] = r'arg(\b)*min'

# 6: TokenInfo('inverse', 1, 6, '{}^{{-1}}'),
REGEX[6] = r'^{-1}'

# 7: TokenInfo('sin', 1, 7, 'sin {}'),
REGEX[7] = r'sin'

# 8: TokenInfo('cos', 1, 8, 'cos {}'),
REGEX[8] = r'cos'

# 9: TokenInfo('tan', 1, 9, 'tan {}'),
REGEX[9] = r'tan'

# 10: TokenInfo('sinh', 1, 10, 'sinh {}'),
REGEX[10] = r'sinh'

# 11: TokenInfo('cosh', 1, 11, 'cosh {}'),
REGEX[11] = r'cosh'

# 12: TokenInfo('tanh', 1, 12, 'tanh {}'),
REGEX[12] = r'tanh'

# 13: TokenInfo('sigmoid', 1, 13, '\\sigma({})'),
REGEX[13] = r'sigma\('

# 14: TokenInfo('transpose', 1, 14, '{}^T'),
REGEX[14] = r'\^T'

# 15: TokenInfo('prime', 1, 15, '{}\''),
REGEX[15] = r"\^'"

# 16: TokenInfo('absolute', 1, 16, '|{}|'),
REGEX[16] = r'\|(\b)*\|'

# 17: TokenInfo('norm', 1, 17, '||{}||'),
REGEX[17] = r'\|\|(\b)*\|\|'

# 18: TokenInfo('mathbbe', 1, 18, '\\mathbb{{E}}[{}]'),
REGEX[18] = r'mathbb\{E\}'

# 19: TokenInfo('mathbbp', 1, 19, '\\mathbb{{P}}[{}]'),
REGEX[19] = r'mathbb\{P\}'

# 20: TokenInfo('maxsub', 2, 20, 'max_{{{}}} {}'),
REGEX[20] = r'max_'

# 21: TokenInfo('minsub', 2, 21, 'min_{{{}}} {}'),
REGEX[21] = r'min_'

# 22: TokenInfo('argmaxsub', 2, 22, 'argmax_{{{}}} {}'),
REGEX[22] = r'arg(\b)*max_'

# 23: TokenInfo('argminsub', 2, 23, 'argmin_{{{}}} {}'),
REGEX[23] = r'arg(\b)*min_'

# 24: TokenInfo('mathbbesub', 2, 24, '\\mathbb{{E}}_{{{}}}[{}]'),
REGEX[24] = r'mathbb\{E\}_'

# 25: TokenInfo('mathbbpsub', 2, 25, '\\mathbb{{P}}_{{{}}}[{}]'),
REGEX[25] = r'mathbb\{E\}_'

# 26: TokenInfo('add', 2, 26, '{} + {}'),
REGEX[26] = r'\+'

# 27: TokenInfo('sub', 2, 27, '{} - {}'),
REGEX[27] = r'-'

# 28: TokenInfo('dot', 2, 28, '{} \\cdot {}'),
REGEX[28] = r'cdot'

# 29: TokenInfo('cross', 2, 29, '{} \\times {}'),
REGEX[29] = r'times'

# 30: TokenInfo('fract', 2, 30, '\\frac{{{}}}{{{}}}'),
REGEX[30] = r'frac'

# 31: TokenInfo('mod', 2, 31, '{} mod {}'),
REGEX[31] = r'mod'

# 32: TokenInfo('power', 2, 32, '{}^{{{}}}'),
REGEX[32] = r'\^'

# 33: TokenInfo('derive', 2, None, '\\frac{{\\delta{}}}{{\\delta {}}}'),
REGEX[33] = r'frac(\b)*delta'

# 34: TokenInfo('sum', 3, 33, '\\sum\\nolimits_{{{}}}^{{{}}} {}'),
REGEX[34] = r'sum'

# 35: TokenInfo('product', 3, 34, '\\prod\\nolimits_{{{}}}^{{{}}} {}'),
REGEX[35] = r'prod'

# 36: TokenInfo('integral', 3, 35, '\\int\\nolimits_{{{}}}^{{{}}} {}'),
REGEX[36] = r'int'

# 37: TokenInfo('equals', 2, 36, '{} = {}'),
REGEX[37] = r'='

# 38: TokenInfo('lesser', 2, 37, '{} < {}'),
REGEX[38] = r'le'

# 39: TokenInfo('greater', 2, 38, '{} > {}'),
REGEX[39] = r'ge'

# 40: TokenInfo('lessereq', 2, 39, '{} \\leq {}'),
REGEX[40] = r'leq'

# 41: TokenInfo('greatereq', 2, 40, '{} \\geq {}'),
REGEX[41] = r'geq'

# 42: TokenInfo('subset', 2, 41, '{} \\subset {}'),
REGEX[42] = r'subset'

# 43: TokenInfo('subseteq', 2, 42, '{} \\subseteq {}'),
REGEX[43] = r'subseteq'

# 44: TokenInfo('union', 2, 43, '{} \\cup {}'),
REGEX[44] = r'cup'

# 45: TokenInfo('difference', 2, 44, '{} \\cap {}'),
REGEX[45] = r'cap'

# 46: TokenInfo('elementof', 2, 45, '{} \\in {}'),
REGEX[46] = r'in'

# 47: TokenInfo('apply', 2, 46, '{}({})'),
REGEX[47] = r'\w\((\d)*\)'

# 48: TokenInfo('brackets', 1, 47, '({})'),
REGEX[48] = r'\((\d)*\)'

# 49: TokenInfo(u'\u0393', 0, 50, '\\Gamma'),
REGEX[49] = r'Gamma'

# 50: TokenInfo(u'\u0394', 0, 51, '\\Delta'),
REGEX[50] = r'Delta'

# 51: TokenInfo(u'\u0398', 0, 55, '\\Theta'),
REGEX[51] = r'Theta'

# 52: TokenInfo(u'\u039B', 0, 58, '\\Lambda'),
REGEX[52] = r'Lambda'

# 53: TokenInfo(u'\u039E', 0, 61, '\\Xi'),
REGEX[53] = r'Xi'

# 54: TokenInfo(u'\u03A0', 0, 63, '\\Pi'),
REGEX[54] = r'Pi'

# 55: TokenInfo(u'\u03A3', 0, 65, '\\Sigma'),
REGEX[55] = r'Sigma'

# 56: TokenInfo(u'\u03A5', 0, 67, '\\Upsilon'),
REGEX[56] = r'Upsilon'

# 57: TokenInfo(u'\u03A6', 0, 68, '\\Phi'),
REGEX[57] = r'Phi'

# 58: TokenInfo(u'\u03A8', 0, 70, '\\Psi'),
REGEX[58] = r'Psi'

# 59: TokenInfo(u'\u03A9', 0, 71, '\\Omega'),
REGEX[59] = r'Omega'

# 60: TokenInfo(u'\u03B1', 0, 72, '\\alpha'),
REGEX[60] = r'alpha'

# 61: TokenInfo(u'\u03B2', 0, 73, '\\beta'),
REGEX[61] = r'beta'

# 62: TokenInfo(u'\u03B3', 0, 74, '\\gamma'),
REGEX[62] = r'gamma'

# 63: TokenInfo(u'\u03B4', 0, 75, '\\delta'),
REGEX[63] = r'delta'

# 64: TokenInfo(u'\u03B5', 0, 76, '\\epsilon'),
REGEX[64] = r'epsilon'

# 65: TokenInfo(u'\u03B6', 0, 77, '\\zeta'),
REGEX[65] = r'zeta'

# 66: TokenInfo(u'\u03B7', 0, 78, '\\eta'),
REGEX[66] = r'eta'

# 67: TokenInfo(u'\u03B8', 0, 79, '\\theta'),
REGEX[67] = r'theta'

# 68: TokenInfo(u'\u03B9', 0, 80, '\\iota'),
REGEX[68] = r'iota'

# 69: TokenInfo(u'\u03BA', 0, 81, '\\kappa'),
REGEX[69] = r'kappa'

# 70: TokenInfo(u'\u03BB', 0, 82, '\\lambda'),
REGEX[70] = r'lambda'

# 71: TokenInfo(u'\u03BC', 0, 83, '\\mu'),
REGEX[71] = r'mu'

# 72: TokenInfo(u'\u03BD', 0, 84, '\\nu'),
REGEX[72] = r'nu'

# 73: TokenInfo(u'\u03BE', 0, 85, '\\xi'),
REGEX[73] = r'xi'

# 74: TokenInfo(u'\u03C0', 0, 87, '\\pi'),
REGEX[74] = r'pi'

# 75: TokenInfo(u'\u03C1', 0, 88, '\\rho'),
REGEX[75] = r'rho'

# 76: TokenInfo(u'\u03C3', 0, 89, '\\sigma'),
REGEX[76] = r'sigma'

# 77: TokenInfo(u'\u03C4', 0, 90, '\\tau'),
REGEX[77] = r'tau'

# 78: TokenInfo(u'\u03C5', 0, 91, '\\upsilon'),
REGEX[78] = r'upsilon'

# 79: TokenInfo(u'\u03C6', 0, 92, '\\phi'),
REGEX[79] = r'phi'

# 80: TokenInfo(u'\u03C7', 0, 93, '\\chi'),
REGEX[80] = r'chi'

# 81: TokenInfo(u'\u03C8', 0, 94, '\\psi'),
REGEX[81] = r'psi'

# 82: TokenInfo(u'\u03C9', 0, 95, '\\omega'),
REGEX[82] = r'omega'

# 83: TokenInfo('A', 0, 96, 'A'),
REGEX[83] = r'\(A+A+A\)+A\b'

# 84: TokenInfo('B', 0, 97, 'B'),
REGEX[84] = r'\(B+B+B\)+B\b'

# 85: TokenInfo('C', 0, 98, 'C'),
REGEX[85] = r'\(C+C+C\)+C\b'

# 86: TokenInfo('D', 0, 99, 'D'),
REGEX[86] = r'\(D+D+D\)+D\b'

# 87: TokenInfo('E', 0, 100, 'E'),
REGEX[87] = r'\(E+E+E\)+E\b'

# 88: TokenInfo('F', 0, 101, 'F'),
REGEX[88] = r'\(F+F+F\)+F\b'

# 89: TokenInfo('G', 0, 102, 'G'),
REGEX[89] = r'\(G+G+G\)+G\b'

# 90: TokenInfo('H', 0, 103, 'H'),
REGEX[90] = r'\(H+H+H\)+H\b'

# 91: TokenInfo('I', 0, 104, 'I'),
REGEX[91] = r'\(I+I+I\)+I\b'

# 92: TokenInfo('J', 0, 105, 'J'),
REGEX[92] = r'\(J+J+J\)+J\b'

# 93: TokenInfo('K', 0, 106, 'K'),
REGEX[93] = r'\(K+K+K\)+K\b'

# 94: TokenInfo('L', 0, 107, 'L'),
REGEX[94] = r'\(L+L+L\)+L\b'

# 95: TokenInfo('M', 0, 108, 'M'),
REGEX[95] = r'\(M+M+M\)+M\b'

# 96: TokenInfo('N', 0, 109, 'N'),
REGEX[96] = r'\(N+N+N\)+N\b'

# 97: TokenInfo('O', 0, 110, 'O'),
REGEX[97] = r'\(O+O+O\)+O\b'

# 98: TokenInfo('P', 0, 111, 'P'),
REGEX[98] = r'\(P+P+P\)+P\b'

# 99: TokenInfo('Q', 0, 112, 'Q'),
REGEX[99] = r'\(Q+Q+Q\)+Q\b'

# 100: TokenInfo('R', 0, 113, 'R'),
REGEX[100] = r'\(R+R+R\)+R\b'

# 101: TokenInfo('S', 0, 114, 'S'),
REGEX[101] = r'\(S+S+S\)+S\b'

# 102: TokenInfo('T', 0, 115, 'T'),
REGEX[102] = r'\(T+T+T\)+T\b'

# 103: TokenInfo('U', 0, 116, 'U'),
REGEX[103] = r'\(U+U+U\)+U\b'

# 104: TokenInfo('V', 0, 117, 'V'),
REGEX[104] = r'\(V+V+V\)+V\b'

# 105: TokenInfo('W', 0, 118, 'W'),
REGEX[105] = r'\(W+W+W\)+W\b'

# 106: TokenInfo('X', 0, 119, 'X'),
REGEX[106] = r'\(X+X+X\)+X\b'

# 107: TokenInfo('Y', 0, 120, 'Y'),
REGEX[107] = r'\(Y+Y+Y\)+Y\b'

# 108: TokenInfo('Z', 0, 121, 'Z'),
REGEX[108] = r'\(Z+Z+Z\)+Z\b'

# 109: TokenInfo('a', 0, 122, 'a'),
REGEX[109] = r'\(a+a+a\)+a\b'

# 110: TokenInfo('b', 0, 123, 'b'),
REGEX[110] = r'\(b+b+b\)+b\b'

# 111: TokenInfo('c', 0, 124, 'c'),
REGEX[111] = r'\(c+c+c\)+c\b'

# 112: TokenInfo('d', 0, 125, 'd'),
REGEX[112] = r'\(d+d+d\)+d\b'

# 113: TokenInfo('e', 0, 126, 'e'),
REGEX[113] = r'\(e+e+e\)+e\b'

# 114: TokenInfo('f', 0, 127, 'f'),
REGEX[114] = r'\(f+f+f\)+f\b'

# 115: TokenInfo('g', 0, 128, 'g'),
REGEX[115] = r'\(g+g+g\)+g\b'

# 116: TokenInfo('h', 0, 129, 'h'),
REGEX[116] = r'\(h+h+h\)+h\b'

# 117: TokenInfo('i', 0, 130, 'i'),
REGEX[117] = r'\(i+i+i\)+i\b'

# 118: TokenInfo('j', 0, 131, 'j'),
REGEX[118] = r'\(j+j+j\)+j\b'

# 119: TokenInfo('k', 0, 132, 'k'),
REGEX[119] = r'\(k+k+k\)+k\b'

# 120: TokenInfo('l', 0, 133, 'l'),
REGEX[120] = r'\(l+l+l\)+l\b'

# 121: TokenInfo('m', 0, 134, 'm'),
REGEX[121] = r'\(m+m+m\)+m\b'

# 122: TokenInfo('n', 0, 135, 'n'),
REGEX[122] = r'\(n+n+n\)+n\b'

# 123: TokenInfo('o', 0, 136, 'o'),
REGEX[123] = r'\(o+o+o\)+o\b'

# 124: TokenInfo('p', 0, 137, 'p'),
REGEX[124] = r'\(p+p+p\)+p\b'

# 125: TokenInfo('q', 0, 138, 'q'),
REGEX[125] = r'\(q+q+q\)+q\b'

# 126: TokenInfo('r', 0, 139, 'r'),
REGEX[126] = r'\(r+r+r\)+r\b'

# 127: TokenInfo('s', 0, 140, 's'),
REGEX[127] = r'\(s+s+s\)+s\b'

# 128: TokenInfo('t', 0, 141, 't'),
REGEX[128] = r'\(t+t+t\)+t\b'

# 129: TokenInfo('u', 0, 142, 'u'),
REGEX[129] = r'\(u+u+u\)+u\b'

# 130: TokenInfo('v', 0, 143, 'v'),
REGEX[130] = r'\(v+v+v\)+v\b'

# 131: TokenInfo('w', 0, 144, 'w'),
REGEX[131] = r'\(w+w+w\)+w\b'

# 132: TokenInfo('x', 0, 145, 'x'),
REGEX[132] = r'\(x+x+x\)+x\b'

# 133: TokenInfo('y', 0, 146, 'y'),
REGEX[133] = r'\(y+y+y\)+y\b'

# 134: TokenInfo('z', 0, 147, 'z'),
REGEX[134] = r'\(z+z+z\)+z\b'

# 135: TokenInfo('1', 0, 148, '1'),
REGEX[135] = r'\(1+1+1\)+1\b'

# 136: TokenInfo('2', 0, 149, '2'),
REGEX[136] = r'\(2+2+2\)+2\b'

# 137: TokenInfo('3', 0, 150, '3'),
REGEX[137] = r'\(3+3+3\)+3\b'

# 138: TokenInfo('4', 0, 151, '4'),
REGEX[138] = r'\(4+4+4\)+4\b'

# 139: TokenInfo('5', 0, 152, '5'),
REGEX[139] = r'\(5+5+5\)+5\b'

# 140: TokenInfo('6', 0, 153, '6'),
REGEX[140] = r'\(6+6+6\)+6\b'

# 141: TokenInfo('7', 0, 154, '7'),
REGEX[141] = r'\(7+7+7\)+7\b'

# 142: TokenInfo('8', 0, 155, '8'),
REGEX[142] = r'\(8+8+8\)+8\b'

# 143: TokenInfo('9', 0, 156, '9'),
REGEX[143] = r'\(9+9+9\)+9\b'

# 144: TokenInfo('0', 0, 157, '0')
REGEX[144] = r'\(0+0+0\)+0\b'

OCCURENCES = [0] * len(REGEX.keys())
PATTERNS = []


def find_all(string):
    for id, pattern in PATTERNS:
        matches = pattern.findall(string)
        OCCURENCES[id] += len(matches)


def scan(directory):
    for id, regex in REGEX.items():
        PATTERNS.append((id, re.compile(regex)))

    iterator = os.scandir(directory)
    for entry in iterator:

        if entry.is_file():
            if entry.name.endswith('.tex'):
                with open(directory + '/' + entry.name, 'r') as file:
                    string = file.read()
                    find_all(string)

        if entry.is_dir():
            scan(directory + '/' + entry.name)

    save(config.paths.distribution_bias)


def save(path):
    with open(path, "w") as file:
        for count in OCCURENCES:
            file.write(str(count) + ',')


def load(path):
    if not os.path.exists(path):
        return None

    with open(path, 'r') as file:
        data = file.read()
        ls = data.split(',')

    occurrences = [int(o) for o in ls]
    total = sum(occurrences)
    distribution = [o / total for o in occurrences]

    return distribution


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('Please specify a directory to scan for latex documents.')

    path = sys.argv[1]

    if not os.path.exists(path):
        raise ValueError('Path does not exist.')

    scan(sys.argv[1])
