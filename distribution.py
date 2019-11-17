from config import paths

import re
import sys
import os
import tokens
import numpy as np

TOKEN_SPECIFICATION = {}  # (token id, regex)

# 0: TokenInfo('root', 1, 0, '\\sqrt{{{}}}'),
TOKEN_SPECIFICATION[0] = r'sqrt'

# 1: TokenInfo('fac', 1, 1, '{}!'),
TOKEN_SPECIFICATION[1] = r'!'

# 2: TokenInfo('max', 1, 2, 'max {}'),
TOKEN_SPECIFICATION[2] = r'max'

# 3: TokenInfo('min', 1, 3, 'min {}'),
TOKEN_SPECIFICATION[3] = r'min'

# 4: TokenInfo('argmax', 1, 4, 'argmax {}'),
TOKEN_SPECIFICATION[4] = r'arg(\b)*max'

# 5: TokenInfo('argmin', 1, 5, 'argmin {}'),
TOKEN_SPECIFICATION[5] = r'arg(\b)*min'

# 6: TokenInfo('inverse', 1, 6, '{}^{{-1}}'),
TOKEN_SPECIFICATION[6] = r'^{-1}'

# 7: TokenInfo('sin', 1, 7, 'sin {}'),
TOKEN_SPECIFICATION[7] = r'sin'

# 8: TokenInfo('cos', 1, 8, 'cos {}'),
TOKEN_SPECIFICATION[8] = r'cos'

# 9: TokenInfo('tan', 1, 9, 'tan {}'),
TOKEN_SPECIFICATION[9] = r'tan'

# 10: TokenInfo('sinh', 1, 10, 'sinh {}'),
TOKEN_SPECIFICATION[10] = r'sinh'

# 11: TokenInfo('cosh', 1, 11, 'cosh {}'),
TOKEN_SPECIFICATION[11] = r'cosh'

# 12: TokenInfo('tanh', 1, 12, 'tanh {}'),
TOKEN_SPECIFICATION[12] = r'tanh'

# 13: TokenInfo('sigmoid', 1, 13, '\\sigma({})'),
TOKEN_SPECIFICATION[13] = r'sigma\('

# 14: TokenInfo('transpose', 1, 14, '{}^T'),
TOKEN_SPECIFICATION[14] = r'\^T'

# 15: TokenInfo('prime', 1, 15, '{}\''),
TOKEN_SPECIFICATION[15] = r"\^'"

# 16: TokenInfo('absolute', 1, 16, '|{}|'),
TOKEN_SPECIFICATION[16] = r'\|(\b)*\|'

# 17: TokenInfo('norm', 1, 17, '||{}||'),
TOKEN_SPECIFICATION[17] = r'\|\|(\b)*\|\|'

# 18: TokenInfo('mathbbe', 1, 18, '\\mathbb{{E}}[{}]'),
TOKEN_SPECIFICATION[18] = r'mathbb\{E\}'

# 19: TokenInfo('mathbbp', 1, 19, '\\mathbb{{P}}[{}]'),
TOKEN_SPECIFICATION[19] = r'mathbb\{P\}'

# 20: TokenInfo('maxsub', 2, 20, 'max_{{{}}} {}'),
TOKEN_SPECIFICATION[20] = r'max_'

# 21: TokenInfo('minsub', 2, 21, 'min_{{{}}} {}'),
TOKEN_SPECIFICATION[21] = r'min_'

# 22: TokenInfo('argmaxsub', 2, 22, 'argmax_{{{}}} {}'),
TOKEN_SPECIFICATION[22] = r'arg(\b)*max_'

# 23: TokenInfo('argminsub', 2, 23, 'argmin_{{{}}} {}'),
TOKEN_SPECIFICATION[23] = r'arg(\b)*min_'

# 24: TokenInfo('mathbbesub', 2, 24, '\\mathbb{{E}}_{{{}}}[{}]'),
TOKEN_SPECIFICATION[24] = r'mathbb\{E\}_'

# 25: TokenInfo('mathbbpsub', 2, 25, '\\mathbb{{P}}_{{{}}}[{}]'),
TOKEN_SPECIFICATION[25] = r'mathbb\{E\}_'

# 26: TokenInfo('add', 2, 26, '{} + {}'),
TOKEN_SPECIFICATION[26] = r'\+'

# 27: TokenInfo('sub', 2, 27, '{} - {}'),
TOKEN_SPECIFICATION[27] = r'-'

# 28: TokenInfo('dot', 2, 28, '{} \\cdot {}'),
TOKEN_SPECIFICATION[28] = r'cdot'

# 29: TokenInfo('cross', 2, 29, '{} \\times {}'),
TOKEN_SPECIFICATION[29] = r'times'

# 30: TokenInfo('fract', 2, 30, '\\frac{{{}}}{{{}}}'),
TOKEN_SPECIFICATION[30] = r'frac'

# 31: TokenInfo('mod', 2, 31, '{} mod {}'),
TOKEN_SPECIFICATION[31] = r'mod'

# 32: TokenInfo('power', 2, 32, '{}^{{{}}}'),
TOKEN_SPECIFICATION[32] = r'\^'

# 33: TokenInfo('derive', 2, None, '\\frac{{\\delta{}}}{{\\delta {}}}'),
TOKEN_SPECIFICATION[33] = r'frac(\b)*delta'

# 34: TokenInfo('sum', 3, 33, '\\sum\\nolimits_{{{}}}^{{{}}} {}'),
TOKEN_SPECIFICATION[34] = r'sum'

# 35: TokenInfo('product', 3, 34, '\\prod\\nolimits_{{{}}}^{{{}}} {}'),
TOKEN_SPECIFICATION[35] = r'prod'

# 36: TokenInfo('integral', 3, 35, '\\int\\nolimits_{{{}}}^{{{}}} {}'),
TOKEN_SPECIFICATION[36] = r'int'

# 37: TokenInfo('equals', 2, 36, '{} = {}'),
TOKEN_SPECIFICATION[37] = r'='

# 38: TokenInfo('lesser', 2, 37, '{} < {}'),
TOKEN_SPECIFICATION[38] = r'le'

# 39: TokenInfo('greater', 2, 38, '{} > {}'),
TOKEN_SPECIFICATION[39] = r'ge'

# 40: TokenInfo('lessereq', 2, 39, '{} \\leq {}'),
TOKEN_SPECIFICATION[40] = r'leq'

# 41: TokenInfo('greatereq', 2, 40, '{} \\geq {}'),
TOKEN_SPECIFICATION[41] = r'geq'

# 42: TokenInfo('subset', 2, 41, '{} \\subset {}'),
TOKEN_SPECIFICATION[42] = r'subset'

# 43: TokenInfo('subseteq', 2, 42, '{} \\subseteq {}'),
TOKEN_SPECIFICATION[43] = r'subseteq'

# 44: TokenInfo('union', 2, 43, '{} \\cup {}'),
TOKEN_SPECIFICATION[44] = r'cup'

# 45: TokenInfo('difference', 2, 44, '{} \\cap {}'),
TOKEN_SPECIFICATION[45] = r'cap'

# 46: TokenInfo('elementof', 2, 45, '{} \\in {}'),
TOKEN_SPECIFICATION[46] = r'in'

# 47: TokenInfo('apply', 2, 46, '{}({})'),
TOKEN_SPECIFICATION[47] = r'\w\((\d)*\)'

# 48: TokenInfo('brackets', 1, 47, '({})'),
TOKEN_SPECIFICATION[48] = r'\((\d)*\)'

# 49: TokenInfo(u'\u0393', 0, 50, '\\Gamma'),
TOKEN_SPECIFICATION[49] = r'Gamma'

# 50: TokenInfo(u'\u0394', 0, 51, '\\Delta'),
TOKEN_SPECIFICATION[50] = r'Delta'

# 51: TokenInfo(u'\u0398', 0, 55, '\\Theta'),
TOKEN_SPECIFICATION[51] = r'Theta'

# 52: TokenInfo(u'\u039B', 0, 58, '\\Lambda'),
TOKEN_SPECIFICATION[52] = r'Lambda'

# 53: TokenInfo(u'\u039E', 0, 61, '\\Xi'),
TOKEN_SPECIFICATION[53] = r'Xi'

# 54: TokenInfo(u'\u03A0', 0, 63, '\\Pi'),
TOKEN_SPECIFICATION[54] = r'Pi'

# 55: TokenInfo(u'\u03A3', 0, 65, '\\Sigma'),
TOKEN_SPECIFICATION[55] = r'Sigma'

# 56: TokenInfo(u'\u03A5', 0, 67, '\\Upsilon'),
TOKEN_SPECIFICATION[56] = r'Upsilon'

# 57: TokenInfo(u'\u03A6', 0, 68, '\\Phi'),
TOKEN_SPECIFICATION[57] = r'Phi'

# 58: TokenInfo(u'\u03A8', 0, 70, '\\Psi'),
TOKEN_SPECIFICATION[58] = r'Psi'

# 59: TokenInfo(u'\u03A9', 0, 71, '\\Omega'),
TOKEN_SPECIFICATION[59] = r'Omega'

# 60: TokenInfo(u'\u03B1', 0, 72, '\\alpha'),
TOKEN_SPECIFICATION[60] = r'alpha'

# 61: TokenInfo(u'\u03B2', 0, 73, '\\beta'),
TOKEN_SPECIFICATION[61] = r'beta'

# 62: TokenInfo(u'\u03B3', 0, 74, '\\gamma'),
TOKEN_SPECIFICATION[62] = r'gamma'

# 63: TokenInfo(u'\u03B4', 0, 75, '\\delta'),
TOKEN_SPECIFICATION[63] = r'delta'

# 64: TokenInfo(u'\u03B5', 0, 76, '\\epsilon'),
TOKEN_SPECIFICATION[64] = r'epsilon'

# 65: TokenInfo(u'\u03B6', 0, 77, '\\zeta'),
TOKEN_SPECIFICATION[65] = r'zeta'

# 66: TokenInfo(u'\u03B7', 0, 78, '\\eta'),
TOKEN_SPECIFICATION[66] = r'eta'

# 67: TokenInfo(u'\u03B8', 0, 79, '\\theta'),
TOKEN_SPECIFICATION[67] = r'theta'

# 68: TokenInfo(u'\u03B9', 0, 80, '\\iota'),
TOKEN_SPECIFICATION[68] = r'iota'

# 69: TokenInfo(u'\u03BA', 0, 81, '\\kappa'),
TOKEN_SPECIFICATION[69] = r'kappa'

# 70: TokenInfo(u'\u03BB', 0, 82, '\\lambda'),
TOKEN_SPECIFICATION[70] = r'lambda'

# 71: TokenInfo(u'\u03BC', 0, 83, '\\mu'),
TOKEN_SPECIFICATION[71] = r'mu'

# 72: TokenInfo(u'\u03BD', 0, 84, '\\nu'),
TOKEN_SPECIFICATION[72] = r'nu'

# 73: TokenInfo(u'\u03BE', 0, 85, '\\xi'),
TOKEN_SPECIFICATION[73] = r'xi'

# 74: TokenInfo(u'\u03C0', 0, 87, '\\pi'),
TOKEN_SPECIFICATION[74] = r'pi'

# 75: TokenInfo(u'\u03C1', 0, 88, '\\rho'),
TOKEN_SPECIFICATION[75] = r'rho'

# 76: TokenInfo(u'\u03C3', 0, 89, '\\sigma'),
TOKEN_SPECIFICATION[76] = r'sigma'

# 77: TokenInfo(u'\u03C4', 0, 90, '\\tau'),
TOKEN_SPECIFICATION[77] = r'tau'

# 78: TokenInfo(u'\u03C5', 0, 91, '\\upsilon'),
TOKEN_SPECIFICATION[78] = r'upsilon'

# 79: TokenInfo(u'\u03C6', 0, 92, '\\phi'),
TOKEN_SPECIFICATION[79] = r'phi'

# 80: TokenInfo(u'\u03C7', 0, 93, '\\chi'),
TOKEN_SPECIFICATION[80] = r'chi'

# 81: TokenInfo(u'\u03C8', 0, 94, '\\psi'),
TOKEN_SPECIFICATION[81] = r'psi'

# 82: TokenInfo(u'\u03C9', 0, 95, '\\omega'),
TOKEN_SPECIFICATION[82] = r'omega'

# 83: TokenInfo('A', 0, 96, 'A'),
TOKEN_SPECIFICATION[83] = r'A'

# 84: TokenInfo('B', 0, 97, 'B'),
TOKEN_SPECIFICATION[84] = r'B'

# 85: TokenInfo('C', 0, 98, 'C'),
TOKEN_SPECIFICATION[85] = r'C'

# 86: TokenInfo('D', 0, 99, 'D'),
TOKEN_SPECIFICATION[86] = r'D'

# 87: TokenInfo('E', 0, 100, 'E'),
TOKEN_SPECIFICATION[87] = r'E'

# 88: TokenInfo('F', 0, 101, 'F'),
TOKEN_SPECIFICATION[88] = r'F'

# 89: TokenInfo('G', 0, 102, 'G'),
TOKEN_SPECIFICATION[89] = r'G'

# 90: TokenInfo('H', 0, 103, 'H'),
TOKEN_SPECIFICATION[90] = r'H'

# 91: TokenInfo('I', 0, 104, 'I'),
TOKEN_SPECIFICATION[91] = r'I'

# 92: TokenInfo('J', 0, 105, 'J'),
TOKEN_SPECIFICATION[92] = r'J'

# 93: TokenInfo('K', 0, 106, 'K'),
TOKEN_SPECIFICATION[93] = r'K'

# 94: TokenInfo('L', 0, 107, 'L'),
TOKEN_SPECIFICATION[94] = r'L'

# 95: TokenInfo('M', 0, 108, 'M'),
TOKEN_SPECIFICATION[95] = r'M'

# 96: TokenInfo('N', 0, 109, 'N'),
TOKEN_SPECIFICATION[96] = r'N'

# 97: TokenInfo('O', 0, 110, 'O'),
TOKEN_SPECIFICATION[97] = r'O'

# 98: TokenInfo('P', 0, 111, 'P'),
TOKEN_SPECIFICATION[98] = r'P'

# 99: TokenInfo('Q', 0, 112, 'Q'),
TOKEN_SPECIFICATION[99] = r'Q'

# 100: TokenInfo('R', 0, 113, 'R'),
TOKEN_SPECIFICATION[100] = r'R'

# 101: TokenInfo('S', 0, 114, 'S'),
TOKEN_SPECIFICATION[101] = r'S'

# 102: TokenInfo('T', 0, 115, 'T'),
TOKEN_SPECIFICATION[102] = r'T'

# 103: TokenInfo('U', 0, 116, 'U'),
TOKEN_SPECIFICATION[103] = r'U'

# 104: TokenInfo('V', 0, 117, 'V'),
TOKEN_SPECIFICATION[104] = r'V'

# 105: TokenInfo('W', 0, 118, 'W'),
TOKEN_SPECIFICATION[105] = r'W'

# 106: TokenInfo('X', 0, 119, 'X'),
TOKEN_SPECIFICATION[106] = r'X'

# 107: TokenInfo('Y', 0, 120, 'Y'),
TOKEN_SPECIFICATION[107] = r'Y'

# 108: TokenInfo('Z', 0, 121, 'Z'),
TOKEN_SPECIFICATION[108] = r'Z'

# 109: TokenInfo('a', 0, 122, 'a'),
TOKEN_SPECIFICATION[109] = r'a'

# 110: TokenInfo('b', 0, 123, 'b'),
TOKEN_SPECIFICATION[110] = r'b'

# 111: TokenInfo('c', 0, 124, 'c'),
TOKEN_SPECIFICATION[111] = r'c'

# 112: TokenInfo('d', 0, 125, 'd'),
TOKEN_SPECIFICATION[112] = r'd'

# 113: TokenInfo('e', 0, 126, 'e'),
TOKEN_SPECIFICATION[113] = r'e'

# 114: TokenInfo('f', 0, 127, 'f'),
TOKEN_SPECIFICATION[114] = r'f'

# 115: TokenInfo('g', 0, 128, 'g'),
TOKEN_SPECIFICATION[115] = r'g'

# 116: TokenInfo('h', 0, 129, 'h'),
TOKEN_SPECIFICATION[116] = r'h'

# 117: TokenInfo('i', 0, 130, 'i'),
TOKEN_SPECIFICATION[117] = r'i'

# 118: TokenInfo('j', 0, 131, 'j'),
TOKEN_SPECIFICATION[118] = r'j'

# 119: TokenInfo('k', 0, 132, 'k'),
TOKEN_SPECIFICATION[119] = r'k'

# 120: TokenInfo('l', 0, 133, 'l'),
TOKEN_SPECIFICATION[120] = r'l'

# 121: TokenInfo('m', 0, 134, 'm'),
TOKEN_SPECIFICATION[121] = r'm'

# 122: TokenInfo('n', 0, 135, 'n'),
TOKEN_SPECIFICATION[122] = r'n'

# 123: TokenInfo('o', 0, 136, 'o'),
TOKEN_SPECIFICATION[123] = r'o'

# 124: TokenInfo('p', 0, 137, 'p'),
TOKEN_SPECIFICATION[124] = r'p'

# 125: TokenInfo('q', 0, 138, 'q'),
TOKEN_SPECIFICATION[125] = r'q'

# 126: TokenInfo('r', 0, 139, 'r'),
TOKEN_SPECIFICATION[126] = r'r'

# 127: TokenInfo('s', 0, 140, 's'),
TOKEN_SPECIFICATION[127] = r's'

# 128: TokenInfo('t', 0, 141, 't'),
TOKEN_SPECIFICATION[128] = r't'

# 129: TokenInfo('u', 0, 142, 'u'),
TOKEN_SPECIFICATION[129] = r'u'

# 130: TokenInfo('v', 0, 143, 'v'),
TOKEN_SPECIFICATION[130] = r'v'

# 131: TokenInfo('w', 0, 144, 'w'),
TOKEN_SPECIFICATION[131] = r'w'

# 132: TokenInfo('x', 0, 145, 'x'),
TOKEN_SPECIFICATION[132] = r'x'

# 133: TokenInfo('y', 0, 146, 'y'),
TOKEN_SPECIFICATION[133] = r'y'

# 134: TokenInfo('z', 0, 147, 'z'),
TOKEN_SPECIFICATION[134] = r'z'

# 135: TokenInfo('1', 0, 148, '1'),
TOKEN_SPECIFICATION[135] = r'1'

# 136: TokenInfo('2', 0, 149, '2'),
TOKEN_SPECIFICATION[136] = r'2'

# 137: TokenInfo('3', 0, 150, '3'),
TOKEN_SPECIFICATION[137] = r'3'

# 138: TokenInfo('4', 0, 151, '4'),
TOKEN_SPECIFICATION[138] = r'4'

# 139: TokenInfo('5', 0, 152, '5'),
TOKEN_SPECIFICATION[139] = r'5'

# 140: TokenInfo('6', 0, 153, '6'),
TOKEN_SPECIFICATION[140] = r'6'

# 141: TokenInfo('7', 0, 154, '7'),
TOKEN_SPECIFICATION[141] = r'7'

# 142: TokenInfo('8', 0, 155, '8'),
TOKEN_SPECIFICATION[142] = r'8'

# 143: TokenInfo('9', 0, 156, '9'),
TOKEN_SPECIFICATION[143] = r'9'

# 144: TokenInfo('0', 0, 157, '0')
TOKEN_SPECIFICATION[144] = r'0'

# 145: TokenInfo('infty', 0, 0, '\\infty'),
TOKEN_SPECIFICATION[145] = r'infty'

# 146: TokenInfo('propto', 0, 0, '\\propto'),
TOKEN_SPECIFICATION[146] = r'propto'

# 147: TokenInfo('negate', 0, 0, '-{}')
TOKEN_SPECIFICATION[147] = r'-\w'

TOKEN_REGEX = r'|'.join('(?P<%s>%s)' % ('g' + str(id), reg) for (id, reg) in TOKEN_SPECIFICATION.items())
OCCURENCES = [0] * len(TOKEN_SPECIFICATION.keys())
assert len(TOKEN_SPECIFICATION) == len(tokens.possibilities())


def find_all(string):
    for match in re.finditer(TOKEN_REGEX, string):  # matches first matching regex, atoms last in TOKEN_REGEX
        group = match.lastgroup
        group_num = int(group[1:])  # delete g, group names starting with a number are not allowed
        OCCURENCES[group_num] += 1


def scan(directory, constrictions):
    iterator = os.scandir(directory)
    total = len(constrictions)
    count = 1
    for entry in iterator:

        if entry.is_file():
            if entry.name.endswith('txt') or entry.name.endswith('tex'):
                with open(directory + '/' + entry.name, 'r') as file:
                    string = file.read()
                    find_all(string)

        if entry.is_dir() and not constrictions:
            scan(directory + '/' + entry.name, constrictions)

        if entry.is_dir() and entry.name in constrictions:
            scan(directory + '/' + entry.name, constrictions)

        print('Directory {} of {} finished..'.format(count, min(total, 2000)))
        count += 1

        if count > 2000:
            break

    save(paths.distribution_bias)


def save(path):
    with open(path, "w") as file:
        string = ''
        for count in OCCURENCES:
            string += str(count) + ','
        string = string[:-1]
        file.write(string)


def load(path):
    if not os.path.exists(path):
        return None

    with open(path, 'r') as file:
        data = file.read()
        ls = data.split(',')

    occurrences = [int(o)/100000 for o in ls]  # arbitrary value to prevent overflow
    # total = sum(occurrences)
    # distribution = [o / total for o in occurrences]

    for i in range(len(occurrences)):
        occurrences[i] += 1

    def softmax(w, t = 1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    distribution = softmax(occurrences)

    return distribution


def read_file_names():
    iterator = os.scandir(paths.arxiv_data)
    filenames = []
    for entry in iterator:
        if entry.is_dir():
            filenames.append(entry.name)
    return filenames


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('Please specify a directory to scan for latex documents.')

    path = sys.argv[1]

    if not os.path.exists(path):
        raise ValueError('Path does not exist.')

    # take only those folders which we actually train on
    constrictions = read_file_names()

    scan(sys.argv[1], constrictions)
