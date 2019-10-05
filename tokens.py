from enum import Enum 
from math import ceil
from math import log
from dataclasses import dataclass

# its beneficial to have all possible tokens or actions respectively defined
# in a single place with uniform data types and all necessary information
# associated with them. even if storing those for atomic types such as
# simple numbers and letters seems verbose and unnecessary, it might
# avoid hacky solutions later on when choices can have different types or
# token information is defined redundantly in different places and has no
# garantuee of being in a legal cohesive state.

# hard code one hot encoding to ensure consistency independant from order of iteration

# assumption: format string for latex translation requires the token arguments in
# sequential order with sequences taking the subscripted argument first

@dataclass
class TokenInfo:

    name: str
    arity: int
    onehot: int
    latex: str

_tokens = {

    # unary
    0   : TokenInfo('root',1,0,'\\sqrt{{{}}}'),
    1   : TokenInfo('fac',1,1,'{}!'),
    2   : TokenInfo('max',1,2,'max {}'),
    3   : TokenInfo('min',1,3,'min {}'),
    4   : TokenInfo('argmax',1,4,'argmax {}'),
    5   : TokenInfo('argmin',1,5,'argmin {}'),
    6   : TokenInfo('inverse',1,6,'{}^{{-1}}'),
    7   : TokenInfo('sin',1,7,'sin {}'),
    8   : TokenInfo('cos',1,8,'cos {}'),
    9  : TokenInfo('tan',1,9,'tan {}'),
    10  : TokenInfo('sinh',1,10,'sinh {}'),
    11  : TokenInfo('cosh',1,11,'cosh {}'),
    12  : TokenInfo('tanh',1,12,'tanh {}'),
    13  : TokenInfo('sigmoid',1,13,'\\sigma({})'),
    14  : TokenInfo('transpose',1,14,'{}^T'),
    15  : TokenInfo('prime',1,15,'{}\''),
    16  : TokenInfo('absolute',1,16,'|{}|'),
    17  : TokenInfo('norm',1,17,'||{}||'),
    18  : TokenInfo('mathbbe',1,18,'\\mathbb{{E}}[{}]'),
    19  : TokenInfo('mathbbp',1,19,'\\mathbb{{P}}[{}]'),

    # subscripted
    20  : TokenInfo('maxsub',2,20,'max_{{{}}} {}'),
    21  : TokenInfo('minsub',2,21,'min_{{{}}} {}'),
    22  : TokenInfo('argmaxsub',2,22,'argmax_{{{}}} {}'),
    23  : TokenInfo('argminsub',2,23,'argmin_{{{}}} {}'),
    24  : TokenInfo('mathbbesub',2,24,'\\mathbb{{E}}_{{{}}}[{}]'),
    25  : TokenInfo('mathbbpsub',2,25,'\\mathbb{{P}}_{{{}}}[{}]'),

    # binary
    26  : TokenInfo('add',2,26,'{} + {}'),
    27  : TokenInfo('sub',2,27,'{} - {}'),
    28  : TokenInfo('dot',2,28,'{} \\cdot {}'),
    29  : TokenInfo('cross',2,29,'{} \\times {}'),
    30  : TokenInfo('fract',2,30,'\\frac{{{}}}{{{}}}'),
    31  : TokenInfo('mod',2,31,'{} mod {}'),
    32  : TokenInfo('power',2,32,'{}^{{{}}}'),
    33  : TokenInfo('derive', 2, None, '\\frac{{\\delta{}}}{{\\delta {}}}'),

    # sequences
    34  : TokenInfo('sum',3,33,'\\sum\\nolimits_{{{}}}^{{{}}} {}'),
    35  : TokenInfo('product',3,34,'\\prod\\nolimits_{{{}}}^{{{}}} {}'),
    36  : TokenInfo('integral',3,35,'\\int\\nolimits_{{{}}}^{{{}}} {}'),

    # equalities
    37  : TokenInfo('equals',2,36,'{} = {}'),
    38  : TokenInfo('lesser',2,37,'{} < {}'),
    39  : TokenInfo('greater',2,38,'{} > {}'),
    40  : TokenInfo('lessereq',2,39,'{} \\leq {}'),
    41  : TokenInfo('greatereq',2,40,'{} \\geq {}'),

    # sets
    42  : TokenInfo('subset',2,41,'{} \\subset {}'),
    43  : TokenInfo('subseteq',2,42,'{} \\subseteq {}'),
    44  : TokenInfo('union',2,43,'{} \\cup {}'),
    45  : TokenInfo('difference',2,44,'{} \\cap {}'),
    46  : TokenInfo('elementof',2,45,'{} \\in {}'),

    # special
    47  : TokenInfo('apply',2,46,'{}({})'),
    48  : TokenInfo('brackets',1,47,'({})'),

    # atomic
    49  : TokenInfo(u'\u0391',0,48,'\\Alpha'),
    50  : TokenInfo(u'\u0392',0,49,'\\Beta'),
    51  : TokenInfo(u'\u0393',0,50,'\\Gamma'),
    52  : TokenInfo(u'\u0394',0,51,'\\Delta'),
    53  : TokenInfo(u'\u0395',0,52,'\\Epsilon'),
    54  : TokenInfo(u'\u0396',0,53,'\\Zeta'),
    55  : TokenInfo(u'\u0397',0,54,'\\Eta'),
    56  : TokenInfo(u'\u0398',0,55,'\\Theta'),
    57  : TokenInfo(u'\u0399',0,56,'\\Iota'),
    58  : TokenInfo(u'\u039A',0,57,'\\Kappa'),
    59  : TokenInfo(u'\u039B',0,58,'\\Lambda'),
    60  : TokenInfo(u'\u039C',0,59,'\\Mu'),
    61  : TokenInfo(u'\u039D',0,60,'\\Nu'),
    62  : TokenInfo(u'\u039E',0,61,'\\Xi'),
    63  : TokenInfo(u'\u039F',0,62,'\\Omicron'),
    64  : TokenInfo(u'\u03A0',0,63,'\\Pi'),
    65  : TokenInfo(u'\u03A1',0,64,'\\Rho'),
    66  : TokenInfo(u'\u03A3',0,65,'\\Sigma'),
    67  : TokenInfo(u'\u03A4',0,66,'\\Tau'),
    68  : TokenInfo(u'\u03A5',0,67,'\\Upsilon'),
    69  : TokenInfo(u'\u03A6',0,68,'\\Phi'),
    70  : TokenInfo(u'\u03A7',0,69,'\\Chi'),
    71  : TokenInfo(u'\u03A8',0,70,'\\Psi'),
    72  : TokenInfo(u'\u03A9',0,71,'\\Omega'),

    73  : TokenInfo(u'\u03B1',0,72,'\\alpha'),
    74  : TokenInfo(u'\u03B2',0,73,'\\beta'),
    75  : TokenInfo(u'\u03B3',0,74,'\\gamma'),
    76  : TokenInfo(u'\u03B4',0,75,'\\delta'),
    77  : TokenInfo(u'\u03B5',0,76,'\\epsilon'),
    78  : TokenInfo(u'\u03B6',0,77,'\\zeta'),
    79  : TokenInfo(u'\u03B7',0,78,'\\eta'),
    80  : TokenInfo(u'\u03B8',0,79,'\\theta'),
    81  : TokenInfo(u'\u03B9',0,80,'\\iota'),
    82  : TokenInfo(u'\u03BA',0,81,'\\kappa'),
    83  : TokenInfo(u'\u03BB',0,82,'\\lambda'),
    84  : TokenInfo(u'\u03BC',0,83,'\\mu'),
    85  : TokenInfo(u'\u03BD',0,84,'\\nu'),
    86  : TokenInfo(u'\u03BE',0,85,'\\xi'),
    87  : TokenInfo(u'\u03BF',0,86,'\\omicron'),
    88  : TokenInfo(u'\u03C0',0,87,'\\pi'),
    89  : TokenInfo(u'\u03C1',0,88,'\\rho'),
    90  : TokenInfo(u'\u03C3',0,89,'\\sigma'),
    91  : TokenInfo(u'\u03C4',0,90,'\\tau'),
    92  : TokenInfo(u'\u03C5',0,91,'\\upsilon'),
    93  : TokenInfo(u'\u03C6',0,92,'\\phi'),
    94  : TokenInfo(u'\u03C7',0,93,'\\chi'),
    95  : TokenInfo(u'\u03C8',0,94,'\\psi'),
    96  : TokenInfo(u'\u03C9',0,95,'\\omega'),

    97  : TokenInfo('A',0,96,'A'),
    98  : TokenInfo('B',0,97,'B'),
    99  : TokenInfo('C',0,98,'C'),
    100 : TokenInfo('D',0,99,'D'),
    101 : TokenInfo('E',0,100,'E'),
    102 : TokenInfo('F',0,101,'F'),
    103 : TokenInfo('G',0,102,'G'),
    104 : TokenInfo('H',0,103,'H'),
    105 : TokenInfo('I',0,104,'I'),
    106 : TokenInfo('J',0,105,'J'),
    107 : TokenInfo('K',0,106,'K'),
    108 : TokenInfo('L',0,107,'L'),
    109 : TokenInfo('M',0,108,'M'),
    110 : TokenInfo('N',0,109,'N'),
    111 : TokenInfo('O',0,110,'O'),
    112 : TokenInfo('P',0,111,'P'),
    113 : TokenInfo('Q',0,112,'Q'),
    114 : TokenInfo('R',0,113,'R'),
    115 : TokenInfo('S',0,114,'S'),
    116 : TokenInfo('T',0,115,'T'),
    117 : TokenInfo('U',0,116,'U'),
    118 : TokenInfo('V',0,117,'V'),
    119 : TokenInfo('W',0,118,'W'),
    120 : TokenInfo('X',0,119,'X'),
    121 : TokenInfo('Y',0,120,'Y'),
    122 : TokenInfo('Z',0,121,'Z'),
    123 : TokenInfo('a',0,122,'a'),
    124 : TokenInfo('b',0,123,'b'),
    125 : TokenInfo('c',0,124,'c'),
    126 : TokenInfo('d',0,125,'d'),
    127 : TokenInfo('e',0,126,'e'),
    128 : TokenInfo('f',0,127,'f'),
    129 : TokenInfo('g',0,128,'g'),
    130 : TokenInfo('h',0,129,'h'),
    131 : TokenInfo('i',0,130,'i'),
    132 : TokenInfo('j',0,131,'j'),
    133 : TokenInfo('k',0,132,'k'),
    134 : TokenInfo('l',0,133,'l'),
    135 : TokenInfo('m',0,134,'m'),
    136 : TokenInfo('n',0,135,'n'),
    137 : TokenInfo('o',0,136,'o'),
    138 : TokenInfo('p',0,137,'p'),
    139 : TokenInfo('q',0,138,'q'),
    140 : TokenInfo('r',0,139,'r'),
    141 : TokenInfo('s',0,140,'s'),
    142 : TokenInfo('t',0,141,'t'),
    143 : TokenInfo('u',0,142,'u'),
    144 : TokenInfo('v',0,143,'v'),
    145 : TokenInfo('w',0,144,'w'),
    146 : TokenInfo('x',0,145,'x'),
    147 : TokenInfo('y',0,146,'y'),
    148 : TokenInfo('z',0,147,'z'),

    149 : TokenInfo('1',0,148,'1'),
    150 : TokenInfo('2',0,149,'2'),
    151 : TokenInfo('3',0,150,'3'),
    152 : TokenInfo('4',0,151,'4'),
    153 : TokenInfo('5',0,152,'5'),
    154 : TokenInfo('6',0,153,'6'),
    155 : TokenInfo('7',0,154,'7'),
    156 : TokenInfo('8',0,155,'8'),
    157 : TokenInfo('9',0,156,'9'),
    158 : TokenInfo('0',0,157,'0')

}


def get(id): return _tokens[id]


def count(): return len(_tokens)


def possibilities(): return list(_tokens.keys())


def empty(): return [0] * len(_tokens)


def id(onehot):

    for i in range(len(onehot)):
        if onehot[i] == 1:
            return i

    return 0


def onehot(id):

    template = [0] * len(_tokens)
    template[id] = 1

    return template


assert len(_tokens) == 159
assert not [i for i in _tokens.keys() if i < 0 or i > 158]


for i,t in _tokens.items():
    t.onehot = onehot(i)
