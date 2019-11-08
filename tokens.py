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
    9   : TokenInfo('tan',1,9,'tan {}'),
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
    33  : TokenInfo('derive', 2, 1, '\\frac{{\\delta{}}}{{\\delta {}}}'),

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
    49  : TokenInfo(u'\u0393',0,50,'\\Gamma'),
    50  : TokenInfo(u'\u0394',0,51,'\\Delta'),
    51  : TokenInfo(u'\u0398',0,55,'\\Theta'),
    52  : TokenInfo(u'\u039B',0,58,'\\Lambda'),
    53  : TokenInfo(u'\u039E',0,61,'\\Xi'),
    54  : TokenInfo(u'\u03A0',0,63,'\\Pi'),
    55  : TokenInfo(u'\u03A3',0,65,'\\Sigma'),
    56  : TokenInfo(u'\u03A5',0,67,'\\Upsilon'),
    57  : TokenInfo(u'\u03A6',0,68,'\\Phi'),
    58  : TokenInfo(u'\u03A8',0,70,'\\Psi'),
    59  : TokenInfo(u'\u03A9',0,71,'\\Omega'),

    60  : TokenInfo(u'\u03B1',0,72,'\\alpha'),
    61  : TokenInfo(u'\u03B2',0,73,'\\beta'),
    62  : TokenInfo(u'\u03B3',0,74,'\\gamma'),
    63  : TokenInfo(u'\u03B4',0,75,'\\delta'),
    64  : TokenInfo(u'\u03B5',0,76,'\\epsilon'),
    65  : TokenInfo(u'\u03B6',0,77,'\\zeta'),
    66  : TokenInfo(u'\u03B7',0,78,'\\eta'),
    67  : TokenInfo(u'\u03B8',0,79,'\\theta'),
    68  : TokenInfo(u'\u03B9',0,80,'\\iota'),
    69  : TokenInfo(u'\u03BA',0,81,'\\kappa'),
    70  : TokenInfo(u'\u03BB',0,82,'\\lambda'),
    71  : TokenInfo(u'\u03BC',0,83,'\\mu'),
    72  : TokenInfo(u'\u03BD',0,84,'\\nu'),
    73  : TokenInfo(u'\u03BE',0,85,'\\xi'),
    74  : TokenInfo(u'\u03C0',0,87,'\\pi'),
    75  : TokenInfo(u'\u03C1',0,88,'\\rho'),
    76  : TokenInfo(u'\u03C3',0,89,'\\sigma'),
    77  : TokenInfo(u'\u03C4',0,90,'\\tau'),
    78  : TokenInfo(u'\u03C5',0,91,'\\upsilon'),
    79  : TokenInfo(u'\u03C6',0,92,'\\phi'),
    80  : TokenInfo(u'\u03C7',0,93,'\\chi'),
    81  : TokenInfo(u'\u03C8',0,94,'\\psi'),
    82  : TokenInfo(u'\u03C9',0,95,'\\omega'),

    83  : TokenInfo('A',0,96,'A'),
    84  : TokenInfo('B',0,97,'B'),
    85  : TokenInfo('C',0,98,'C'),
    86 : TokenInfo('D',0,99,'D'),
    87 : TokenInfo('E',0,100,'E'),
    88 : TokenInfo('F',0,101,'F'),
    89 : TokenInfo('G',0,102,'G'),
    90 : TokenInfo('H',0,103,'H'),
    91 : TokenInfo('I',0,104,'I'),
    92 : TokenInfo('J',0,105,'J'),
    93 : TokenInfo('K',0,106,'K'),
    94 : TokenInfo('L',0,107,'L'),
    95 : TokenInfo('M',0,108,'M'),
    96 : TokenInfo('N',0,109,'N'),
    97 : TokenInfo('O',0,110,'O'),
    98 : TokenInfo('P',0,111,'P'),
    99 : TokenInfo('Q',0,112,'Q'),
    100 : TokenInfo('R',0,113,'R'),
    101 : TokenInfo('S',0,114,'S'),
    102 : TokenInfo('T',0,115,'T'),
    103 : TokenInfo('U',0,116,'U'),
    104 : TokenInfo('V',0,117,'V'),
    105 : TokenInfo('W',0,118,'W'),
    106 : TokenInfo('X',0,119,'X'),
    107 : TokenInfo('Y',0,120,'Y'),
    108 : TokenInfo('Z',0,121,'Z'),
    109 : TokenInfo('a',0,122,'a'),
    110 : TokenInfo('b',0,123,'b'),
    111 : TokenInfo('c',0,124,'c'),
    112 : TokenInfo('d',0,125,'d'),
    113 : TokenInfo('e',0,126,'e'),
    114 : TokenInfo('f',0,127,'f'),
    115 : TokenInfo('g',0,128,'g'),
    116 : TokenInfo('h',0,129,'h'),
    117 : TokenInfo('i',0,130,'i'),
    118 : TokenInfo('j',0,131,'j'),
    119 : TokenInfo('k',0,132,'k'),
    120 : TokenInfo('l',0,133,'l'),
    121 : TokenInfo('m',0,134,'m'),
    122 : TokenInfo('n',0,135,'n'),
    123 : TokenInfo('o',0,136,'o'),
    124 : TokenInfo('p',0,137,'p'),
    125 : TokenInfo('q',0,138,'q'),
    126 : TokenInfo('r',0,139,'r'),
    127 : TokenInfo('s',0,140,'s'),
    128 : TokenInfo('t',0,141,'t'),
    129 : TokenInfo('u',0,142,'u'),
    130 : TokenInfo('v',0,143,'v'),
    131 : TokenInfo('w',0,144,'w'),
    132 : TokenInfo('x',0,145,'x'),
    133 : TokenInfo('y',0,146,'y'),
    134 : TokenInfo('z',0,147,'z'),

    135 : TokenInfo('1',0,148,'1'),
    136 : TokenInfo('2',0,149,'2'),
    137 : TokenInfo('3',0,150,'3'),
    138 : TokenInfo('4',0,151,'4'),
    139 : TokenInfo('5',0,152,'5'),
    140 : TokenInfo('6',0,153,'6'),
    141 : TokenInfo('7',0,154,'7'),
    142 : TokenInfo('8',0,155,'8'),
    143 : TokenInfo('9',0,156,'9'),
    144 : TokenInfo('0',0,157,'0'),

    # added later
    145 : TokenInfo('infty', 0, 0, '\\infty'),
    146 : TokenInfo('propto', 0, 0, '\\propto'),
    147 : TokenInfo('negate', 1, 0, '-{}')
}


def get(id): return _tokens[id]


def count(): return len(_tokens)


def possibilities(): return list(_tokens.keys())


def empty(): return [0] * len(_tokens)


def id(onehot):

    for i in range(len(onehot)):
        if onehot[i] == 1:
            return i

    raise ValueError('Got encoding of empty start token, but start token has no ID.')


def onehot(id):

    template = [0] * len(_tokens)
    template[id] = 1

    return template


assert len(_tokens) == 148
assert not [i for i in _tokens.keys() if i < 0 or i > 148]
assert not [(i,(k,t)) for (i,(k,t)) in enumerate(_tokens.items()) if not i == k]


for i,t in _tokens.items():
    t.onehot = onehot(i)
