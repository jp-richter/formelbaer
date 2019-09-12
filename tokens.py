from enum import Enum 
from collections import namedtuple

# its beneficial to have all possible tokens or actions respectively defined
# in a single place with uniform data types and all necessary information 
# associated with them. even if storing those for atomic types such as 
# simple numbers and letters seems verbose and unnecessary, it might 
# avoid hacky solutions later on when choices can have different types or
# token information is defined redundantly in different places and has no
# garantuee of being in a legal cohesive state. 

TokenInfo = namedtuple('TokenInfo', 'name arity code')


class Token(Enum):

	# unary
	Root = TokenInfo('root',1,0)
	Fac = TokenInfo('fac',1,0)
	Max = TokenInfo('max',1,0)
	Min = TokenInfo('min',1,0)
	ArgMax = TokenInfo('argmax',1,0)
	ArgMin = TokenInfo('argmin',1,0)
	Inverse = TokenInfo('inverse',1,0)
	Sin = TokenInfo('sin',1,0)
	Cos = TokenInfo('cos',1,0)
	Tan = TokenInfo('tan',1,0)
	SinH = TokenInfo('sinh',1,0)
	CosH = TokenInfo('cosh',1,0)
	TanH = TokenInfo('tanh',1,0)
	Sigmoid = TokenInfo('sigmoid',1,0)
	Transpose = TokenInfo('transpose',1,0)
	Prime = TokenInfo('prime',1,0)
	Absolute = TokenInfo('absolute',1,0)
	Norm = TokenInfo('norm',1,0)
	MathbbE = TokenInfo('mathbbe',1,0)
	MathbbP = TokenInfo('mathbbp',1,0)

	# subscripted
	MaxSub = TokenInfo('maxsub',2,0)
	MinSub = TokenInfo('minsub',2,0)
	ArgMaxSub = TokenInfo('argmaxsub',2,0)
	ArgMinSub = TokenInfo('argminsub',2,0)
	MathbbESub = TokenInfo('mathbbesub',2,0)
	MathbbPSub = TokenInfo('mathbbpsub',2,0)

	# binary
	Add = TokenInfo('add',2,0)
	Sub = TokenInfo('sub',2,0)
	Dot = TokenInfo('dot',2,0)
	Cross = TokenInfo('cross',2,0)
	Fract = TokenInfo('fract',2,0)
	Mod = TokenInfo('mod',2,0)
	Power = TokenInfo('power',2,0)

	# sequences
	Sum = TokenInfo('sum',3,0)
	Product = TokenInfo('product',3,0)
	Integral = TokenInfo('integral',3,0)

	# equalities
	Equals = TokenInfo('equals',2,0)
	Lesser = TokenInfo('lesser',2,0)
	Greater = TokenInfo('greater',2,0)
	LesserEq = TokenInfo('lessereq',2,0)
	GreaterEq = TokenInfo('greatereq',2,0)

	# sets
	Subset = TokenInfo('subset',2,0)
	SubsetEq = TokenInfo('subseteq',2,0)
	Union = TokenInfo('union',2,0)
	Difference = TokenInfo('difference',2,0)
	ElemOf = TokenInfo('elementof',2,0)

	# special
	Apply = TokenInfo('apply',2,0)
	Brackets = TokenInfo('brackets',1,0)

	# atomic
	Alpha = TokenInfo(u'\u0391',1,0)
	Beta = TokenInfo(u'\u0392',1,0)
	Gamma = TokenInfo(u'\u0393',1,0)
	Delta = TokenInfo(u'\u0394',1,0)
	Epsilon = TokenInfo(u'\u0395',1,0)
	Zeta = TokenInfo(u'\u0396',1,0)
	Eta = TokenInfo(u'\u0397',1,0)
	Theta = TokenInfo(u'\u0398',1,0)
	Iota = TokenInfo(u'\u0399',1,0)
	Kappa = TokenInfo(u'\u039A',1,0)
	Lambda = TokenInfo(u'\u039B',1,0)
	Mu = TokenInfo(u'\u039C',1,0)
	Nu = TokenInfo(u'\u039D',1,0)
	Xi = TokenInfo(u'\u039E',1,0)
	Omicron = TokenInfo(u'\u039F',1,0)
	Pi = TokenInfo(u'\u039A0',1,0)
	Rho = TokenInfo(u'\u03A1',1,0)
	Sigma = TokenInfo(u'\u03A3',1,0)
	Tau = TokenInfo(u'\u03A4',1,0)
	Upsilon = TokenInfo(u'\u03A5',1,0)
	Phi = TokenInfo(u'\u03A6',1,0)
	Chi = TokenInfo(u'\u03A7',1,0)
	Psi = TokenInfo(u'\u03A8',1,0)
	Omega = TokenInfo(u'\u03A9',1,0)

	alpha = TokenInfo(u'\u03B1',1,0)
	beta = TokenInfo(u'\u03B2',1,0)
	gamma = TokenInfo(u'\u03B3',1,0)
	delta = TokenInfo(u'\u03B4',1,0)
	epsilon = TokenInfo(u'\u03B5',1,0)
	zeta = TokenInfo(u'\u03B6',1,0)
	eta = TokenInfo(u'\u03B7',1,0)
	theta = TokenInfo(u'\u03B8',1,0)
	iota = TokenInfo(u'\u03B9',1,0)
	kappa = TokenInfo(u'\u03BA',1,0)
	lamda = TokenInfo(u'\u03BB',1,0)
	mu = TokenInfo(u'\u03BC',1,0)
	nu = TokenInfo(u'\u03BD',1,0)
	xi = TokenInfo(u'\u03BE',1,0)
	omicron = TokenInfo(u'\u03BF',1,0)
	pi = TokenInfo(u'\u03C0',1,0)
	rho = TokenInfo(u'\u03C1',1,0)
	sigma = TokenInfo(u'\u03C3',1,0)
	tau = TokenInfo(u'\u03C4',1,0)
	upsilon = TokenInfo(u'\u03C5',1,0)
	phi = TokenInfo(u'\u03C6',1,0)
	chi = TokenInfo(u'\u03C7',1,0)
	psi = TokenInfo(u'\u03C8',1,0)
	omega = TokenInfo(u'\u03C9',1,0)

	A = TokenInfo('A',1,0)
	B = TokenInfo('B',1,0)
	C = TokenInfo('C',1,0)
	D = TokenInfo('D',1,0)
	E = TokenInfo('E',1,0)
	F = TokenInfo('F',1,0)
	G = TokenInfo('G',1,0)
	H = TokenInfo('H',1,0)
	I = TokenInfo('I',1,0)
	J = TokenInfo('J',1,0)
	K = TokenInfo('K',1,0)
	L = TokenInfo('L',1,0)
	M = TokenInfo('M',1,0)
	N = TokenInfo('N',1,0)
	O = TokenInfo('O',1,0)
	P = TokenInfo('P',1,0)
	Q = TokenInfo('Q',1,0)
	R = TokenInfo('R',1,0)
	S = TokenInfo('S',1,0)
	T = TokenInfo('T',1,0)
	U = TokenInfo('U',1,0)
	V = TokenInfo('V',1,0)
	W = TokenInfo('W',1,0)
	X = TokenInfo('X',1,0)
	Y = TokenInfo('Y',1,0)
	Z = TokenInfo('Z',1,0)

	a = TokenInfo('a',1,0)
	b = TokenInfo('b',1,0)
	c = TokenInfo('c',1,0)
	d = TokenInfo('d',1,0)
	e = TokenInfo('e',1,0)
	f = TokenInfo('f',1,0)
	g = TokenInfo('g',1,0)
	h = TokenInfo('h',1,0)
	i = TokenInfo('i',1,0)
	j = TokenInfo('j',1,0)
	k = TokenInfo('k',1,0)
	l = TokenInfo('l',1,0)
	m = TokenInfo('m',1,0)
	n = TokenInfo('n',1,0)
	o = TokenInfo('o',1,0)
	p = TokenInfo('p',1,0)
	q = TokenInfo('q',1,0)
	r = TokenInfo('r',1,0)
	s = TokenInfo('s',1,0)
	t = TokenInfo('t',1,0)
	u = TokenInfo('u',1,0)
	v = TokenInfo('v',1,0)
	w = TokenInfo('w',1,0)
	x = TokenInfo('x',1,0)
	y = TokenInfo('y',1,0)
	z = TokenInfo('z',1,0)

	One = TokenInfo('1',1,0)
	Two = TokenInfo('2',1,0)
	Three = TokenInfo('3',1,0)
	Four = TokenInfo('4',1,0)
	Five = TokenInfo('5',1,0)
	Six = TokenInfo('6',1,0)
	Seven = TokenInfo('7',1,0)
	Eight = TokenInfo('8',1,0)
	Nine = TokenInfo('9',1,0)
	Zero = TokenInfo('0',1,0)


template = [0] * len(Token)
position = 0

for token in Token:
	encoding = list(template) # creates copy

	encoding[position] = 1
	token.value = TokenInfo(token.value.name, token.value.arity, encoding) # immutable 
	
	position += 1

for token in Token:
	print(token.value.code)


