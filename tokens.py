from enum import Enum 
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
	code: int
	latex: str

class Token(Enum):

	# unary
	Root 		= TokenInfo('root',1,0,'\\sqrt{{{}}}')
	Fac 		= TokenInfo('fac',1,1,'{}!')
	Max 		= TokenInfo('max',1,2,'max {}')
	Min 		= TokenInfo('min',1,3,'min {}')
	ArgMax 		= TokenInfo('argmax',1,4,'argmax {}')
	ArgMin 		= TokenInfo('argmin',1,5,'argmin {}')
	Inverse 	= TokenInfo('inverse',1,6,'{}^{{-1}}')
	Sin 		= TokenInfo('sin',1,7,'sin {}')
	Cos 		= TokenInfo('cos',1,8,'cos {}')
	Tan 		= TokenInfo('tan',1,9,'tan {}')
	SinH 		= TokenInfo('sinh',1,10,'sinh {}')
	CosH 		= TokenInfo('cosh',1,11,'cosh {}')
	TanH 		= TokenInfo('tanh',1,12,'tanh {}')
	Sigmoid 	= TokenInfo('sigmoid',1,13,'\\sigma({})')
	Transpose 	= TokenInfo('transpose',1,14,'{}^T')
	Prime 		= TokenInfo('prime',1,15,'{}\'')
	Absolute 	= TokenInfo('absolute',1,16,'|{}|')
	Norm 		= TokenInfo('norm',1,17,'||{}||')
	MathbbE 	= TokenInfo('mathbbe',1,18,'\\mathbb{{E}}[{}]')
	MathbbP 	= TokenInfo('mathbbp',1,19,'\\mathbb{{P}}[{}]')

	# subscripted
	MaxSub 		= TokenInfo('maxsub',2,20,'max_{{{}}} {}')
	MinSub 		= TokenInfo('minsub',2,21,'min_{{{}}} {}')
	ArgMaxSub 	= TokenInfo('argmaxsub',2,22,'argmax_{{{}}} {}')
	ArgMinSub 	= TokenInfo('argminsub',2,23,'argmin_{{{}}} {}')
	MathbbESub 	= TokenInfo('mathbbesub',2,24,'\\mathbb{{E}}_{{{}}}[{}]')
	MathbbPSub 	= TokenInfo('mathbbpsub',2,25,'\\mathbb{{P}}_{{{}}}[{}]')

	# binary
	Add 		= TokenInfo('add',2,26,'{} + {}')
	Sub 		= TokenInfo('sub',2,27,'{} - {}')
	Dot 		= TokenInfo('dot',2,28,'{} \\cdot {}')
	Cross 		= TokenInfo('cross',2,29,'{} \\times {}')
	Fract 		= TokenInfo('fract',2,30,'\\frac{{{}}}{{{}}}')
	Mod 		= TokenInfo('mod',2,31,'{} mod {}')
	Power 		= TokenInfo('power',2,32,'{}^{{{}}}')

	# sequences
	Sum 		= TokenInfo('sum',3,33,'\\sum\\nolimits_{{{}}}^{{{}}} {}')
	Product 	= TokenInfo('product',3,34,'\\prod\\nolimits_{{{}}}^{{{}}} {}')
	Integral 	= TokenInfo('integral',3,35,'\\int\\nolimits_{{{}}}^{{{}}} {}')

	# equalities
	Equals 		= TokenInfo('equals',2,36,'{} = {}')
	Lesser 		= TokenInfo('lesser',2,37,'{} < {}')
	Greater 	= TokenInfo('greater',2,38,'{} > {}')
	LesserEq 	= TokenInfo('lessereq',2,39,'{} \\leq {}')
	GreaterEq 	= TokenInfo('greatereq',2,40,'{} \\geq {}')

	# sets
	Subset 		= TokenInfo('subset',2,41,'{} \\subset {}')
	SubsetEq 	= TokenInfo('subseteq',2,42,'{} \\subseteq {}')
	Union 		= TokenInfo('union',2,43,'{} \\cup {}')
	Difference 	= TokenInfo('difference',2,44,'{} \\cap {}')
	ElemOf 		= TokenInfo('elementof',2,45,'{} \\in {}')

	# special
	Apply 		= TokenInfo('apply',2,46,'{}({})')
	Brackets 	= TokenInfo('brackets',1,47,'({})')

	# atomic
	Alpha 		= TokenInfo(u'\u0391',0,48,'\\Alpha')
	Beta 		= TokenInfo(u'\u0392',0,49,'\\Beta')
	Gamma 		= TokenInfo(u'\u0393',0,50,'\\Gamma')
	Delta 		= TokenInfo(u'\u0394',0,51,'\\Delta')
	Epsilon 	= TokenInfo(u'\u0395',0,52,'\\Epsilon')
	Zeta 		= TokenInfo(u'\u0396',0,53,'\\Zeta')
	Eta 		= TokenInfo(u'\u0397',0,54,'\\Eta')
	Theta 		= TokenInfo(u'\u0398',0,55,'\\Theta')
	Iota 		= TokenInfo(u'\u0399',0,56,'\\Iota')
	Kappa 		= TokenInfo(u'\u039A',0,57,'\\Kappa')
	Lambda 		= TokenInfo(u'\u039B',0,58,'\\Lambda')
	Mu 			= TokenInfo(u'\u039C',0,59,'\\Mu')
	Nu 			= TokenInfo(u'\u039D',0,60,'\\Nu')
	Xi 			= TokenInfo(u'\u039E',0,61,'\\Xi')
	Omicron 	= TokenInfo(u'\u039F',0,62,'\\Omicron')
	Pi 			= TokenInfo(u'\u03A0',0,63,'\\Pi')
	Rho 		= TokenInfo(u'\u03A1',0,64,'\\Rho')
	Sigma 		= TokenInfo(u'\u03A3',0,65,'\\Sigma')
	Tau 		= TokenInfo(u'\u03A4',0,66,'\\Tau')
	Upsilon 	= TokenInfo(u'\u03A5',0,67,'\\Upsilon')
	Phi 		= TokenInfo(u'\u03A6',0,68,'\\Phi')
	Chi 		= TokenInfo(u'\u03A7',0,69,'\\Chi')
	Psi 		= TokenInfo(u'\u03A8',0,70,'\\Psi')
	Omega 		= TokenInfo(u'\u03A9',0,71,'\\Omega')

	alpha 		= TokenInfo(u'\u03B1',0,72,'\\alpha')
	beta 		= TokenInfo(u'\u03B2',0,73,'\\beta')
	gamma 		= TokenInfo(u'\u03B3',0,74,'\\gamma')
	delta 		= TokenInfo(u'\u03B4',0,75,'\\delta')
	epsilon 	= TokenInfo(u'\u03B5',0,76,'\\epsilon')
	zeta 		= TokenInfo(u'\u03B6',0,77,'\\zeta')
	eta 		= TokenInfo(u'\u03B7',0,78,'\\eta')
	theta 		= TokenInfo(u'\u03B8',0,79,'\\theta')
	iota 		= TokenInfo(u'\u03B9',0,80,'\\iota')
	kappa 		= TokenInfo(u'\u03BA',0,81,'\\kappa')
	lamda 		= TokenInfo(u'\u03BB',0,82,'\\lambda')
	mu 			= TokenInfo(u'\u03BC',0,83,'\\mu')
	nu 			= TokenInfo(u'\u03BD',0,84,'\\nu')
	xi 			= TokenInfo(u'\u03BE',0,85,'\\xi')
	omicron 	= TokenInfo(u'\u03BF',0,86,'\\omicron')
	pi 			= TokenInfo(u'\u03C0',0,87,'\\pi')
	rho 		= TokenInfo(u'\u03C1',0,88,'\\rho')
	sigma 		= TokenInfo(u'\u03C3',0,89,'\\sigma')
	tau 		= TokenInfo(u'\u03C4',0,90,'\\tau')
	upsilon 	= TokenInfo(u'\u03C5',0,91,'\\upsilon')
	phi 		= TokenInfo(u'\u03C6',0,92,'\\phi')
	chi 		= TokenInfo(u'\u03C7',0,93,'\\chi')
	psi 		= TokenInfo(u'\u03C8',0,94,'\\psi')
	omega 		= TokenInfo(u'\u03C9',0,95,'\\omega')

	A 			= TokenInfo('A',0,96,'A')
	B 			= TokenInfo('B',0,97,'B')
	C 			= TokenInfo('C',0,98,'C')
	D 			= TokenInfo('D',0,99,'D')
	E 			= TokenInfo('E',0,100,'E')
	F 			= TokenInfo('F',0,101,'F')
	G 			= TokenInfo('G',0,102,'G')
	H 			= TokenInfo('H',0,103,'H')
	I 			= TokenInfo('I',0,104,'I')
	J 			= TokenInfo('J',0,105,'J')
	K 			= TokenInfo('K',0,106,'K')
	L 			= TokenInfo('L',0,107,'L')
	M 			= TokenInfo('M',0,108,'M')
	N 			= TokenInfo('N',0,109,'N')
	O 			= TokenInfo('O',0,110,'O')
	P 			= TokenInfo('P',0,111,'P')
	Q 			= TokenInfo('Q',0,112,'Q')
	R 			= TokenInfo('R',0,113,'R')
	S 			= TokenInfo('S',0,114,'S')
	T 			= TokenInfo('T',0,115,'T')
	U 			= TokenInfo('U',0,116,'U')
	V 			= TokenInfo('V',0,117,'V')
	W 			= TokenInfo('W',0,118,'W')
	X 			= TokenInfo('X',0,119,'X')
	Y 			= TokenInfo('Y',0,120,'Y')
	Z 			= TokenInfo('Z',0,121,'Z')
	a 			= TokenInfo('a',0,122,'a')
	b 			= TokenInfo('b',0,123,'b')
	c 			= TokenInfo('c',0,124,'c')
	d 			= TokenInfo('d',0,125,'d')
	e 			= TokenInfo('e',0,126,'e')
	f 			= TokenInfo('f',0,127,'f')
	g 			= TokenInfo('g',0,128,'g')
	h 			= TokenInfo('h',0,129,'h')
	i 			= TokenInfo('i',0,130,'i')
	j 			= TokenInfo('j',0,131,'j')
	k 			= TokenInfo('k',0,132,'k')
	l 			= TokenInfo('l',0,133,'l')
	m 			= TokenInfo('m',0,134,'m')
	n 			= TokenInfo('n',0,135,'n')
	o 			= TokenInfo('o',0,136,'o')
	p 			= TokenInfo('p',0,137,'p')
	q 			= TokenInfo('q',0,138,'q')
	r 			= TokenInfo('r',0,139,'r')
	s 			= TokenInfo('s',0,140,'s')
	t 			= TokenInfo('t',0,141,'t')
	u 			= TokenInfo('u',0,142,'u')
	v 			= TokenInfo('v',0,143,'v')
	w 			= TokenInfo('w',0,144,'w')
	x 			= TokenInfo('x',0,145,'x')
	y 			= TokenInfo('y',0,146,'y')
	z 			= TokenInfo('z',0,147,'z')

	One 		= TokenInfo('1',0,148,'1')
	Two 		= TokenInfo('2',0,149,'2')
	Three 		= TokenInfo('3',0,150,'3')
	Four 		= TokenInfo('4',0,151,'4')
	Five		= TokenInfo('5',0,152,'5')
	Six 		= TokenInfo('6',0,153,'6')
	Seven 		= TokenInfo('7',0,154,'7')
	Eight 		= TokenInfo('8',0,155,'8')
	Nine 		= TokenInfo('9',0,156,'9')
	Zero 		= TokenInfo('0',0,157,'0')


position = 0
for item in Token:
	assert(item.value.code == position)
	position += 1


def token(encoding):
	for item in Token:
		if item.value.code == encoding:
			return item 

	raise SystemError()
