
from string import Template

def render(tree):

	pass


template = 'hello {} {}'

string = template.format('1','{}')

string.format('2')

print(string)


template2 = Template('hallo ${} ${}')

string = template2.safe_substitute('hello')

print(string)


template3 = 'hello {} {}'

string = template3.format(['ha', 'hu'])

print(string)