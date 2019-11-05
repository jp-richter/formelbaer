
DOCUMENT_START = '''%&preamble

\\begin{{document}}
'''

FORMULAR = '''
\\begin{{page}}
\\begin{{minipage}}[c][1cm]{{50cm}}
\\centering{{
    $ {} $
}}
\\end{{minipage}}
\\end{{page}}
'''

DOCUMENT_END = '''\\end{{document}}'''


def get_template(formulars: list):

    template = DOCUMENT_START

    for _ in range(len(formulars)):
        template += FORMULAR

    template += DOCUMENT_END
    template = template.format(*formulars)

    return template
