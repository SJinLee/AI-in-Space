"""논문에서 쓰는 LaTeX 수식 부분집합 -> 한글 수식 편집기 스크립트."""
import re

SYM = {
    'alpha': 'alpha', 'beta': 'beta', 'delta': 'delta', 'lambda': 'lambda', 'pi': 'pi',
    'approx': 'approx', 'ast': 'ast', 'cdot': 'cdot', 'cdots': 'cdots', 'ldots': 'cdots',
    'ge': 'geq', 'le': 'leq', 'in': 'in', 'infty': 'inf', 'll': '<<', 'ln': 'ln', 'log': 'log', 'exp': 'exp',
    'pm': '+-', 'sim': 'sim', 'sum': 'sum', 'times': 'times', 'to': 'rarrow', 'leftarrow': 'larrow',
    'square': '□', 'qquad': '~~~~', 'quad': '~~', ',': '`', ';': '~', ' ': '~', '%': '%', '{': '{rm "{" it}', '}': '{rm "}" it}',
    '!': '', '_': '_', 'big': '', 'displaystyle': '', 'textstyle': '', 'limits': '', 'nolimits': '',
}
TOK = re.compile(r'\\[A-Za-z]+|\\.|[0-9]+(?:[.,][0-9]+)*|\s+|.', re.S)
unknown = set()


class P:
    def __init__(self, s):
        self.t = [x for x in TOK.findall(s) if not x.isspace()]
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        x = self.t[self.i]; self.i += 1; return x

    def raw_group(self):
        """{...} 의 원문(텍스트용)."""
        assert self.next() == '{'
        d, out = 1, []
        while True:
            x = self.next()
            if x == '{': d += 1
            elif x == '}':
                d -= 1
                if d == 0: break
            out.append(x)
        return out

    def arg(self):
        if self.peek() == '{':
            self.next()
            r = self.seq('}')
            return r
        return self.atom()

    def seq(self, end=None):
        out = []
        while self.peek() is not None:
            if self.peek() == end:
                self.next(); break
            out.append(self.atom())
        return ' '.join(x for x in out if x != '')

    def atom(self):
        x = self.next()
        if x == '{':
            inner = self.seq('}')
            return ',' if inner == ',' else '{' + inner + '}'
        if x in '^_':
            return x + '{' + self.arg() + '}'
        if x == '~':
            return '~'
        if x.startswith('\\'):
            c = x[1:]
            if c in ('frac', 'tfrac', 'dfrac'):
                a = self.arg(); b = self.arg()
                return '{' + a + '} over {' + b + '}'
            if c == 'sqrt':
                return 'sqrt {' + self.arg() + '}'
            if c == 'bar':
                return 'bar {' + self.arg() + '}'
            if c in ('mathrm', 'text', 'mathbb', 'mathit', 'textbf', 'mathbf', 'operatorname'):
                raw = self.raw_group()
                txt = ''.join(' ' if r in ('\\ ', '~') else r.lstrip('\\') if r in ('\\_', '\\%') else r for r in raw)
                if re.fullmatch(r'[A-Za-z0-9]+', txt):
                    return '{rm ' + txt + ' it}'
                return '{rm "' + txt + '" it}'
            if c in ('left', 'right'):
                d = self.next()
                d = {'\\{': 'lbrace', '\\}': 'rbrace', '.': ''}.get(d, d)
                return (c.upper() + ' ' + d) if d else ''
            if c in SYM:
                return SYM[c]
            unknown.add(x)
            return c
        return x


def conv(tex, macros=None):
    if macros:
        tex = re.sub(r'\\([A-Za-z]+)(\{\})?', lambda m: macros[m.group(1)] if m.group(1) in macros else m.group(0), tex)
    tex = re.sub(r'(?<=text)\{([^{}]*)\}', lambda m: '{' + m.group(1).replace(' ', '~') + '}', tex)
    s = P(tex).seq()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


if __name__ == '__main__':
    for t in [r'w = (-1)^{S}\times 2^{E-127}\times\left(1+\frac{M}{2^{23}}\right)', r'k_{50}^{\mathrm{TMR}}\approx\sqrt{\frac{32\,N_w\ln 2}{3\,q}}.',
              r'P(\text{치명 없음})=\sum_k e^{-\lambda}\lambda^k/k!\,(1-q)^k', r'9{,}620', r'r=10^{-6}', r'\pm\infty', r'N_{\mathrm{bits}}=32N_w', r'\bar{k}', r'0\to1']:
        print(conv(t))
    print(unknown)
