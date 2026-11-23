"""main.tex -> pandoc 이 읽기 쉬운 main_hwp.tex (번호·인용·참고문헌을 글자로 풀어 넣음)."""
import re, sys, os, shutil

paper, out = sys.argv[1], sys.argv[2]
rd = lambda p: open(os.path.join(paper, p), encoding='utf-8').read()
src = rd('main.tex')
aux = rd('main.aux')
bbl = rd('main.bbl')

labels = {m.group(1): m.group(2) for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', aux)}

body = src.split(r'\begin{document}')[1].split(r'\end{document}')[0]

# ---- 보호할 조각(verbatim) ----
keep = []
def stash(txt):
    keep.append(txt)
    return f'@@KEEP{len(keep)-1}@@'

def brace_arg(s, i):
    """s[i] == '{' 에서 시작하는 인자의 (내용, 끝 다음 위치)."""
    assert s[i] == '{', s[i:i+30]
    d = 0
    for j in range(i, len(s)):
        if s[j] == '{' and s[j-1] != '\\': d += 1
        elif s[j] == '}' and s[j-1] != '\\':
            d -= 1
            if d == 0:
                return s[i+1:j], j+1
    raise ValueError

# ---- lstlisting ----
def lst(m):
    opts, code = m.group(1), m.group(2)
    i = opts.index('caption=') + len('caption=')
    cap, _ = brace_arg(opts, i)
    lab = re.search(r'label=\{([^}]+)\}', opts).group(1)
    n = labels[lab]
    return f'\n\n\\noindent\\textbf{{코드 {n}.}} {cap}\n\n' + stash('\\begin{verbatim}\n' + code + '\\end{verbatim}') + '\n\n'
body = re.sub(r'\\begin\{lstlisting\}\[(.*?)\]\n(.*?)\\end\{lstlisting\}', lst, body, flags=re.S)

# ---- 주석 제거 ----
body = '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in body.split('\n'))

# ---- \input ----
def inp(m):
    t = rd(m.group(1) + '.tex')
    return t
body = re.sub(r'\\input\{([^}]+)\}', inp, body)

# ---- tikz 그림 ----
m = re.search(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', body, flags=re.S)
os.makedirs(os.path.join(out, 'tikz'), exist_ok=True)
open(os.path.join(out, 'tikz', 'fig_float32.tex'), 'w', encoding='utf-8').write(
    '\\documentclass[11pt]{article}\n\\usepackage{kotex}\n\\usepackage{tikz}\n\\usepackage[paperwidth=14cm,paperheight=3.4cm,margin=2mm]{geometry}\n'
    '\\pagestyle{empty}\n\\begin{document}\n\\centering\n' + m.group(0) + '\n\\end{document}\n')
body = body.replace(m.group(0), '\\includegraphics[width=0.8\\linewidth]{fig_float32}')

# ---- texorpdfstring ----
while True:
    i = body.find('\\texorpdfstring{')
    if i < 0: break
    a, j = brace_arg(body, i + len('\\texorpdfstring'))
    b, k = brace_arg(body, j)
    body = body[:i] + a + body[k:]

# ---- 노트·예 ----
cnt = {'note': 0, 'example': 0}
name = {'note': '노트', 'example': '예'}
def box(m):
    env, title = m.group(1), m.group(2)
    cnt[env] += 1
    t = f' ({title})' if title else ''
    return f'\\begin{{quote}}\n\\textbf{{{name[env]} {cnt[env]}{t}.}} '
body = re.sub(r'\\begin\{(note|example)\}(?:\[(.*)\])?[ \t]*\n', box, body)
body = re.sub(r'\\end\{(note|example)\}', r'\\end{quote}', body)

# ---- 그림·표 제목 번호 ----
def number_caps(body, env, word):
    n = 0
    def f(m):
        nonlocal n
        blk = m.group(0)
        if '\\caption{' not in blk:
            return blk
        n += 1
        return blk.replace('\\caption{', f'\\caption{{\\textbf{{{word} {n}.}} ', 1)
    return re.sub(r'\\begin\{%s\}.*?\\end\{%s\}' % (env, env), f, body, flags=re.S), n
body, nfig = number_caps(body, 'figure', '그림')
body, ntab = number_caps(body, 'table', '표')

# ---- 수식 번호 ----
neq = 0
def eq(m):
    global neq
    neq += 1
    inner = re.sub(r'\\label\{[^}]*\}', '', m.group(1)).rstrip()
    return f'\\begin{{equation}}{inner}\n  \\qquad\\qquad (\\mathrm{{{neq}}})\n\\end{{equation}}'
body = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', eq, body, flags=re.S)

# ---- 절 번호 ----
sec = [0, 0]; appendix = False
def heading(m):
    global appendix
    kind = m.group(1)
    if kind == 'section':
        sec[0] += 1; sec[1] = 0
        num = f'부록 {chr(64 + sec[0])}.' if appendix else f'{sec[0]}'
    else:
        sec[1] += 1
        num = f'{chr(64 + sec[0]) if appendix else sec[0]}.{sec[1]}'
    return f'\\{kind}{{{num} '
body = re.sub(r'\\renewcommand\{\\[a-z]+name\}\{[^}]*\}\n', '', body)
parts = re.split(r'\\appendix(?![A-Za-z])', body)
assert len(parts) == 2
parts[0] = re.sub(r'\\(section|subsection)\{', heading, parts[0])
appendix = True; sec = [0, 0]
parts[1] = re.sub(r'\\(section|subsection)\{', heading, parts[1])
body = parts[0] + parts[1]

# ---- 상호 참조 ----
missing = set()
def ref(m):
    k = m.group(2)
    if k not in labels: missing.add(k); return '??'
    return f'({labels[k]})' if m.group(1) == 'eqref' else labels[k]
body = re.sub(r'\\(eqref|ref)\{([^}]+)\}', ref, body)
body = re.sub(r'[ \t]*\\label\{[^}]*\}', '', body)

# ---- 인용 ----
order = re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', bbl)
num = {k: i + 1 for i, k in enumerate(order)}
def cite(m):
    ns = sorted(num[k.strip()] for k in m.group(1).split(','))
    runs, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        runs.append(f'{ns[i]}--{ns[j]}' if j - i >= 2 else ', '.join(str(x) for x in ns[i:j + 1]))
        i = j + 1
    return '{[}' + ', '.join(runs) + '{]}'
body = re.sub(r'\\citep\{([^}]*)\}', cite, body)

# ---- 참고문헌 ----
ents = re.split(r'\\bibitem\[[^\]]*\]\{[^}]+\}', bbl.split('\\BIBdecl')[-1])[1:]
bib = ['\\section*{참고문헌}', '']
for i, e in enumerate(ents, 1):
    e = e.split('\\end{thebibliography}')[0]
    e = re.sub(r'\\BIBentry(ALT|STD)interwordspacing', '', e)
    e = re.sub(r'\\(newblock|relax)\b', '', e)
    e = ' '.join(e.split())
    bib += [f'{{}}[{i}] {e}', '']
body = re.sub(r'\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}', lambda m: '\n'.join(bib), body)

# ---- Q1.. 목록 ----
def qlist(m):
    n = 0
    def it(_):
        nonlocal n
        n += 1
        return f'\n\n\\noindent\\textbf{{Q{n}.}} '
    return re.sub(r'[ \t]*\\item\s*', it, m.group(1)) + '\n\n'
body = re.sub(r'\\begin\{enumerate\}\[label=[^\n]*\n(.*?)\\end\{enumerate\}', qlist, body, flags=re.S)
body = re.sub(r'\\begin\{(itemize|enumerate)\}\[[^\]\n]*\]', r'\\begin{\1}', body)

# ---- 자잘한 것 ----
body = re.sub(r'\\path\|([^|]*)\|', lambda m: '\\texttt{' + m.group(1).replace('_', '\\_') + '}', body)
body = body.replace('\\hspace*{2em}', '~~~~')
body = re.sub(r'\\cmidrule\([a-z]*\)\{[^}]*\}', '', body)
body = re.sub(r'\\setlength\{[^}]*\}\{[^}]*\}', '', body)
body = re.sub(r'\\renewcommand\{\\[a-z]+name\}\{[^}]*\}\n', '', body)
body = re.sub(r'\\(addlinespace|medskip|scriptsize|small|appendix)(?![A-Za-z])', '', body)
body = re.sub(r'(\\includegraphics\[[^\]]*\])\{([^}]+)\}', r'\1{figs/\2.png}', body)
# ---- 수식을 자리표시 글자로 바꾸고, 한글 수식 스크립트를 따로 저장 ----
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tex2hwpeq
macros = {m.group(1): m.group(2) for m in re.finditer(r'^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$', rd('tables/numbers.tex'), flags=re.M)}
maths = []
def disp(m):
    maths.append({'tex': ' '.join(m.group(1).split()), 'display': True})
    return f'\n\nMQ{len(maths)-1:04d}QM\n\n'
def split_commas(tex):
    """긴 줄 안 수식은 최상위 쉼표에서 나눠 줄바꿈이 되게 한다."""
    parts, d, cur, i = [], 0, '', 0
    while i < len(tex):
        c = tex[i]
        if c == '\\' and i + 1 < len(tex):
            cur += tex[i:i+2]; i += 2; continue
        if c == '{': d += 1
        elif c == '}': d -= 1
        if c == ',' and d == 0:
            parts.append(cur); cur = ''
        else:
            cur += c
        i += 1
    parts.append(cur)
    return [re.sub(r'^(\\,|\s)+', '', p).strip() for p in parts]
def inl(m):
    tex = ' '.join(m.group(1).split())
    pieces = split_commas(tex) if len(tex) > 70 else [tex]
    toks = []
    for p in pieces:
        maths.append({'tex': p, 'display': False})
        toks.append(f'MQ{len(maths)-1:04d}QM')
    return ', '.join(toks)
body = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', disp, body, flags=re.S)
body = re.sub(r'(?<!\\)\\\[(.*?)\\\]', disp, body, flags=re.S)
body = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', inl, body, flags=re.S)
for x in maths:
    x['hwp'] = tex2hwpeq.conv(x['tex'], macros)
json.dump(maths, open(os.path.join(out, 'maths.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('maths', len(maths), 'display', sum(x['display'] for x in maths), 'unknown', tex2hwpeq.unknown)
body = re.sub(r'@@KEEP(\d+)@@', lambda m: keep[int(m.group(1))], body)

numbers = '\n'.join(ln for ln in rd('tables/numbers.tex').split('\n') if ln.strip().startswith('\\newcommand'))
pre = r'''\documentclass{article}
\usepackage{amsmath,amssymb,graphicx,booktabs,multirow}
''' + numbers + r'''
\newcommand{\code}[1]{\texttt{#1}}
\title{저궤도 우주 방사선 단일 사건 오류(SEU)가 인공신경망 가중치에 미치는 영향}
\author{이태린\textsuperscript{1}, 최재원\textsuperscript{2}, 조교: 이유림\textsuperscript{3}, 지도교사: 이성진\textsuperscript{3} \and \textsuperscript{1}민락중학교, \textsuperscript{2}선덕중학교, \textsuperscript{3}대진대학교}
\date{2026년 10월}
\begin{document}
'''
open(os.path.join(out, 'main_hwp.tex'), 'w', encoding='utf-8', newline='\n').write(pre + body + '\n\\end{document}\n')

os.makedirs(os.path.join(out, 'figs'), exist_ok=True)
for f in re.findall(r'figs/([\w]+\.png)', body):
    p = os.path.join(paper, 'figures', f)
    if os.path.exists(p): shutil.copy(p, os.path.join(out, 'figs', f))
print('figures', nfig, 'tables', ntab, 'equations', neq, 'notes', cnt, 'refs', len(ents), 'missing labels', missing)
for k in ('fig:e5', 'tab:related', 'eq:tmr', 'sec:related', 'note:saa'):
    print(k, labels.get(k))
print(bib[2][:200]); print(bib[-2][:200])
