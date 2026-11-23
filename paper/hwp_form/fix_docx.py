import sys, re, zipfile
src, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
d = files['word/document.xml'].decode('utf-8')
TW = 9638
def tbl(m):
    t = m.group(0)
    ws = [int(x) for x in re.findall(r'<w:gridCol w:w="(\d+)"', t)]
    k = TW / sum(ws)
    t = re.sub(r'<w:gridCol w:w="(\d+)"', lambda g: '<w:gridCol w:w="%d"' % round(int(g.group(1)) * k), t)
    t = t.replace('<w:tblW w:type="pct" w:w="5000" />', '<w:tblW w:type="dxa" w:w="%d" />' % TW)
    return t
d, n = re.subn(r'<w:tbl>.*?</w:tbl>', tbl, d, flags=re.S)
# 코드 문단을 한 칸짜리 표에 넣어 상자로 만든다(문단 테두리는 한글에서 아래 문단과 붙는다).
def codebox(m):
    edge = ''.join('<w:%s w:val="single" w:sz="4" w:space="0" w:color="808080" />' % k for k in ('top', 'left', 'bottom', 'right'))
    return ('<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="%d" /><w:tblBorders>%s</w:tblBorders>'
            '<w:tblCellMar><w:left w:w="113" w:type="dxa" /><w:right w:w="113" w:type="dxa" /></w:tblCellMar></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="%d" /></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa" /></w:tcPr>%s</w:tc></w:tr></w:tbl>'
            % (TW, edge, TW, TW, m.group(0)))
d, nbox = re.subn(r'<w:p><w:pPr><w:pStyle w:val="SourceCode" /></w:pPr>.*?</w:p>', codebox, d, flags=re.S)
print('code boxes', nbox)
# 별행 수식 바로 뒤에서 문장이 이어지는 문단("으로 분해된다" 등)은 들여쓰지 않는다.
PART = ('으로', '로 ', '이다', '이고', '이며', '을 ', '를 ', '은 ', '는 ', '와 ', '과 ', '에 ', '의 ', '이 ', '가 ')
paras = re.split(r'(<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>)', d, flags=re.S)
text = lambda x: re.sub(r'<[^>]+>', '', x).strip()
ncont = 0
for i in range(1, len(paras) - 2, 2):
    if re.fullmatch(r'MQ\d{4}QM', text(paras[i])) and text(paras[i + 2]).startswith(PART):
        q = paras[i + 2]
        if '<w:pStyle w:val="BodyText" />' in q or '<w:pStyle w:val="FirstParagraph" />' in q:
            paras[i + 2] = re.sub(r'(<w:pStyle w:val="(?:BodyText|FirstParagraph)" />)', r'\1<w:ind w:firstLine="0" />', q, count=1)
            ncont += 1
d = ''.join(paras)
print('continued paragraphs', ncont)
files['word/document.xml'] = d.encode('utf-8')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zo:
    for name, b in files.items(): zo.writestr(name, b)
print('tables', n)
