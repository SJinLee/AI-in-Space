"""pandoc 기본 reference.docx 의 스타일을 논문용(A4, 여백 25 mm, 본문 10 pt)으로 고친다."""
import re, sys, os, zipfile, shutil

src, out = sys.argv[1], sys.argv[2]
BODY, HEAD = '맑은 고딕', '맑은 고딕'
z = zipfile.ZipFile(src)
files = {n: z.read(n) for n in z.namelist()}
s = files['word/styles.xml'].decode('utf-8')

# 문서 기본값
s = re.sub(r'<w:rFonts w:asciiTheme="minorHAnsi"[^>]*/>',
           f'<w:rFonts w:ascii="{BODY}" w:eastAsia="{BODY}" w:hAnsi="{BODY}" w:cs="{BODY}" />', s, count=1)
s = s.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />', '<w:sz w:val="20" />\n        <w:szCs w:val="20" />', 1)
s = s.replace('<w:pPr>\n        <w:spacing w:after="200" />', '<w:pPr>\n        <w:spacing w:after="60" w:line="312" w:lineRule="auto" />\n        <w:jc w:val="both" />', 1)


def style(sid, fn):
    global s
    m = re.search(r'<w:style [^>]*w:styleId="%s"[^>]*>.*?</w:style>' % sid, s, flags=re.S)
    assert m, sid
    s = s[:m.start()] + fn(m.group(0)) + s[m.end():]


def set_rpr(b, sz=None, font=None, color=None, bold=None, noitalic=False):
    if '<w:rPr>' not in b:
        b = b.replace('</w:style>', '  <w:rPr>\n    </w:rPr>\n  </w:style>')
    b = re.sub(r'\s*<w:(sz|szCs)\b[^>]*/>', '', b) if sz else b
    b = re.sub(r'\s*<w:rFonts\b[^>]*/>', '', b) if font else b
    b = re.sub(r'\s*<w:color\b[^>]*/>', '', b) if color else b
    b = re.sub(r'\s*<w:(i|iCs)\b[^>]*/>', '', b) if noitalic else b
    add = ''
    if font: add += f'<w:rFonts w:ascii="{font}" w:eastAsia="{font}" w:hAnsi="{font}" w:cs="{font}" />'
    if bold is True and '<w:b ' not in b and '<w:b/>' not in b: add += '<w:b />'
    if color: add += f'<w:color w:val="{color}" />'
    if sz: add += f'<w:sz w:val="{sz}" /><w:szCs w:val="{sz}" />'
    return b.replace('<w:rPr>', '<w:rPr>' + add, 1)


def set_spacing(b, before, after, jc=None, extra=''):
    b = re.sub(r'\s*<w:spacing\b[^>]*/>', '', b)
    if jc: b = re.sub(r'\s*<w:jc\b[^>]*/>', '', b)
    if '<w:pPr>' not in b:
        b = re.sub(r'(<w:rPr>|</w:style>)', r'<w:pPr></w:pPr>\1', b, count=1)
    add = f'<w:spacing w:before="{before}" w:after="{after}" {extra}/>' + (f'<w:jc w:val="{jc}" />' if jc else '')
    return b.replace('<w:pPr>', '<w:pPr>' + add, 1)


style('BodyText', lambda b: set_spacing(b, 0, 80, jc='both'))
style('Compact', lambda b: set_rpr(set_spacing(b, 0, 0, jc='center', extra='w:line="240" w:lineRule="auto" '), sz=16))
style('Title', lambda b: set_rpr(set_spacing(b, 240, 120, jc='center'), sz=36, font=HEAD, color='000000'))
style('Subtitle', lambda b: set_rpr(set_spacing(b, 0, 240, jc='center'), sz=24, font=HEAD, color='000000'))
style('SubtitleChar', lambda b: set_rpr(b, sz=24, font=HEAD, color='000000'))
style('TitleChar', lambda b: set_rpr(b, sz=36, font=HEAD))
style('Author', lambda b: set_rpr(set_spacing(b, 0, 40, jc='center'), sz=22))
style('Date', lambda b: set_rpr(set_spacing(b, 0, 200, jc='center'), sz=20))
style('AbstractTitle', lambda b: set_rpr(set_spacing(b, 120, 80, jc='center'), sz=20, font=HEAD))
style('Abstract', lambda b: set_rpr(set_spacing(b, 0, 80, jc='both'), sz=18))
for sid, sz, bf in (('Heading1', 28, 360), ('Heading2', 24, 240), ('Heading3', 22, 200)):
    style(sid, lambda b, sz=sz, bf=bf: set_rpr(set_spacing(b, bf, 120, jc='left'), sz=sz, font=HEAD, color='000000', bold=True))
    style(sid + 'Char', lambda b, sz=sz: set_rpr(b, sz=sz, font=HEAD, color='000000', bold=True))
style('BlockText', lambda b: set_rpr(set_spacing(b, 40, 80), sz=18))
style('Caption', lambda b: set_rpr(set_spacing(b, 60, 120, jc='both'), sz=18, font=HEAD, noitalic=True))
style('VerbatimChar', lambda b: set_rpr(b, sz=18, font='Consolas'))
style('Normal', lambda b: set_rpr(b, font=BODY, sz=20))
style('Compact', lambda b: b.replace('<w:sz w:val="16" /><w:szCs w:val="16" />', '<w:sz w:val="15" /><w:szCs w:val="15" />'))
files['word/styles.xml'] = s.encode('utf-8')

th = files['word/theme/theme1.xml'].decode('utf-8')
def theme_font(th, key, font):
    m = re.search('<a:%s>.*?</a:%s>' % (key, key), th, flags=re.S)
    b = m.group(0).replace('<a:ea typeface=""/>', '<a:ea typeface="%s"/>' % font).replace('<a:font script="Hang" typeface="맑은 고딕"/>', '<a:font script="Hang" typeface="%s"/>' % font)
    b = re.sub(r'<a:latin typeface="[^"]*"[^>]*/>', '<a:latin typeface="%s"/>' % font, b)
    return th[:m.start()] + b + th[m.end():]
th = theme_font(th, 'minorFont', BODY)
th = theme_font(th, 'majorFont', HEAD)
files['word/theme/theme1.xml'] = th.encode('utf-8')

d = files['word/document.xml'].decode('utf-8')
d = re.sub(r'<w:sectPr\b.*?</w:sectPr>|<w:sectPr\b[^>]*/>',
           '<w:sectPr><w:pgSz w:w="11906" w:h="16838" /><w:pgMar w:top="1417" w:right="1417" w:bottom="1417" w:left="1417" w:header="709" w:footer="709" w:gutter="0" /></w:sectPr>',
           d, flags=re.S)
assert 'pgMar' in d
files['word/document.xml'] = d.encode('utf-8')

with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zo:
    for n, b in files.items():
        zo.writestr(n, b)
print('ok', out)
