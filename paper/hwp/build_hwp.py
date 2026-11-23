import sys, json, time
import win32com.client as win32
src, out, pdf, mj = sys.argv[1:5]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
maths = json.load(open(mj, encoding="utf-8"))
hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
hwp.SetMessageBoxMode(0x00010000 | 0x00000001 | 0x00000010)
missing = []
try:
    print("open", hwp.Open(src, "", ""), hwp.PageCount)
    c = hwp.HeadCtrl; nt = 0
    while c:
        if c.CtrlID == "tbl":
            pr = c.Properties; pr.SetItem("TreatAsChar", 1); c.Properties = pr; nt += 1
        c = c.Next
    print("tables as char", nt)
    t0 = time.time()
    for i, m in enumerate(maths[:limit]):
        tok = "MQ%04dQM" % i
        hwp.HAction.Run("MoveDocBegin")
        fs = hwp.HParameterSet.HFindReplace
        hwp.HAction.GetDefault("RepeatFind", fs.HSet)
        fs.FindString = tok
        fs.Direction = hwp.FindDir("AllDoc")
        fs.IgnoreMessage = 1
        fs.WholeWordOnly = 0
        fs.FindRegExp = 0
        if not hwp.HAction.Execute("RepeatFind", fs.HSet):
            missing.append(tok); continue
        h = hwp.CharShape.Item("Height")
        hwp.HAction.Run("Delete")
        eq = hwp.HParameterSet.HEqEdit
        hwp.HAction.GetDefault("EquationCreate", eq.HSet)
        eq.EqFontName = "HYhwpEQ"
        eq.string = m["hwp"]
        eq.BaseUnit = h
        hwp.HAction.Execute("EquationCreate", eq.HSet)
        hwp.FindCtrl()
        hwp.HAction.GetDefault("EquationModify", eq.HSet)
        eq.string = eq.VisualString
        eq.Version = "Equation Version 60"
        hwp.HAction.Execute("EquationModify", eq.HSet)
        c = hwp.CurSelectedCtrl
        if c is not None:
            pr = c.Properties
            pr.SetItem("TreatAsChar", 1)
            c.Properties = pr
        hwp.HAction.Run("Cancel")
        if m["display"]:
            hwp.HAction.Run("ParagraphShapeAlignCenter")
    print("inserted", len(maths[:limit]) - len(missing), "missing", missing, round(time.time() - t0), "s")
    hwp.HAction.Run("MoveDocBegin")
    hwp.HAction.Run("SelectAll")
    def show(tag):
        q = hwp.ParaShape
        print(tag, {k: q.Item(k) for k in ("LineSpacingType", "LineSpacing", "FontLineHeight", "KeepWithNext", "KeepLinesTogether", "WidowOrphan", "PagebreakBefore", "PrevSpacing", "NextSpacing")})
    show("before")
    act = hwp.CreateAction("ParagraphShape")
    pset = act.CreateSet()
    pset.SetItem("LineSpacingType", 0)
    pset.SetItem("LineSpacing", 160)
    pset.SetItem("FontLineHeight", 0)
    print("linespacing", act.Execute(pset))
    show("after")
    hwp.HAction.Run("Cancel")
    hwp.HAction.Run("MoveDocBegin")
    print("save", hwp.SaveAs(out, "HWP", ""), "pdf", hwp.SaveAs(pdf, "PDF", ""), "pages", hwp.PageCount)
finally:
    hwp.Clear(1); hwp.Quit()
