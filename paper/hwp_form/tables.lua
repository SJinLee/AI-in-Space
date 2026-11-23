-- 표의 열 너비: 각 열에서 가장 긴 낱말이 잘리지 않을 최소 너비를 먼저 주고,
-- 남는 폭을 내용 길이에 비례해 나눠 본문 폭 전체를 쓰게 한다.
local AVAIL = 125   -- 본문 폭 160 mm 를 7.5 pt 숫자 한 글자(약 1.35 mm) 단위로 센 값
local PAD = 3.4     -- 셀 좌우 여백 몫

local function width(s)
  s = s:gsub("MQ%d%d%d%dQM", "xx")
  local n = 0
  for _, c in utf8.codes(s) do
    n = n + ((c > 0x2E80) and 1.8 or 1)
  end
  return n
end

function Table(t)
  local n = #t.colspecs
  local minw, natw = {}, {}
  for i = 1, n do minw[i] = 2; natw[i] = 2 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      local col = 1
      for _, cell in ipairs(row.cells) do
        if cell.col_span == 1 then
          local s = pandoc.utils.stringify(cell.contents)
          local k = s:find("_") and 1.5 or 1
          local nat = width(s) * k
          if nat > 30 then nat = 30 end
          if nat > natw[col] then natw[col] = nat end
          for word in s:gmatch("%S+") do
            local w = width(word) * k
            if w > 16 then w = 16 end
            if w > minw[col] then minw[col] = w end
          end
        end
        col = col + cell.col_span
      end
    end
  end
  scan(t.head.rows)
  for _, b in ipairs(t.bodies) do scan(b.head); scan(b.body) end
  scan(t.foot.rows)
  local summin, sumextra = 0, 0
  for i = 1, n do
    summin = summin + minw[i] + PAD
    sumextra = sumextra + (natw[i] - minw[i])
  end
  local w, total = {}, 0
  local spare = AVAIL - summin
  for i = 1, n do
    w[i] = minw[i] + PAD
    if spare > 0 then
      if sumextra > 0 then
        local want = natw[i] - minw[i]
        local give = spare * want / sumextra
        if sumextra <= spare then give = want + (spare - sumextra) / n end
        w[i] = w[i] + give
      else
        w[i] = w[i] + spare / n
      end
    end
    total = total + w[i]
  end
  for i = 1, n do t.colspecs[i][2] = w[i] / total end
  return t
end
