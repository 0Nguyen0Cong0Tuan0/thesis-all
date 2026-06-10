import os, glob, json, re

BASE = r"D:\Thesis\All\papers\3dgs_papers"
WORK = os.path.join(BASE, "_work")
idx = json.load(open(os.path.join(WORK, "_index.json"), encoding="utf-8"))

# group by category
bycat = {}
for r in idx:
    bycat.setdefault(r["cat"], []).append(r)

def clean(t):
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

WORDS = 1500
for cat, recs in bycat.items():
    out = []
    for r in recs:
        p = os.path.join(WORK, r["stem"] + ".txt")
        if not os.path.exists(p):
            continue
        t = clean(open(p, encoding="utf-8").read())
        words = t.split(" ")
        head = " ".join(words[:WORDS])
        out.append(f"\n\n{'='*90}\n### {r['stem']}  (pages={r['pages']}, img={r['img']})\n{'='*90}\n{head}")
    with open(os.path.join(WORK, f"digest_{cat}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("wrote digest for", cat, "->", len(recs), "papers")
