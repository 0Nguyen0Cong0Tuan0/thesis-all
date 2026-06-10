import fitz, os, re, json, glob

BASE = r"D:\Thesis\All\papers\3dgs_papers"
WORK = os.path.join(BASE, "_work")
IMGDIR = os.path.join(BASE, "_review", "images")
os.makedirs(IMGDIR, exist_ok=True)

cats = ["00_foundational","01_fast_densification","02_specular_reflection",
        "03_quality_structure","04_method_blocks","05_surveys"]

index = []
for cat in cats:
    cdir = os.path.join(BASE, cat)
    if not os.path.isdir(cdir):
        continue
    for pdf in sorted(glob.glob(os.path.join(cdir, "*.pdf"))):
        stem = os.path.splitext(os.path.basename(pdf))[0]
        rec = {"cat": cat, "stem": stem, "pages": 0, "img": None, "err": None}
        try:
            doc = fitz.open(pdf)
            rec["pages"] = doc.page_count
            # --- text: first ~6000 words (covers abstract/intro/method for most) ---
            txt = []
            wc = 0
            for pg in doc:
                t = pg.get_text()
                txt.append(t)
                wc += len(t.split())
                if wc > 6500:
                    break
            full = "\n".join(txt)
            with open(os.path.join(WORK, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(full)
            # --- largest embedded raster image across first 4 pages ---
            best = None  # (area, xref, w, h)
            for pi in range(min(4, doc.page_count)):
                for im in doc[pi].get_images(full=True):
                    xref = im[0]
                    try:
                        pm = fitz.Pixmap(doc, xref)
                    except Exception:
                        continue
                    w, h = pm.width, pm.height
                    area = w * h
                    if w >= 350 and h >= 200 and area > (best[0] if best else 0):
                        best = (area, xref, w, h)
            saved = None
            if best:
                try:
                    pm = fitz.Pixmap(doc, best[1])
                    if pm.n - pm.alpha >= 4:  # CMYK -> RGB
                        pm = fitz.Pixmap(fitz.csRGB, pm)
                    out = os.path.join(IMGDIR, stem + "_fig.png")
                    pm.save(out)
                    saved = os.path.basename(out)
                except Exception as e:
                    rec["err"] = "imgsave:" + str(e)
            if not saved:
                # fallback: render page 1 (teaser usually here) at 130 dpi
                pix = doc[0].get_pixmap(dpi=130)
                out = os.path.join(IMGDIR, stem + "_p1.png")
                pix.save(out)
                saved = os.path.basename(out)
            rec["img"] = saved
            doc.close()
        except Exception as e:
            rec["err"] = str(e)
        index.append(rec)
        print(f"{cat[:2]} {stem[:40]:40} pages={rec['pages']:>3} img={rec['img']}")

with open(os.path.join(WORK, "_index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)
print("TOTAL", len(index))
