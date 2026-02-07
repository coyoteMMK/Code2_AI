import csv
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

PATH = "../datasource/train.json"   # o data/valid.json, o el que quieras
OUT_CSV = "../datasource/distribucion_dataset.csv"
OUT_PLOT_COMPLEJIDAD = "../datasource/distribucion_complejidad.png"
OUT_PLOT_TIPOS = "../datasource/distribucion_tipos.png"
OUT_PLOT_APILADAS = "../datasource/tipo_vs_complejidad.png"
TITLE = "Distribución del dataset sintético por tipo de instrucción y complejidad de secuencias"

def infer_tipo(line):
    line = line.strip()
    if not line:
        return "Vacia"
    if line.startswith("Error:"):
        return "Error"
    # output válido empieza por "LD ", "ST ", "ADDS ", "SUBS "
    head = line.split()[0]
    if head in {"LD","ST","ADDS","SUBS"}:
        return head
    return "Otro"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1) Complejidad: nº de instrucciones por ejemplo
len_counter = Counter()

# 2) Distribución global por tipo (contando líneas)
tipo_counter = Counter()

# 3) Tipo vs complejidad (conteo de líneas por tipo, agrupado por longitud del bloque)
tipo_por_len = defaultdict(Counter)

for ex in data:
    out_lines = [ln for ln in ex["output"].split("\n") if ln.strip() != ""]
    L = len(out_lines)
    len_counter[L] += 1

    for ln in out_lines:
        t = infer_tipo(ln)
        tipo_counter[t] += 1
        tipo_por_len[L][t] += 1

# ======= PLOTS =======

# A) Histograma de complejidad (secuencias por nº de instrucciones)
xs = sorted(len_counter)
ys = [len_counter[x] for x in xs]
plt.figure()
plt.bar(xs, ys)
plt.xlabel("Nº de instrucciones en la secuencia")
plt.ylabel("Nº de ejemplos")
plt.title("Distribución de complejidad (longitud de secuencia)")
plt.xticks(xs)
plt.tight_layout()
plt.savefig(OUT_PLOT_COMPLEJIDAD, dpi=200)
plt.show()

# B) Barras: distribución por tipo (a nivel de línea de output)
tipos = [t for t,_ in tipo_counter.most_common()]
vals = [tipo_counter[t] for t in tipos]
plt.figure()
plt.bar(tipos, vals)
plt.xlabel("Tipo de instrucción")
plt.ylabel("Nº de líneas")
plt.title("Distribución del dataset por tipo de instrucción")
plt.tight_layout()
plt.savefig(OUT_PLOT_TIPOS, dpi=200)
plt.show()

# C) (Opcional) Tipo vs Complejidad: barras apiladas por longitud
tipos_base = ["LD","ST","ADDS","SUBS","Error"]
xs = sorted(tipo_por_len)
bottom = [0]*len(xs)

plt.figure()
for t in tipos_base:
    vals = [tipo_por_len[L][t] for L in xs]
    plt.bar(xs, vals, bottom=bottom, label=t)
    bottom = [b+v for b,v in zip(bottom, vals)]

plt.xlabel("Nº de instrucciones en la secuencia")
plt.ylabel("Nº de líneas")
plt.title("Tipo de instrucción vs complejidad (barras apiladas)")
plt.xticks(xs)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT_APILADAS, dpi=200)
plt.show()

print("Ejemplos por longitud:", dict(len_counter))
print("Líneas por tipo:", dict(tipo_counter))

# ======= CSV EXPORT =======
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([TITLE])
    writer.writerow([])
    writer.writerow(["Seccion", "Categoria", "Conteo"])
    for L in sorted(len_counter):
        writer.writerow(["Ejemplos por longitud", L, len_counter[L]])
    for t, c in tipo_counter.most_common():
        writer.writerow(["Lineas por tipo", t, c])
