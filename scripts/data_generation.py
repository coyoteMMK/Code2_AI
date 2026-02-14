# scripts/data-generation.py
import argparse
import json
import random
import time

# ======= Tu generador (resumen de tu lógica) =======
REGISTROS = [f"r{i:X}" for i in range(16)]

PLANTILLAS = {
    "LD": [
        "Carga el valor de la dirección {} en r{}",
        "Lee la dirección {} y guárdala en r{}",
        "Trae el contenido de {} y ponlo en r{}",
        "Copia lo que hay en {} a r{}",
        "Obtén el dato de {} y almacénalo en r{}",
        "Toma el valor de {} y colócalo en r{}",
        "Carga la dirección {} en el registro {}",
        "Lee {} y almacénalo en el registro {}",
        "Trae el dato de {} y ponlo en registro {}"
    ],
    "ST": [
        "Guarda el contenido de r{} en la dirección {}",
        "Almacena el valor de r{} en {}",
        "Escribe r{} en la memoria {}",
        "Coloca lo que hay en r{} dentro de {}",
        "Pon el dato de r{} en la dirección {}",
        "Graba el valor de r{} en la posición de memoria {}",
        "Guarda el registro {} en la dirección {}",
        "Escribe el contenido del registro {} en memoria {}",
        "Almacena el registro {} en la posición {}"
    ],
    "ADDS": [
        "Suma r{} y r{} y guarda el resultado en r{}",
        "Añade r{} a r{} y almacena en r{}",
        "Calcula la suma de r{} y r{} y pon el resultado en r{}",
        "Combina r{} con r{} y guarda en r{}",
        "Haz la suma entre r{} y r{} y guárdala en r{}",
        "Une r{} y r{} y deja el resultado en r{}",
        "Suma el registro {} al registro {} y guarda en registro {}",
        "Suma r{} al r{} y almacena el resultado en r{}",
        "Añade el registro {} con el registro {} y pon en r{}",
        "Calcula r{} más r{} y guarda en el registro {}",
        "Combina registro {} con registro {} y deja en r{}",
        "Suma la registro {} a la registro {} y guarda en registro {}"
    ],
    "SUBS": [
        "Resta r{} y r{} y guárdalo en r{}",
        "Calcula r{} menos r{} y guarda en r{}",
        "Sustrae r{} de r{} y pon el resultado en r{}",
        "Haz r{} - r{} y almacena en r{}",
        "Obtén la diferencia entre r{} y r{} y guarda en r{}",
        "Quita r{} a r{} y almacénalo en r{}",
        "Resta el registro {} del registro {} y guarda en registro {}",
        "Sustrae r{} de r{} y almacena en r{}",
        "Quita el registro {} a r{} y pon en r{}",
        "Calcula r{} menos r{} y guarda en el registro {}",
        "Obtén la diferencia de r{} y r{} y deja en r{}",
        "Resta registro {} de registro {} y almacena en r{}"
    ]
}

def variar_capitalizacion(frase):
    variantes = [
        frase.lower(),
        frase.upper(),
        frase.capitalize(),
        " ".join([w.capitalize() for w in frase.split()]),
        frase
    ]
    return random.choice(variantes)

def registro_invalido():
    opciones = [
        "",
        "r",
        "r" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=1)),
        "r" + str(random.randint(16, 255)),
        "r" + "".join(random.choices("XYZ123", k=random.randint(2, 4)))
    ]
    return random.choice(opciones)

def registro_para_nl(registro):
    if registro.startswith("r") and len(registro) > 1:
        return registro[1:]
    return registro

def direccion_invalida():
    opciones = []
    opciones.append("")
    opciones.append("".join(random.choices("0123456789ABCDEF", k=random.randint(5, 6))))
    base = "".join(random.choices("0123456789ABCDEF", k=random.randint(2, 3)))
    ruido = "".join(random.choices("GHIJKLMNOPQRSTUVWXYZ@$%", k=random.randint(1, 2)))
    opciones.append("".join(random.sample(base + ruido, len(base) + len(ruido))))
    opciones.append("mem" + str(random.randint(100, 999)))
    opciones.append("0A" + random.choice([" ", "@", "!"]) + random.choice("123GZ"))
    return random.choice(opciones)

def generar_par():
    tipo = random.choice(["LD", "ST", "ADDS", "SUBS"])
    es_error = random.random() < 0.1
    error_tipo = random.choice(["reg", "dir", "ambos"]) if es_error else "ninguno"

    if tipo in ["ADDS", "SUBS"]:
        r1 = registro_invalido() if error_tipo in ["reg", "ambos"] and random.random() < 0.5 else random.choice(REGISTROS)
        r2 = registro_invalido() if error_tipo in ["reg", "ambos"] and random.random() < 0.5 else random.choice(REGISTROS)
        r3 = registro_invalido() if error_tipo in ["reg", "ambos"] and random.random() < 0.5 else random.choice(REGISTROS)

        nl = variar_capitalizacion(
            random.choice(PLANTILLAS[tipo]).format(
                registro_para_nl(r1),
                registro_para_nl(r2),
                registro_para_nl(r3),
            )
        )

        errores = []
        for r in [r1, r2, r3]:
            r_display = registro_para_nl(r)
            if not r or r == "r":
                errores.append(f"registro '{r_display}' ambiguo")
            elif not r.startswith("r") or len(r) < 2:
                errores.append(f"registro '{r_display}' mal formado")
            elif r not in REGISTROS:
                errores.append(f"registro '{r_display}' invalido")

        if errores:
            code2 = f"Error: {' y '.join(errores)}"
        else:
            code2 = f"{tipo} {r3},{r1},{r2}"

    elif tipo == "LD":
        reg = registro_invalido() if error_tipo in ["reg", "ambos"] else random.choice(REGISTROS)
        direccion = direccion_invalida() if error_tipo in ["dir", "ambos"] else "".join(random.choices("0123456789ABCDEF", k=random.randint(1, 4)))
        nl = variar_capitalizacion(
            random.choice(PLANTILLAS["LD"]).format(direccion, registro_para_nl(reg))
        )

        errores = []
        reg_display = registro_para_nl(reg)
        if not reg or reg == "r":
            errores.append(f"registro '{reg_display}' ambiguo")
        elif not reg.startswith("r") or len(reg) < 2:
            errores.append(f"registro '{reg_display}' mal formado")
        elif reg not in REGISTROS:
            errores.append(f"registro '{reg_display}' invalido")

        if not direccion:
            errores.append("dirección vacia (ambigua)")
        else:
            if any(c not in "0123456789ABCDEFabcdef" for c in direccion):
                errores.append(f"dirección '{direccion}' contiene caracteres no hexadecimales")
            if len(direccion) > 4:
                errores.append(f"dirección '{direccion}' excede 4 digitos")

        if errores:
            code2 = f"Error: {' y '.join(errores)}"
        else:
            d = direccion.upper().rjust(4, "0")
            v = d[-2:]
            rd_base = d[:-2] or "00"
            code2 = f"LD {reg},[rD+H'{v}']  ; rD = {rd_base}00"

    else:  # ST
        reg = registro_invalido() if error_tipo in ["reg", "ambos"] else random.choice(REGISTROS)
        direccion = direccion_invalida() if error_tipo in ["dir", "ambos"] else "".join(random.choices("0123456789ABCDEF", k=random.randint(1, 4)))
        nl = variar_capitalizacion(
            random.choice(PLANTILLAS["ST"]).format(registro_para_nl(reg), direccion)
        )

        errores = []
        reg_display = registro_para_nl(reg)
        if not reg or reg == "r":
            errores.append(f"registro '{reg_display}' ambiguo")
        elif not reg.startswith("r") or len(reg) < 2:
            errores.append(f"registro '{reg_display}' mal formado")
        elif reg not in REGISTROS:
            errores.append(f"registro '{reg_display}' invalido")

        if not direccion:
            errores.append("dirección vacía (ambigua)")
        else:
            if any(c not in "0123456789ABCDEFabcdef" for c in direccion):
                errores.append(f"dirección '{direccion}' contiene caracteres no hexadecimales")
            if len(direccion) > 4:
                errores.append(f"dirección '{direccion}' excede 4 digitos")

        if errores:
            code2 = f"Error: {' y '.join(errores)}"
        else:
            d = direccion.upper().rjust(4, "0")
            v = d[-2:]
            rd_base = d[:-2] or "00"
            code2 = f"ST [rD+H'{v}'],{reg}  ; rD = {rd_base}00"

    return nl, code2

def generar_bloque(max_instrucciones=11, preserve_newlines=False):
    # distribución parecida a la que usas (más probabilidad en longitudes medias-altas)
    n = random.choices(list(range(1, max_instrucciones + 1)), weights=[1,1,2,3,4,5,5,4,3,2,1][:max_instrucciones])[0]
    lineas_nl, lineas_code2 = [], []
    for _ in range(n):
        nl, code = generar_par()
        lineas_nl.append(nl)
        lineas_code2.append(code)

    input_text = "\n".join(lineas_nl)
    output_text = "\n".join(lineas_code2)
    if preserve_newlines:
        input_text = input_text.replace("\n", " SALTO ")
        output_text = output_text.replace("\n", " SALTO ")

    return {
        "task": "nl_to_code2",
        "input": input_text,
        "output": output_text,
    }

def generar_dataset(n_ejemplos, max_instrucciones=11, preserve_newlines=False):
    dataset = []
    seen = set()
    intentos = 0
    while len(dataset) < n_ejemplos:
        intentos += 1
        ex = generar_bloque(
            max_instrucciones=max_instrucciones,
            preserve_newlines=preserve_newlines,
        )
        key = ex["input"] + "|" + ex["output"]
        if key not in seen:
            seen.add(key)
            dataset.append(ex)
        if len(dataset) % 5000 == 0 and len(dataset) != 0:
            print(f"▌ Progreso: {len(dataset)}/{n_ejemplos} ejemplos únicos...")
    print(f"🔍 Intentos totales: {intentos}")
    return dataset

def max_lens(examples):
    max_in, max_out = 0, 0
    max_in_lines, max_out_lines = 0, 0
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        max_in = max(max_in, len(inp))
        max_out = max(max_out, len(out))
        max_in_lines = max(max_in_lines, inp.count("\n") + 1 if inp else 0)
        max_out_lines = max(max_out_lines, out.count("\n") + 1 if out else 0)
    return max_in, max_out, max_in_lines, max_out_lines


# ======= Main =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../datasource_18K", help="Carpeta de salida")
    ap.add_argument("--n_total", type=int, default=18000, help="Número total de ejemplos a generar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_instr", type=int, default=11)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--valid_ratio", type=float, default=0.20)  
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--test_n", type=int, default=0, help="Tamaño fijo de test (0 para usar test_ratio)")
    ap.add_argument(
        "--preserve_newlines",
        action="store_true",
        help="Reemplaza saltos de linea por SALTO en input/output",
    )
    args = ap.parse_args()

    if args.test_n > 0:
        if args.train_ratio + args.valid_ratio <= 0:
            raise ValueError("train_ratio + valid_ratio debe ser mayor que 0")
    else:
        if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + valid_ratio + test_ratio debe sumar 1.0")

    random.seed(args.seed)

    print("\n⚙️ INICIANDO GENERADOR")
    time.sleep(0.2)

    dataset = generar_dataset(
        args.n_total,
        max_instrucciones=args.max_instr,
        preserve_newlines=args.preserve_newlines,
    )
    random.shuffle(dataset)

    max_in, max_out, max_in_lines, max_out_lines = max_lens(dataset)
    print("\n📏 Longitudes máximas en el dataset (antes del split):")
    print(f"   Input : {max_in} caracteres | {max_in_lines} líneas")
    print(f"   Output: {max_out} caracteres | {max_out_lines} líneas")

    if args.test_n > 0:
        if args.test_n > args.n_total:
            raise ValueError("test_n no puede ser mayor que n_total")
        n_test = args.test_n
        restante = args.n_total - n_test
        ratio_total = args.train_ratio + args.valid_ratio
        n_train = int((args.train_ratio / ratio_total) * restante)
        n_valid = restante - n_train
    else:
        n_train = int(args.train_ratio * args.n_total)
        n_valid = int(args.valid_ratio * args.n_total)
        n_test = args.n_total - n_train - n_valid

    train = dataset[:n_train]
    valid = dataset[n_train:n_train + n_valid]
    test = dataset[n_train + n_valid:] if n_test > 0 else []

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    with open(f"{args.out_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(f"{args.out_dir}/valid.json", "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    if n_test > 0:
        with open(f"{args.out_dir}/test.json", "w", encoding="utf-8") as f:
            json.dump(test, f, indent=2, ensure_ascii=False)

    print("\n✅ Dataset generado:")
    print(f"   Train: {len(train)} -> {args.out_dir}/train.json")
    print(f"   Valid: {len(valid)} -> {args.out_dir}/valid.json")
    if n_test > 0:
        print(f"   Test : {len(test)} -> {args.out_dir}/test.json")

if __name__ == "__main__":
    main()
