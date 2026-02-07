# code2_cli_multi.py
import os
import re
import time
import sys

def mostrar_banner():
    """Muestra el banner ASCII art de CODE-2"""
    banner = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ██████╗ ██████╗ ███████╗    ██████╗                           ║
║  ██╔════╝██╔═══██╗██╔══██╗██╔════╝    ╚════██╗                          ║
║  ██║     ██║   ██║██║  ██║█████╗█████╗ █████╔╝                          ║
║  ██║     ██║   ██║██║  ██║██╔══╝╚════╝██╔═══╝                           ║
║  ╚██████╗╚██████╔╝██████╔╝███████╗    ███████╗                          ║
║   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚══════╝                          ║
║                                                                           ║
║              Natural Language to Assembly Code Generator                  ║
║                         Version 1.0.0 - 2025                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def leer_tecla():
    """Lee una tecla del teclado de forma no bloqueante (Windows)"""
    if sys.platform == 'win32':
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\xe0':  # Teclas especiales
                key = msvcrt.getch()
                if key == b'H':  # Flecha arriba
                    return 'up'
                elif key == b'P':  # Flecha abajo
                    return 'down'
            elif key == b'\r':  # Enter
                return 'enter'
            elif key == b'\x1b':  # ESC
                return 'esc'
    return None


def limpiar_salida(texto: str) -> str:
    # Convierte '\n' literal a salto real
    texto = texto.replace("\\n", "\n")

    # Quita tokens especiales típicos si aparecen en texto
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)

    # Quita espacios antes/después del salto de línea
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)

    # Normaliza espacios múltiples
    texto = re.sub(r" +", " ", texto)

    # Quita indentación al principio de cada línea
    lineas = texto.splitlines()
    lineas = [ln.lstrip() for ln in lineas]

    return "\n".join(lineas).strip()


def leer_entrada_multilinea() -> str:
    print("\n╭─────────────────────────────────────────────────────────────────────╮")
    print("│  📝 Instrucción en Lenguaje Natural                                │")
    print("│  (Presiona ENTER en línea vacía para ejecutar, o 'salir' para terminar) │")
    print("╰─────────────────────────────────────────────────────────────────────╯")
    print("\n> ", end="")
    lineas = []
    while True:
        linea = input()  # Leer línea del usuario
        if linea.lower() == "salir":
            return "__SALIR__"
        if linea == "":  # Si la línea está vacía, salir
            break
        lineas.append(linea)
        if linea:  # Si hay más texto, mostrar el prompt
            print("> ", end="")
    return "\n".join(lineas).strip()


def elegir_modelo(base_dir: str):
    opciones = [
        {
            "nombre": "Full Fine-Tuning (PyTorch FP32)",
            "descripcion": "Modelo completo en precisión FP32 - Mayor precisión",
            "ruta": "models/full_fp32",
            "tipo": "pytorch"
        },
        {
            "nombre": "ONNX INT8 Dynamic Quantization",
            "descripcion": "Modelo optimizado INT8 - Mayor velocidad y eficiencia",
            "ruta": "models/onnx_int8_dynamic",
            "tipo": "onnx"
        }
    ]
    
    seleccion = 0
    
    def mostrar_menu():
        os.system('cls' if os.name == 'nt' else 'clear')
        mostrar_banner()
        print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║                    SELECCIÓN DE MODELO Y CONFIGURACIÓN                   ║")
        print("╚═══════════════════════════════════════════════════════════════════════════╝\n")
        print("  Use las flechas ↑↓ para navegar y ENTER para seleccionar:\n")
        
        for i, opcion in enumerate(opciones):
            if i == seleccion:
                print(f"  ► {i+1}. {opcion['nombre']}")
                print(f"      {opcion['descripcion']}")
            else:
                print(f"    {i+1}. {opcion['nombre']}")
                print(f"      {opcion['descripcion']}")
        print()
    
    # Mostrar menú inicial
    mostrar_menu()
    
    import msvcrt
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\xe0':  # Teclas especiales
                key = msvcrt.getch()
                if key == b'H':  # Flecha arriba
                    seleccion = (seleccion - 1) % len(opciones)
                    mostrar_menu()
                elif key == b'P':  # Flecha abajo
                    seleccion = (seleccion + 1) % len(opciones)
                    mostrar_menu()
            elif key == b'\r':  # Enter
                break
            elif key in [b'1', b'2']:
                seleccion = int(key) - 49  # ASCII '1' = 49
                break
    
    opcion_elegida = opciones[seleccion]
    print(f"\n  ✓ Modelo seleccionado: {opcion_elegida['nombre']}\n")

    torch_path = os.path.join(base_dir, "../models/full_fp32")
    onnx_path  = os.path.join(base_dir, "../models/onnx_int8_dynamic")

    if opcion_elegida['tipo'] == "pytorch":
        # PyTorch
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        if not os.path.isdir(torch_path):
            raise FileNotFoundError(f"No encuentro la carpeta del modelo PyTorch: {torch_path}")

        tokenizer = T5Tokenizer.from_pretrained(torch_path)

        # Si por lo que sea no está guardado el token especial, lo añadimos (no pasa nada si ya está)
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = T5ForConditionalGeneration.from_pretrained(torch_path).to(device)
        model.eval()

        def generar(texto_entrada: str):
            inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,        # precisión > creatividad
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            t1 = time.perf_counter()

            bruto = tokenizer.decode(out_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return limpiar_salida(bruto), (t1 - t0)

        return "PyTorch FP32 (Full FT)", generar

    else:
        # ONNX INT8 dynamic
        from transformers import T5Tokenizer
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        if not os.path.isdir(onnx_path):
            raise FileNotFoundError(f"No encuentro la carpeta del modelo ONNX INT8: {onnx_path}")

        tokenizer = T5Tokenizer.from_pretrained(onnx_path)

        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
        except Exception:
            pass

        model = ORTModelForSeq2SeqLM.from_pretrained(onnx_path)

        def generar(texto_entrada: str):
            inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)

            t0 = time.perf_counter()
            out_ids = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                do_sample=False,
            )
            t1 = time.perf_counter()

            bruto = tokenizer.decode(out_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return limpiar_salida(bruto), (t1 - t0)

        return "ONNX INT8 dynamic (CPU)", generar


def main():
    # Limpiar pantalla
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Mostrar banner
    mostrar_banner()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nombre_modelo, generar = elegir_modelo(base_dir)

    print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                           GENERADOR ACTIVO                                ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    while True:
        entrada = leer_entrada_multilinea()
        if entrada == "__SALIR__":
            print("\n╭─────────────────────────────────────────────────────────────────────╮")
            print("│  👋 Gracias por usar CODE-2. ¡Hasta pronto!                        │")
            print("╰─────────────────────────────────────────────────────────────────────╯\n")
            break
        if not entrada:
            print("\n  ⚠️  Entrada vacía. Por favor, introduce una instrucción.\n")
            continue

        print("\n  ⚙️  Generando código ensamblador...\n")
        salida, dt = generar(entrada)
        
        print("╭─────────────────────────────────────────────────────────────────────╮")
        print("│  📦 CÓDIGO ENSAMBLADOR GENERADO                                    │")
        print("╰─────────────────────────────────────────────────────────────────────╯\n")
        print(salida)
        print(f"\n╭─────────────────────────────────────────────────────────────────────╮")
        print(f"│  ⏱️  Tiempo de inferencia: {dt:.4f} segundos                        ")
        print("╰─────────────────────────────────────────────────────────────────────╯")


if __name__ == "__main__":
    main()
