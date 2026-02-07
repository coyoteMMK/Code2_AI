# scripts/tokenizer_demo.py
import argparse
from transformers import T5Tokenizer


def main():
    ap = argparse.ArgumentParser(
        description="Visualiza cómo tokeniza T5 el texto de entrada"
    )
    
    ap.add_argument(
        "--model_path",
        default="models/full_fp32",
        help="Ruta al modelo/tokenizador (por defecto: models/full_fp32)"
    )
    
    ap.add_argument(
        "--text",
        type=str,
        required=True,
        help="Texto a tokenizar"
    )
    
    ap.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Longitud máxima (opcional, sin truncamiento por defecto)"
    )
    
    ap.add_argument(
        "--show_special",
        action="store_true",
        help="Mostrar tokens especiales en la salida"
    )

    ap.add_argument(
        "--simple",
        action="store_true",
        help="Salida simple: muestra solo la lista de tokens en una linea"
    )

    ap.add_argument(
        "--preserve_newlines",
        action="store_true",
        help="Preserva saltos de linea reemplazandolos por SALTO antes de tokenizar"
    )
    
    args = ap.parse_args()
    
    print(f"Cargando tokenizador desde: {args.model_path}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_path, legacy=False)
    
    print(f"\nTexto de entrada:\n{args.text}\n")

    text_for_tokenize = args.text
    if args.preserve_newlines:
        text_for_tokenize = text_for_tokenize.replace("\n", " SALTO ")
    
    # Tokenizar
    encoded = tokenizer(
        text_for_tokenize,
        max_length=args.max_length,
        truncation=args.max_length is not None,
        add_special_tokens=args.show_special,
    )
    
    # Obtener información
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    if args.simple:
        print(f"Tokens: {len(input_ids)}  Characters: {len(args.text)}")
        print(" ".join(tokens))
        decoded_simple = tokenizer.decode(
            input_ids, skip_special_tokens=not args.show_special
        )
        if args.preserve_newlines:
            decoded_simple = decoded_simple.replace("SALTO", "\n")
        print(f"Decoded: {decoded_simple}")
        return
    
    # Mostrar resultados
    print("=" * 60)
    print("RESULTADOS DE TOKENIZACIÓN")
    print("=" * 60)
    
    print(f"\nNúmero de tokens: {len(input_ids)}")
    
    print(f"\nIDs de tokens:")
    print(input_ids)
    
    print(f"\nTokens:")
    for i, (token, token_id) in enumerate(zip(tokens, input_ids)):
        print(f"  [{i:3d}] {token_id:6d} → '{token}'")
    
    print(f"\nTexto decodificado:")
    decoded = tokenizer.decode(input_ids, skip_special_tokens=not args.show_special)
    if args.preserve_newlines:
        decoded = decoded.replace("SALTO", "\n")
    print(f"{decoded}")

    # Información adicional del tokenizador
    print("\n" + "=" * 60)
    print("INFORMACIÓN DEL TOKENIZADOR")
    print("=" * 60)
    print(f"Tamaño del vocabulario: {len(tokenizer)}")
    print(f"Modelo: {tokenizer.name_or_path}")
    
    if args.show_special:
        print(f"\nTokens especiales:")
        print(f"  PAD: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
        print(f"  EOS: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
        print(f"  UNK: '{tokenizer.unk_token}' (id: {tokenizer.unk_token_id})")
        
        if tokenizer.additional_special_tokens:
            print(f"  Adicionales: {tokenizer.additional_special_tokens}")


if __name__ == "__main__":
    main()
