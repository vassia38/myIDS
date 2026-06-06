from pathlib import Path
import joblib
import m2cgen as m2c

SRC_DIR = Path(__file__).resolve().parent
MODELS_DIR = SRC_DIR.parent / "models"


def get_available_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.sav"))


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_choice = "Y/n" if default else "y/N"
    choice = input(f"{prompt} [{default_choice}] ").strip().lower()
    if choice == "":
        return default
    return choice in {"y", "yes"}


def prompt_model_path() -> Path:
    raw_path = input("Enter the path to the model file to transpile: ").strip()
    if not raw_path:
        raise ValueError("No model path provided.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = SRC_DIR / path
    return path.resolve()


def transpile_model(model_path: Path):
    print(f"Loading model {model_path}...")
    model = joblib.load(model_path)
    print(f"Transpiling {model_path.name} to Go...")
    go_code = m2c.export_to_go(model)
    transpiled_dir = MODELS_DIR / "transpiled"
    transpiled_dir.mkdir(parents=True, exist_ok=True)
    output_file = transpiled_dir / f"{model_path.stem}_model.go"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("package main\n\n")
        f.write(go_code)
    print(f"Successfully exported {output_file}")


def main():
    available_models = get_available_models()
    if available_models:
        print(f"Found {len(available_models)} model(s) in {MODELS_DIR}")
        for i, model_path in enumerate(available_models, start=1):
            print(f"  {i}. {model_path.name}")

        if prompt_yes_no("Transpile all models found in ../models/?", default=True):
            selected_models = available_models
        else:
            selected_models = [prompt_model_path()]
    else:
        print(f"No models found in {MODELS_DIR}.")
        selected_models = [prompt_model_path()]

    for model_path in selected_models:
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        transpile_model(model_path)


if __name__ == "__main__":
    main()
