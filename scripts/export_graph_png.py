from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import load_runtime_config
from app.graph import build_graph


def main() -> None:
    output_path = Path("graph.png")
    load_runtime_config(
        config_path=ROOT_DIR / "config.yml",
        env_path=ROOT_DIR / ".env",
    )
    graph = build_graph()
    png_bytes = graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_bytes)
    print(f"Graph PNG saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
