"""XAI image lookup helpers for the demo."""

from __future__ import annotations

from demo.settings import XAI_MAPS


def get_xai_images(case: str) -> tuple[str | None, str | None]:
    folder = XAI_MAPS.get(case)
    if not folder or not folder.is_dir():
        return None, None
    lime = folder / "lime_importance_final.png"
    shap = folder / "shap_importance_final.png"
    return (str(lime) if lime.exists() else None, str(shap) if shap.exists() else None)
