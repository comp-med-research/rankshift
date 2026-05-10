"""Canonical display names for model labels used in plots."""

MODEL_DISPLAY_NAMES = {
    "chandra2": "Chandra 2",
    "chatgpt_api": "GPT-5.5",
    "deepseek_ocr_2": "DeepSeek-OCR 2",
    "docling_ocr": "Docling",
    "dolphin_1_5": "Dolphin 1.5",
    "dotsocr": "dots.ocr",
    "glmocr": "GLMOCR",
    "glm_ocr": "GLMOCR",
    "got_ocr2": "GOT-OCR2.0",
    "hunyuanocr": "Hunyuan OCR",
    "mineru_1_2b": "MinerU2.5",
    "monkeyocr_pro_3b": "MonkeyOCR-pro-3B",
    "paddleocrVL_1_5": "PaddleOCR-VL-1.5",
    "paddleocr_vl_1_5": "PaddleOCR-VL-1.5",
    "paddleocr_vl_1_5_vllm": "PaddleOCR-VL-1.5",
    "rolmocr": "RolmOCR",
    "youtu": "Youtu-Parsing",
}


def display_model_name(model: object) -> str:
    """Return a publication-friendly model name without changing raw CSV IDs."""
    text = str(model)
    return MODEL_DISPLAY_NAMES.get(text, text)


def add_model_display_column(df, source: str = "model_name", target: str = "model_display"):
    """Return a copy of df with a display-name column for plotting."""
    out = df.copy()
    out[target] = out[source].map(display_model_name)
    return out
