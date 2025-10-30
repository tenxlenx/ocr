# TODO

## What This Application Does

This is a **screen-to-text OCR tool** that lets you:

1. **Select a rectangular region** on your screen by clicking and dragging
2. **Automatically capture** that region as a screenshot
3. **Perform OCR** (Optical Character Recognition) on the selected area using the DeepSeek-OCR model
4. **Copy the extracted text** directly to your clipboard

**Primary Use Case:** Copying text from images, PDFs, videos, or any non-selectable text on your screen without manually typing it out.

## How It Works

### Client (`ocr.py`)
- Run `ocr` (no arguments) to launch an interactive screen selector (assumes `ocr.py` is installed as `ocr` in your PATH)
- Click and drag to select the region containing text
- The tool captures the selected area, sends it to the OCR API, and copies the recognized text to your clipboard
- Alternatively, pass an image file path: `ocr image.png` to OCR a saved image
- If not installed, run directly: `python3 ocr.py` or `./ocr.py` (after `chmod +x ocr.py`)

### Server (`ocr_api_allinone.py`)
- FastAPI server that receives images via `POST /infer`
- Uses the **DeepSeek-OCR** model (quantized 4-bit version) for text recognition
- Streams the OCR results as plain text in real-time
- Configurable via `.env` file for model settings, resolution presets, and prompts

## Example Workflow

```bash
# Interactive screen selection (if installed in PATH)
ocr
# (Select region with mouse, text appears on screen and in clipboard)

# Or process a saved image
ocr document.png

# If not installed, use:
python3 ocr.py
python3 ocr.py document.png
```

## Future Improvements

- [ ] Add support for multiple language detection and switching
- [ ] Implement batch processing for multiple images
- [ ] Add a configuration file for client-side settings
- [ ] Support output to file in addition to clipboard
- [ ] Add keyboard shortcuts for common operations
- [ ] Implement a system tray icon for quick access
- [ ] Add OCR result history/cache
- [ ] Support for different output formats (JSON, markdown, plain text)
- [ ] Implement confidence scoring display
- [ ] Add preprocessing filters (deskew, denoise, contrast enhancement)
