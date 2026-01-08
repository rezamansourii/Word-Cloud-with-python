# Persian RTL Word Cloud Generator

Streamlit-based UI for generating Persian (RTL) word clouds from URLs, PDFs, or plain text. The app handles Arabic/Persian shaping and proper RTL display so the output renders correctly.

<table>
  <tr>
    <td align="center">
      <img
        src="https://raw.githubusercontent.com/rezamansourii/Word-Cloud-with-python/main/Images/Word-cloud-generator.png"
        width="300"
        alt="Sample Output 1"
      />
    </td>
    <td align="center">
      <img
        src="https://raw.githubusercontent.com/rezamansourii/Word-Cloud-with-python/main/Images/word-cloud-generator2.png"
        width="300"
        alt="Sample Output 2"
      />
    </td>
  </tr>
  <tr>
    <td align="center">
      <img
        src="https://raw.githubusercontent.com/rezamansourii/Word-Cloud-with-python/main/Images/Sample-output.png"
        width="300"
        alt="Word Cloud Generator UI 1"
      />
    </td>
    <td align="center">
      <img
        src="https://raw.githubusercontent.com/rezamansourii/Word-Cloud-with-python/main/Images/Sample-output2.png"
        width="300"
        alt="Word Cloud Generator UI 2"
      />
    </td>
  </tr>
</table>

## Features

- **Inputs:** URL, PDF, or plain text
- **RTL support:** Arabic/Persian reshaping + bidi handling
- **Customizable:** stopwords, font, size, background, and more
- **Download:** export word clouds as PNG

## Requirements

- Python 3.9+
- A Persian-capable font file (TTF/OTF), e.g. Vazirmatn, Noto Naskh Arabic

## Setup

### Conda

```bash
conda create -n persian-wordcloud python=3.9
conda activate persian-wordcloud
conda install -c conda-forge --file requirements.txt
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run word-cloud.py
```

## Notes

- You must provide a valid Persian-capable font path or upload a font in the UI.
- WordCloud itself is not RTL-aware; this app reshapes text and applies bidi to render correctly.

## License

MIT. See [LICENSE](LICENSE).
