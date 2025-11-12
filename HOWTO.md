# MemoBar Documentation

MemoBar is a macOS menubar application for vocabulary learning that displays foreign words with their translations. This guide explains how to create and configure custom vocabulary datasets.

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Configuration](#dataset-configuration)
- [Word Dataset Formats](#word-dataset-formats)
- [Audio Support](#audio-support)
- [Repository Structure](#repository-structure)
- [Examples](#examples)

## Quick Start

1. **Using Remote Datasets**: Open MemoBar Settings and select from available remote datasets
2. **Custom URL**: Enter a GitHub raw URL pointing to your word list (JSON or INI format)
3. **Audio Files**: Add audio files with proper configuration in audio/config.json

## Dataset Configuration

### config.json Format

The `config.json` file defines available datasets and their metadata:

```json
[
  {
    "file": "basic-spanish-to-polish.txt",
    "format": "ini",
    "category": "basic",
    "description": "entry level vocabulary",
    "level": "A1",
    "language-in": "es",
    "language-out": "pl",
    "version": "1.0"
  }
]
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Filename of the word dataset |
| `format` | string | Either "ini" or "json" |
| `category` | string | Category grouping (e.g., "basic", "advanced") |
| `description` | string | Human-readable description |
| `level` | string | Difficulty level (e.g., "A1", "B2", "intermediate") |
| `language-in` | string | Source language code (e.g., "es", "en", "de") |
| `language-out` | string | Target language code (e.g., "pl", "en", "fr") |
| `version` | string | Dataset version number |

#### Language Codes

Use standard ISO 639-1 language codes:
- `en` - English
- `es` - Spanish
- `de` - German
- `fr` - French
- `pl` - Polish
- `it` - Italian
- `pt` - Portuguese

## Word Dataset Formats

MemoBar supports two formats for word datasets: **INI Format** and **JSON Format**.

### INI Format (.txt files)

The INI format uses a simple text structure with sections and key-value pairs:

#### Structure
```ini
[section/category]
    ID = foreign_word :: translation
    ID = foreign_word :: translation
    ...
```

#### Example: basic-spanish-to-polish.txt
```ini
[spanish/basic]
    1 = ¡Buenos días! :: Dzień dobry!
    2 = ¡Buenas tardes! :: Dzień dobry! (po południu)
    3 = ¡Buenas noches! :: Dobry wieczór!
    4 = ¡Hola! :: Cześć!
    5 = Vivo en… :: Mieszkam w…
    6 = ¿Cómo estás? :: Jak się masz?
```

#### INI Format Rules
- **Section headers**: `[category/subcategory]` (optional but recommended)
- **Entry format**: `ID = foreign_word :: translation`
- **ID**: Sequential number starting from 1
- **Separator**: ` :: ` (space-double-colon-space)
- **Empty lines**: Ignored
- **Comments**: Lines starting with `;` or `#` are ignored

### JSON Format (.json files)

The JSON format provides more structured data with support for additional metadata:

#### Structure Option 1: Object with words array
```json
{
  "words": [
    {
      "id": 1,
      "foreign": "foreign_word",
      "translation": "translation"
    }
  ]
}
```

#### Structure Option 2: Direct array
```json
[
  {
    "id": 1,
    "foreign": "foreign_word",
    "translation": "translation"
  }
]
```

#### Example: vocabulary.json
```json
{
  "words": [
    {
      "id": 1,
      "foreign": "Hola",
      "translation": "Hello"
    },
    {
      "id": 2,
      "foreign": "Adiós",
      "translation": "Goodbye"
    },
    {
      "id": 3,
      "foreign": "Por favor",
      "translation": "Please"
    }
  ]
}
```

#### JSON Format Rules
- **Required fields**: `id`, `foreign`, `translation`
- **ID**: Unique numeric identifier
- **foreign**: Source language word/phrase
- **translation**: Target language word/phrase
- **Encoding**: UTF-8

## Audio Support

MemoBar supports audio files for pronunciation with a sophisticated organization system.

### Audio Directory Structure

Audio files are organized in letter-based subdirectories within the `audio/` folder:

```
audio/
├── config.json          # Audio configuration file
├── C/                   # Files starting with 'C'
│   ├── Como_estas.mp3
│   └── Cuanto_es.mp3
├── H/                   # Files starting with 'H'
│   └── Hazme_un_favor.mp3
├── M/                   # Files starting with 'M'
│   ├── Me_gusta.mp3
│   ├── Me_traes_el_menu_por_favor.mp3
│   └── Muchas_gracias.mp3
└── P/                   # Files starting with 'P'
    ├── Por_favor.mp3
    └── Puedes_repetir.mp3
```

### Audio Configuration (audio/config.json)

The `audio/config.json` file maps text phrases to corresponding audio files for any language:

```json
[
    {
        "text": "¿Cómo estás?",
        "file": "C/Como_estas.mp3",
        "language": "es",
        "timestamp": "2025-11-09T14:30:00"
    },
    {
        "text": "Guten Morgen",
        "file": "G/Guten_Morgen.mp3",
        "language": "de",
        "timestamp": "2025-11-09T14:30:00"
    },
    {
        "text": "Bonjour",
        "file": "B/Bonjour.mp3",
        "language": "fr",
        "timestamp": "2025-11-09T14:30:00"
    }
]
```

#### Audio Config Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Exact text from the word dataset that has audio |
| `file` | string | Relative path to audio file (letter/filename.ext) |
| `language` | string | Language code (e.g., "es", "en", "de") |
| `timestamp` | string | ISO 8601 timestamp when audio was added |

### Supported Audio Formats
- **MP3** - Good compatibility and compression
- **M4A (AAC)** - Recommended for best balance of quality and file size
- **CAF** - Native macOS format
- **WAV** - Uncompressed (larger files)

### Audio File Naming Convention

#### Universal Character Normalization Rules
- Files are organized by first letter of the text (ignoring punctuation)
- Spaces in text become underscores in filenames
- **All accented and special characters are simplified/removed**
- **Universal for ALL languages** - Spanish, German, French, Polish, Italian, Portuguese, etc.

#### Character Simplification Examples
| Language | Original Character | Simplified |
|----------|-------------------|------------|
| Spanish | á, é, í, ó, ú | a, e, i, o, u |
| German | ä, ö, ü, ß | a, o, u, ss |
| French | à, è, é, ê, ç | a, e, e, e, c |
| Polish | ą, ć, ę, ł, ń, ó, ś, ź, ż | a, c, e, l, n, o, s, z, z |
| Portuguese | ã, õ, ç | a, o, c |

#### Filename Examples by Language
- **Spanish**: `"¿Cómo estás?"` → `C/Como_estas.mp3`
- **German**: `"Guten Morgen"` → `G/Guten_Morgen.mp3`
- **French**: `"Où est la gare?"` → `O/Ou_est_la_gare.mp3`
- **Polish**: `"Dzień dobry"` → `D/Dzien_dobry.mp3`
- **Portuguese**: `"Obrigação"` → `O/Obrigacao.mp3`
- **Italian**: `"Buongiorno"` → `B/Buongiorno.mp3`

### Audio Integration
- Audio files are linked to text phrases via the audio/config.json mapping
- MemoBar displays a speaker icon when audio is available for a phrase
- Click the speaker icon to play pronunciation
- Audio matching is done by exact text comparison

## Repository Structure

```
your-repo/
├── config.json                    # Dataset configuration
├── HOWTO.md                       # This documentation
├── basic-spanish-to-polish.txt    # INI format dataset
├── advanced-vocabulary.json       # JSON format dataset
└── audio/                         # Audio files directory
    ├── config.json                # Audio configuration mapping
    ├── C/                         # Files starting with 'C'
    │   ├── Como_estas.mp3
    │   └── Cuanto_es.mp3
    ├── H/
    │   └── Hazme_un_favor.mp3
    ├── M/
    │   ├── Me_gusta.mp3
    │   ├── Me_traes_el_menu_por_favor.mp3
    │   └── Muchas_gracias.mp3
    └── P/
        ├── Por_favor.mp3
        └── Puedes_repetir.mp3
```

## Examples

### Example 1: Basic Spanish-Polish Dataset

**config.json**:
```json
[
  {
    "file": "basic-spanish-to-polish.txt",
    "format": "ini",
    "category": "basic",
    "description": "Basic Spanish to Polish vocabulary",
    "level": "A1",
    "language-in": "es",
    "language-out": "pl",
    "version": "1.0"
  }
]
```

**basic-spanish-to-polish.txt**:
```ini
[spanish/basic]
    1 = ¡Hola! :: Cześć!
    2 = ¡Adiós! :: Do widzenia!
    3 = Por favor :: Proszę
    4 = Gracias :: Dziękuję
```

**audio/config.json** (accented characters simplified):
```json
[
    {
        "text": "¡Hola!",
        "file": "H/Hola.mp3",
        "language": "es",
        "timestamp": "2025-11-09T14:30:00"
    },
    {
        "text": "Por favor",
        "file": "P/Por_favor.mp3",
        "language": "es",
        "timestamp": "2025-11-09T14:30:00"
    }
]
```

### Example 2: Advanced English-German JSON Dataset

**config.json**:
```json
[
  {
    "file": "advanced-english-german.json",
    "format": "json",
    "category": "advanced",
    "description": "Advanced English to German vocabulary",
    "level": "C1",
    "language-in": "en",
    "language-out": "de",
    "version": "2.0"
  }
]
```

**advanced-english-german.json**:
```json
{
  "words": [
    {
      "id": 1,
      "foreign": "serendipity",
      "translation": "der glückliche Zufall"
    },
    {
      "id": 2,
      "foreign": "ubiquitous",
      "translation": "allgegenwärtig"
    }
  ]
}
```

### Example 3: Using Custom GitHub Repository

1. Create a GitHub repository with your datasets
2. Add your `config.json` and word files
3. Add audio files in the proper directory structure
4. Get the raw URL for your files:
   - Repository: `https://github.com/username/my-vocab`
   - Raw config URL: `https://raw.githubusercontent.com/username/my-vocab/main/config.json`
   - Raw dataset URL: `https://raw.githubusercontent.com/username/my-vocab/main/my-words.json`

5. In MemoBar Settings:
   - For remote datasets: MemoBar will automatically load from the config.json URL
   - For custom URL: Enter the direct raw URL to your word file

## Best Practices

### Dataset Creation
1. **Use descriptive IDs**: Start from 1 and increment sequentially
2. **Consistent formatting**: Maintain consistent punctuation and spacing
3. **Version control**: Update version numbers when modifying datasets
4. **UTF-8 encoding**: Ensure all files are saved with UTF-8 encoding
5. **Test locally**: Validate JSON format before publishing

### Audio Files
1. **Universal naming**: Apply character normalization rules for ANY language
2. **Letter directories**: Organize files by first letter for easy browsing
3. **Config mapping**: Always update audio/config.json when adding files
4. **Text matching**: Ensure exact text match between dataset and audio config
5. **Character simplification**: Remove ALL accents and special characters (á→a, ü→u, ç→c, etc.)
6. **Small file sizes**: Consider MP3 or M4A format for optimal size
7. **Quality**: Use clear pronunciation recordings

## Troubleshooting

### Common Issues

**"No words loaded"**:
- Check file format (JSON must be valid)
- Verify raw URL is accessible
- Ensure proper UTF-8 encoding

**"Audio not playing"**:
- Check audio file exists in correct letter subdirectory
- Verify audio/config.json has correct mapping
- Ensure text matches exactly between dataset and audio config
- Use supported audio formats (MP3, M4A, CAF, WAV)

**"Dataset not appearing"**:
- Verify `config.json` format is correct
- Check all required fields are present
- Ensure `format` field matches actual file format

**"Audio not linking to words"**:
- Verify text in audio/config.json exactly matches text in word dataset
- Check letter subdirectory organization
- Ensure file paths in audio config are correct

### Validation

Before publishing your dataset:

1. **JSON Validation**: Use online JSON validators for JSON format files
2. **File Access**: Test raw URLs in browser to ensure they're accessible
3. **Character Encoding**: Verify files display correctly with special characters
4. **Audio Testing**: Check audio files play correctly on macOS
5. **Text Matching**: Verify audio config text exactly matches dataset entries
6. **Directory Structure**: Confirm letter-based organization in audio/

---

For support, bug reports, or feature requests, visit the [MemoBar GitHub repository](https://github.com/ppiecuch/memobar-public).