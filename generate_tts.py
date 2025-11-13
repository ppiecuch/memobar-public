#!/usr/bin/env python3
"""
MemoBar TTS Generator
====================
Pure Python script to generate TTS audio files for vocabulary learning.
Based on Google Translate TTS API, with rate limiting and caching.

Usage:
    python3 generate_tts.py [--config config.json] [--cache-dir audio] [--dry-run]

Features:
- Reads config.json to discover vocabulary files
- Parses INI and JSON format vocabulary files
- Downloads TTS audio from Google Translate
- Implements rate limiting to respect Google's limits
- Caches downloaded audio files
- Supports multiple languages
- Pure Python implementation (no external dependencies)

Text Preprocessing:
- Removes square bracket annotations: "SER [być]" → "SER"
- Cleans up unused MP3 files automatically

Commands:
- --test: Run internal tests for preprocessing functionality
- --cleanup DAYS: Remove cache files older than specified days
- --cleanup-unused: Remove MP3 files not used by current dataset
"""

# =============================================================================
# CONFIGURATION PARAMETERS - Modify these settings as needed
# =============================================================================

# Cache Management
CACHE_EXPIRATION_DAYS = 30          # Don't replace files newer than this (days)
MIN_AUDIO_FILE_SIZE = 1000          # Minimum valid audio file size (bytes)
MAX_AUDIO_FILE_SIZE = 500000        # Maximum valid audio file size (bytes)

# Rate Limiting
DEFAULT_DELAY_SECONDS = 5.0         # Default delay between requests (seconds)
EXPONENTIAL_BACKOFF_MULTIPLIER = 2  # Multiplier when rate limited (429 error)

# Text Processing
MAX_TEXT_LENGTH = 200               # Maximum text length for single TTS request
FILENAME_MAX_LENGTH = 50            # Maximum length for sanitized filenames

# Google TTS API Settings
GOOGLE_TTS_BASE_URL = "https://translate.google.com/translate_tts"
USER_AGENT = "stagefright/1.2 (Linux;Android 9.0)"
REFERER = "http://translate.google.com/"
REQUEST_TIMEOUT = 30                # HTTP request timeout (seconds)

# File Organization
AUDIO_CACHE_DIR = "audio"           # Default cache directory name
CHUNK_DELAY_SECONDS = 0.5           # Delay between chunks for multi-part audio

# =============================================================================
# END CONFIGURATION - Do not modify below this line unless you know what you're doing
# =============================================================================

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import hashlib
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def simplify_diacritics(text: str) -> str:
    """
    Simplify diacritics and accented characters to their base ASCII equivalents.

    Examples:
    - á, à, ä, ã → a
    - é, è, ê, ë → e
    - ñ → n
    - ç → c
    - ü, ù, û, ú → u
    """
    # Normalize to decomposed form (NFD) - separates base character from combining marks
    normalized = unicodedata.normalize('NFD', text)

    # Remove all combining diacritical marks
    ascii_text = ''.join(char for char in normalized
                        if unicodedata.category(char) != 'Mn')

    # Additional manual replacements for common cases not covered by NFD
    replacements = {
        'ø': 'o', 'œ': 'oe', 'æ': 'ae', 'ß': 'ss',
        'ł': 'l', 'đ': 'd', 'ħ': 'h', 'ŧ': 't',
        'ð': 'd', 'þ': 'th', 'ŋ': 'ng'
    }

    for accented, base in replacements.items():
        ascii_text = ascii_text.replace(accented, base)
        ascii_text = ascii_text.replace(accented.upper(), base.upper())

    return ascii_text


def preprocess_tts_text(text: str) -> str:
    """
    Preprocess text for TTS generation by cleaning annotations.

    Rules:
    1. Remove square bracket annotations: "SER [być]" → "SER"

    Args:
        text: Input text to preprocess

    Returns:
        Cleaned text ready for TTS
    """
    # Remove square bracket annotations [annotation]
    cleaned_text = re.sub(r'\s*\[[^\]]+\]', '', text)

    # Clean up any extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def sanitize_filename(text: str, max_length: int = FILENAME_MAX_LENGTH) -> str:
    """
    Sanitize text to create valid filename with only ASCII chars and underscores.

    Rules:
    - Simplify diacritics (é → e, ñ → n, etc.)
    - Only ASCII characters and underscores allowed
    - Multiple underscores converted to single underscore
    - Leading and trailing underscores removed
    - Limit length to max_length characters
    """
    # First simplify diacritics using our custom function
    simplified = simplify_diacritics(text)

    # Ensure we only have ASCII characters
    ascii_text = simplified.encode('ascii', 'ignore').decode('ascii')

    # Replace any non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', ascii_text)

    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    # Ensure we have at least something
    if not sanitized:
        sanitized = 'audio'

    return sanitized


class TTSConfig:
    """Configuration for TTS generation"""

    def __init__(self):
        self.base_url = GOOGLE_TTS_BASE_URL
        self.user_agent = USER_AGENT
        self.referer = REFERER
        self.max_text_length = MAX_TEXT_LENGTH
        self.rate_limit_delay = DEFAULT_DELAY_SECONDS
        self.cache_dir = AUDIO_CACHE_DIR
        self.dry_run = False


class TTSRateLimiter:
    """Rate limiter to respect Google's API limits"""

    def __init__(self, delay: float = DEFAULT_DELAY_SECONDS):
        self.delay = delay
        self.last_request_time = 0.0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            print(f"🕐 Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class TTSCache:
    """Manages caching of downloaded TTS files"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, text: str, language: str) -> Path:
        """Generate cache file path for text and language using sanitized filename in alphabetical subfolders"""
        # Create sanitized filename from text
        sanitized_text = sanitize_filename(text)  # Clean filename without language prefix
        filename = f"{sanitized_text}.mp3"

        # Get first letter for subfolder organization
        first_letter = sanitized_text[0].upper() if sanitized_text else 'Z'
        # Ensure it's A-Z, fallback to 'Z' for numbers or special chars
        if not first_letter.isalpha():
            first_letter = 'Z'

        # Create subfolder path
        subfolder = self.cache_dir / first_letter
        subfolder.mkdir(exist_ok=True)

        # Handle potential filename collisions
        cache_path = subfolder / filename
        if cache_path.exists():
            # Check if existing file is likely for the same content
            # by comparing text hash - if same text, reuse the existing file
            expected_hash = hashlib.md5(f"{language}:{text}".encode('utf-8')).hexdigest()[:8]

            # Check file age and validity before reusing
            try:
                file_stat = cache_path.stat()
                file_size = file_stat.st_size
                file_age_days = (time.time() - file_stat.st_mtime) / (24 * 3600)

                # Only reuse file if it's valid size and less than specified days old
                if MIN_AUDIO_FILE_SIZE <= file_size <= MAX_AUDIO_FILE_SIZE and file_age_days < CACHE_EXPIRATION_DAYS:
                    # File exists, is valid, and recent - reuse it
                    pass  # Keep existing path
                else:
                    # File exists but is too old or invalid - create new with hash
                    content_hash = expected_hash
                    filename = f"{sanitized_text}_{content_hash}.mp3"
                    cache_path = subfolder / filename
            except OSError:
                # File exists but can't read stats - create new with hash
                content_hash = expected_hash
                filename = f"{sanitized_text}_{content_hash}.mp3"
                cache_path = subfolder / filename

        return cache_path

    def get_filename_only(self, text: str, language: str) -> str:
        """Get just the filename (without path) for storing in config"""
        cache_path = self.get_cache_path(text, language)
        return cache_path.name

    def is_cached(self, text: str, language: str) -> bool:
        """Check if TTS audio is already cached"""
        cache_path = self.get_cache_path(text, language)
        return cache_path.exists() and cache_path.stat().st_size > 0

    def cleanup_old_files(self, max_age_days: int = 30):
        """Remove cached files older than specified days"""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        removed_count = 0
        for file_path in self.cache_dir.glob("*.mp3"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                removed_count += 1

        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count} old cache files")

    def cleanup_unused_files(self, expected_filenames: set):
        """Remove MP3 files that don't correspond to any entries in the dataset"""
        removed_count = 0
        total_files = 0

        # Recursively find all MP3 files in cache directory and subdirectories
        for file_path in self.cache_dir.rglob("*.mp3"):
            total_files += 1
            if file_path.name not in expected_filenames:
                try:
                    file_path.unlink()
                    removed_count += 1
                    print(f"🗑️ Removed unused: {file_path.name}")
                except OSError as e:
                    print(f"⚠️ Warning: Could not remove {file_path.name}: {e}")

        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count}/{total_files} unused MP3 files")
        else:
            print(f"✅ No unused files found ({total_files} MP3 files checked)")


class VocabularyEntry:
    """Represents a single vocabulary entry"""

    def __init__(self, entry_id: int, foreign: str, translation: str,
                 language_in: str, language_out: str, section: str = ""):
        self.id = entry_id
        self.foreign = foreign.strip()
        self.translation = translation.strip()
        self.language_in = language_in
        self.language_out = language_out
        self.section = section
        self.audio_filename: Optional[str] = None  # Will be set after TTS generation

    def __str__(self):
        return f"{self.foreign} :: {self.translation}"


class VocabularyParser:
    """Parses vocabulary files in different formats"""

    @staticmethod
    def parse_ini_file(file_path: str, language_in: str, language_out: str) -> List[VocabularyEntry]:
        """Parse INI format vocabulary file"""
        entries = []
        current_section = ""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue

                    # Parse section headers [section/name]
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        continue

                    # Parse entries: ID = foreign :: translation
                    if ' = ' in line and ' :: ' in line:
                        try:
                            # Split on first ' = '
                            id_part, rest = line.split(' = ', 1)
                            entry_id = int(id_part.strip())

                            # Split on ' :: '
                            if ' :: ' in rest:
                                foreign, translation = rest.split(' :: ', 1)
                                entry = VocabularyEntry(
                                    entry_id, foreign, translation,
                                    language_in, language_out, current_section
                                )
                                entries.append(entry)
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Warning: Skipping malformed line {line_num}: {line}")
                            continue

        except FileNotFoundError:
            print(f"❌ Error: File not found: {file_path}")
            return []
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            return []

        return entries

    @staticmethod
    def parse_json_file(file_path: str, language_in: str, language_out: str) -> List[VocabularyEntry]:
        """Parse JSON format vocabulary file"""
        entries = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            words_list = []
            if isinstance(data, dict) and 'words' in data:
                words_list = data['words']
            elif isinstance(data, list):
                words_list = data
            else:
                print(f"❌ Error: Unsupported JSON format in {file_path}")
                return []

            for i, item in enumerate(words_list):
                if isinstance(item, dict):
                    entry_id = item.get('id', i + 1)
                    foreign = item.get('foreign', '')
                    translation = item.get('translation', '')

                    if foreign and translation:
                        entry = VocabularyEntry(
                            entry_id, foreign, translation,
                            language_in, language_out
                        )
                        entries.append(entry)

        except FileNotFoundError:
            print(f"❌ Error: File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {file_path}: {e}")
            return []
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            return []

        return entries


class TTSDownloader:
    """Downloads TTS audio files from Google Translate"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.rate_limiter = TTSRateLimiter(config.rate_limit_delay)
        self.cache = TTSCache(config.cache_dir)

    def url_encode(self, text: str) -> str:
        """URL encode text for Google TTS API"""
        return urllib.parse.quote(text.encode('utf-8'))

    def build_tts_url(self, text: str, language: str) -> str:
        """Build Google TTS URL with required parameters"""
        encoded_text = self.url_encode(text)
        text_length = len(text.encode('utf-8'))

        params = {
            'ie': 'UTF-8',
            'q': encoded_text,
            'tl': language,
            'total': '1',
            'idx': '0',
            'textlen': str(text_length),
            'client': 'tw-ob',
            'prev': 'input'
        }

        # Build URL manually to match C++ implementation exactly
        url = f"{self.config.base_url}?"
        param_parts = []
        for key, value in params.items():
            if key == 'q':
                # Don't double-encode the text
                param_parts.append(f"{key}={value}")
            else:
                param_parts.append(f"{key}={urllib.parse.quote(str(value))}")

        url += "&".join(param_parts)
        return url

    def download_tts_audio(self, text: str, language: str) -> bool:
        """Download TTS audio file for given text and language"""
        # Check cache first
        cache_path = self.cache.get_cache_path(text, language)
        if self.cache.is_cached(text, language):
            print(f"💾 Cached: {text[:50]}..." if len(text) > 50 else f"💾 Cached: {text}")
            return True

        # Respect rate limits
        self.rate_limiter.wait_if_needed()

        # Build URL
        url = self.build_tts_url(text, language)

        print(f"🔊 Downloading TTS for ({language}): {text[:50]}..." if len(text) > 50 else f"🔊 Downloading TTS for ({language}): {text}")

        if self.config.dry_run:
            print(f"🚀 DRY RUN: Would download from: {url}")
            print(f"🚀 DRY RUN: Would save to: {cache_path}")
            return True

        try:
            # Prepare request with proper headers
            request = urllib.request.Request(url)
            request.add_header('User-Agent', self.config.user_agent)
            request.add_header('Referer', self.config.referer)

            # Download audio
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                if response.getcode() == 200:
                    audio_data = response.read()

                    # Verify we got actual audio data
                    if len(audio_data) > MIN_AUDIO_FILE_SIZE:  # Basic sanity check
                        # Save to cache
                        with open(cache_path, 'wb') as f:
                            f.write(audio_data)

                        print(f"✅ Downloaded: {len(audio_data)} bytes -> {cache_path.name}")
                        return True
                    else:
                        print(f"⚠️ Warning: Received small response ({len(audio_data)} bytes), might be an error")
                        return False
                else:
                    print(f"❌ HTTP Error {response.getcode()}")
                    return False

        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            if e.code == 429:
                print("⏳ Rate limited, increasing delay...")
                self.rate_limiter.delay *= EXPONENTIAL_BACKOFF_MULTIPLIER  # Exponential backoff
            return False
        except urllib.error.URLError as e:
            print(f"❌ Network Error: {e.reason}")
            return False
        except Exception as e:
            print(f"❌ Error downloading TTS: {e}")
            return False

    def chunk_text(self, text: str) -> List[str]:
        """Split long text into chunks suitable for TTS"""
        if len(text) <= self.config.max_text_length:
            return [text]

        chunks = []
        words = text.split()
        current_chunk = ""

        for word in words:
            test_chunk = f"{current_chunk} {word}".strip()
            if len(test_chunk) <= self.config.max_text_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class TTSGenerator:
    """Main TTS generator class"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.downloader = TTSDownloader(config)

    def load_config(self, config_file: str) -> List[Dict]:
        """Load dataset configuration from JSON file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                print(f"❌ Error: Invalid config format in {config_file}")
                return []

        except FileNotFoundError:
            print(f"❌ Error: Config file not found: {config_file}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {config_file}: {e}")
            return []

    def process_dataset(self, dataset_config: Dict, entries: List = None, base_path: str = ".") -> int:
        """Process a single dataset and generate TTS files"""
        filename = dataset_config.get('file', '')
        file_format = dataset_config.get('format', 'ini')
        language_in = dataset_config.get('language-in', 'en')
        language_out = dataset_config.get('language-out', 'en')
        category = dataset_config.get('category', 'unknown')

        if not filename:
            print("⚠️ Warning: Dataset missing 'file' field, skipping")
            return 0

        file_path = os.path.join(base_path, filename)

        print(f"\n📚 Processing dataset: {filename}")
        print(f"   Format: {file_format}")
        print(f"   Languages: {language_in} → {language_out}")
        print(f"   Category: {category}")

        # Use provided entries or parse vocabulary file
        if entries is None:
            if file_format.lower() == 'ini':
                entries = VocabularyParser.parse_ini_file(file_path, language_in, language_out)
            elif file_format.lower() == 'json':
                entries = VocabularyParser.parse_json_file(file_path, language_in, language_out)
            else:
                print(f"❌ Error: Unsupported format '{file_format}'")
                return 0

        if not entries:
            print("⚠️ No vocabulary entries found")
            return 0

        print(f"📝 Found {len(entries)} vocabulary entries")

        # Generate TTS for foreign language words
        success_count = 0
        total_downloads = 0

        for entry in entries:
            # Preprocess foreign text to clean annotations and expand alternatives
            processed_foreign_text = preprocess_tts_text(entry.foreign)

            # Process foreign text
            foreign_chunks = self.downloader.chunk_text(processed_foreign_text)
            entry_audio_files = []

            for chunk in foreign_chunks:
                if self.downloader.download_tts_audio(chunk, language_in):
                    # Get the filename for this chunk
                    audio_filename = self.downloader.cache.get_filename_only(chunk, language_in)
                    entry_audio_files.append(audio_filename)
                    success_count += 1
                total_downloads += 1

                # Small delay between chunks
                if len(foreign_chunks) > 1:
                    time.sleep(CHUNK_DELAY_SECONDS)

            # Set the audio filename(s) for this entry
            if entry_audio_files:
                # If single file, store filename directly; if multiple chunks, store as JSON array
                if len(entry_audio_files) == 1:
                    entry.audio_filename = entry_audio_files[0]
                else:
                    entry.audio_filename = json.dumps(entry_audio_files)

        print(f"✅ Generated TTS for {success_count}/{total_downloads} audio files")
        return success_count

    def save_audio_mappings(self, datasets_with_entries: List[Tuple[Dict, List]], config_file: str):
        """Save audio filename mappings to a config file"""

        # Create audio mapping structure (array format)
        audio_mappings = {}
        current_timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")

        for dataset_config, entries in datasets_with_entries:
            dataset_file = dataset_config.get('file', 'unknown')
            dataset_entries = []

            for entry in entries:
                if entry.audio_filename:
                    # Create entry in array format with timestamp
                    entry_data = {
                        'foreign': entry.foreign,
                        'translation': entry.translation,
                        'audio_filename': entry.audio_filename,
                        'language': entry.language_in,
                        'last_update': current_timestamp
                    }
                    dataset_entries.append(entry_data)

            if dataset_entries:
                audio_mappings[dataset_file] = dataset_entries

        if not audio_mappings:
            print("⚠️ No audio mappings to save")
            return

        # Generate the config file path
        config_dir = os.path.dirname(os.path.abspath(config_file))
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        audio_config_file = os.path.join(self.config.cache_dir, f"{config_name}.json")

        if self.config.dry_run:
            # In dry-run mode, show what would be generated
            print(f"🚨 DRY RUN - Would save audio mappings to: {audio_config_file}")
            print(f"   Total datasets with audio: {len(audio_mappings)}")

            total_entries = sum(len(entries) for entries in audio_mappings.values())
            print(f"   Total entries with audio: {total_entries}")

            # Show a sample of what would be saved
            print(f"\n📋 Sample audio mapping structure:")
            sample_config = {}
            for dataset_file, entries in list(audio_mappings.items())[:1]:  # Show first dataset
                sample_entries = entries[:3]  # Show first 3 entries
                if len(entries) > 3:
                    # Add placeholder entry to show more exist
                    sample_entries.append(f"... and {len(entries) - 3} more entries")
                sample_config[dataset_file] = sample_entries

            print(json.dumps(sample_config, indent=2, ensure_ascii=False))
            return

        # Save the actual file in normal mode
        try:
            with open(audio_config_file, 'w', encoding='utf-8') as f:
                json.dump(audio_mappings, f, indent=2, ensure_ascii=False)

            print(f"💾 Audio mappings saved to: {audio_config_file}")
            print(f"   Total datasets with audio: {len(audio_mappings)}")

            # Show summary
            total_entries = sum(len(entries) for entries in audio_mappings.values())
            print(f"   Total entries with audio: {total_entries}")

        except Exception as e:
            print(f"❌ Error saving audio mappings: {e}")

    def generate_all(self, config_file: str = "config.json") -> int:
        """Generate TTS for all datasets in config file"""
        base_path = os.path.dirname(os.path.abspath(config_file))
        datasets = self.load_config(config_file)

        if not datasets:
            print("❌ No datasets found in config file")
            return 0

        print(f"🚀 Starting TTS generation for {len(datasets)} datasets")
        print(f"📁 Cache directory: {self.config.cache_dir}")
        print(f"⏱️ Rate limit delay: {self.config.rate_limit_delay}s")

        if self.config.dry_run:
            print("🚨 DRY RUN MODE - No files will be downloaded")

        # Clean up old cache files
        self.downloader.cache.cleanup_old_files()

        total_success = 0
        datasets_with_entries = []  # Track datasets and their entries for audio mapping
        expected_filenames = set()  # Track expected MP3 filenames for cleanup

        for i, dataset in enumerate(datasets, 1):
            print(f"\n{'='*60}")
            print(f"Dataset {i}/{len(datasets)}")
            print('='*60)

            # Parse vocabulary file to get entries before processing
            filename = dataset.get('file', '')
            file_format = dataset.get('format', 'ini')
            language_in = dataset.get('language-in', 'en')
            language_out = dataset.get('language-out', 'en')

            if filename:
                file_path = os.path.join(base_path, filename)

                # Parse vocabulary file
                if file_format.lower() == 'ini':
                    entries = VocabularyParser.parse_ini_file(file_path, language_in, language_out)
                elif file_format.lower() == 'json':
                    entries = VocabularyParser.parse_json_file(file_path, language_in, language_out)
                else:
                    entries = []

                # Store the dataset and entries for audio mapping
                datasets_with_entries.append((dataset, entries))

                # Collect expected filenames for cleanup
                for entry in entries:
                    # Preprocess the text the same way as in process_dataset
                    processed_foreign_text = preprocess_tts_text(entry.foreign)
                    chunks = self.downloader.chunk_text(processed_foreign_text)
                    for chunk in chunks:
                        expected_filename = self.downloader.cache.get_filename_only(chunk, language_in)
                        expected_filenames.add(expected_filename)

            success = self.process_dataset(dataset, entries, base_path)
            total_success += success

        print(f"\n{'='*60}")
        print(f"🎉 TTS Generation Complete!")
        print(f"✅ Successfully generated {total_success} TTS audio files")
        print(f"📁 Audio files saved in: {self.config.cache_dir}")

        # Save audio mappings to config file
        if datasets_with_entries:
            print(f"\n📋 Saving audio mappings...")
            self.save_audio_mappings(datasets_with_entries, config_file)

        # Clean up unused MP3 files
        print(f"\n🧹 Cleaning up unused MP3 files...")
        self.downloader.cache.cleanup_unused_files(expected_filenames)

        print('='*60)

        return total_success


def run_internal_tests() -> int:
    """Run internal tests for text preprocessing and other functionality"""

    # Test simplify_diacritics function
    diacritics_test_cases = [
        # Test case 1: Spanish characters
        {
            'input': 'niño',
            'expected': 'nino',
            'description': 'Spanish ñ to n'
        },

        # Test case 2: French characters
        {
            'input': 'café',
            'expected': 'cafe',
            'description': 'French é to e'
        },

        # Test case 3: German characters
        {
            'input': 'Müller',
            'expected': 'Muller',
            'description': 'German ü to u'
        },

        # Test case 4: Polish characters
        {
            'input': 'być',
            'expected': 'byc',
            'description': 'Polish ć to c'
        },

        # Test case 5: Complex mixed diacritics
        {
            'input': 'résumé',
            'expected': 'resume',
            'description': 'Mixed French accents'
        },

        # Test case 6: Special characters
        {
            'input': 'naïve',
            'expected': 'naive',
            'description': 'Diaeresis removal'
        },

        # Test case 7: Nordic characters
        {
            'input': 'børn',
            'expected': 'born',
            'description': 'Nordic ø to o'
        },

        # Test case 8: Already ASCII
        {
            'input': 'hello',
            'expected': 'hello',
            'description': 'ASCII text unchanged'
        },

        # Test case 9: Mixed case with diacritics
        {
            'input': 'CAFÉ',
            'expected': 'CAFE',
            'description': 'Uppercase diacritics'
        },

        # Test case 10: Complex Spanish text
        {
            'input': '¡Adiós!',
            'expected': '¡Adios!',
            'description': 'Spanish exclamation with accent'
        }
    ]

    # Test sanitize_filename function
    filename_test_cases = [
        # Test case 1: Basic sanitization
        {
            'input': 'hello world',
            'expected': 'hello_world',
            'description': 'Spaces to underscores'
        },

        # Test case 2: Diacritics in filename
        {
            'input': 'café niño',
            'expected': 'cafe_nino',
            'description': 'Diacritics simplified and spaces replaced'
        },

        # Test case 3: Special characters
        {
            'input': 'hello@world!',
            'expected': 'hello_world',
            'description': 'Special characters removed'
        },

        # Test case 4: Multiple underscores
        {
            'input': 'hello   world!!!',
            'expected': 'hello_world',
            'description': 'Multiple spaces and punctuation'
        },

        # Test case 5: Leading/trailing spaces
        {
            'input': '  hello world  ',
            'expected': 'hello_world',
            'description': 'Leading and trailing whitespace'
        },

        # Test case 6: Long text truncation
        {
            'input': 'this is a very long text that should be truncated because it exceeds the maximum length',
            'expected': 'this_is_a_very_long_text_that_should_be_truncated',
            'description': 'Long text truncation'
        },

        # Test case 7: Only special characters
        {
            'input': '!!!@@@###',
            'expected': 'audio',
            'description': 'Only special chars fallback to default'
        },

        # Test case 8: Spanish punctuation
        {
            'input': '¡Buenos días!',
            'expected': 'Buenos_dias',
            'description': 'Spanish exclamation marks and accents'
        },

        # Test case 9: Numbers and letters
        {
            'input': 'word123test',
            'expected': 'word123test',
            'description': 'Numbers preserved with letters'
        },

        # Test case 10: Empty string
        {
            'input': '',
            'expected': 'audio',
            'description': 'Empty string fallback'
        },

        # Test case 11: Complex real-world case with slash and diacritics
        {
            'input': 'Aquí tienes/Aquí tienes tú',
            'expected': 'Aqui_tienes_Aqui_tienes_tu',
            'description': 'Real vocabulary entry with slashes and diacritics'
        },

        # Test case 12: Multiple slashes and complex punctuation
        {
            'input': '¿Cómo estás?/¿Qué tal?',
            'expected': 'Como_estas_Que_tal',
            'description': 'Multiple questions with slashes and punctuation'
        }
    ]

    # Test text preprocessing
    preprocessing_test_cases = [
        # Test case 1: Remove square bracket annotations
        {
            'input': 'SER [być]',
            'expected': 'SER',
            'description': 'Remove square bracket annotation'
        },

        # Test case 2: Complex case with slashes (no processing needed)
        {
            'input': 'soy/eres/es somos/sois/son',
            'expected': 'soy/eres/es somos/sois/son',
            'description': 'Leave existing alternatives unchanged'
        },

        # Test case 4: Remove multiple annotations
        {
            'input': 'ESTAR [to be] [permanent]',
            'expected': 'ESTAR',
            'description': 'Remove multiple annotations'
        },

        # Test case 5: Remove annotation with text after
        {
            'input': 'HACER [to do] something',
            'expected': 'HACER something',
            'description': 'Remove annotation with text after'
        },

        # Test case 6: No changes needed
        {
            'input': 'Hello world',
            'expected': 'Hello world',
            'description': 'No preprocessing needed'
        },

        # Test case 7: Empty string
        {
            'input': '',
            'expected': '',
            'description': 'Handle empty string'
        },

        # Test case 8: Complex annotation removal
        {
            'input': 'TENER [to have] [possession] algo importante',
            'expected': 'TENER algo importante',
            'description': 'Remove multiple annotations with text after'
        }
    ]

    print("🧪 Running Internal TTS Tests")
    print("=" * 80)

    all_passed = True
    total_test_count = 0

    # Run diacritics tests
    print("📝 Testing simplify_diacritics() function...")
    print("-" * 50)

    for i, test_case in enumerate(diacritics_test_cases, 1):
        input_text = test_case['input']
        expected = test_case['expected']
        description = test_case['description']

        result = simplify_diacritics(input_text)
        passed = result == expected
        total_test_count += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"Diacritics {i:2d}: {description}")
        print(f"   Input:    '{input_text}'")
        print(f"   Expected: '{expected}'")
        print(f"   Got:      '{result}'")
        print(f"   Status:   {status}")
        print()

        if not passed:
            all_passed = False

    # Run filename sanitization tests
    print("📁 Testing sanitize_filename() function...")
    print("-" * 50)

    for i, test_case in enumerate(filename_test_cases, 1):
        input_text = test_case['input']
        expected = test_case['expected']
        description = test_case['description']

        result = sanitize_filename(input_text)
        passed = result == expected
        total_test_count += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"Filename {i:2d}: {description}")
        print(f"   Input:    '{input_text}'")
        print(f"   Expected: '{expected}'")
        print(f"   Got:      '{result}'")
        print(f"   Status:   {status}")
        print()

        if not passed:
            all_passed = False

    # Run text preprocessing tests
    print("🔧 Testing preprocess_tts_text() function...")
    print("-" * 50)

    for i, test_case in enumerate(preprocessing_test_cases, 1):
        input_text = test_case['input']
        expected = test_case['expected']
        description = test_case['description']

        result = preprocess_tts_text(input_text)
        passed = result == expected
        total_test_count += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"Preprocess {i:1d}: {description}")
        print(f"   Input:    '{input_text}'")
        print(f"   Expected: '{expected}'")
        print(f"   Got:      '{result}'")
        print(f"   Status:   {status}")
        print()

        if not passed:
            all_passed = False

    # Test cache path generation
    print("📂 Testing cache path generation...")
    print("-" * 50)

    cache = TTSCache("test_cache")
    cache_test_cases = [
        {
            'text': 'hello world',
            'language': 'es',
            'description': 'Basic cache path generation'
        },
        {
            'text': '¡Buenos días!',
            'language': 'es',
            'description': 'Spanish text with diacritics'
        },
        {
            'text': 'café niño',
            'language': 'fr',
            'description': 'Mixed diacritics'
        },
        {
            'text': 'Aquí tienes/Aquí tienes tú',
            'language': 'es',
            'description': 'Real vocabulary entry with slashes and diacritics'
        },
        {
            'text': '¿Cómo estás?/¿Qué tal?',
            'language': 'es',
            'description': 'Complex questions with punctuation'
        }
    ]

    for i, test_case in enumerate(cache_test_cases, 1):
        text = test_case['text']
        language = test_case['language']
        description = test_case['description']

        try:
            cache_path = cache.get_cache_path(text, language)
            filename_only = cache.get_filename_only(text, language)

            # Validate the results
            path_valid = cache_path.suffix == '.mp3'
            filename_valid = filename_only.endswith('.mp3')

            passed = path_valid and filename_valid
            total_test_count += 1

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"Cache {i:1d}: {description}")
            print(f"   Input:    '{text}' ({language})")
            print(f"   Path:     {cache_path}")
            print(f"   Filename: {filename_only}")
            print(f"   Status:   {status}")
            print()

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"Cache {i:1d}: {description}")
            print(f"   Input:    '{text}' ({language})")
            print(f"   Error:    {e}")
            print(f"   Status:   ❌ FAIL")
            print()
            all_passed = False
            total_test_count += 1

    print("=" * 80)
    if all_passed:
        print(f"🎉 All {total_test_count} tests passed!")
        return 0
    else:
        print(f"💥 Some tests failed! ({total_test_count} total tests)")
        return 1


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TTS audio files for MemoBar vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_tts.py
  python3 generate_tts.py --config my_config.json --cache-dir ./tts_cache
  python3 generate_tts.py --dry-run --config config.json
        """
    )

    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to config.json file (default: config.json)'
    )

    parser.add_argument(
        '--cache-dir',
        default='audio',
        help='Directory to store TTS audio files (default: audio)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f'Rate limit delay in seconds (default: {DEFAULT_DELAY_SECONDS})'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )

    parser.add_argument(
        '--cleanup',
        type=int,
        metavar='DAYS',
        help='Clean up cache files older than specified days and exit'
    )

    parser.add_argument(
        '--cleanup-unused',
        action='store_true',
        help='Remove MP3 files that don\'t correspond to any entries in the dataset and exit'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run internal tests for text preprocessing and other functionality'
    )

    args = parser.parse_args()

    # Initialize configuration
    config = TTSConfig()
    config.cache_dir = args.cache_dir
    config.rate_limit_delay = args.delay
    config.dry_run = args.dry_run

    # Handle cleanup mode
    if args.cleanup is not None:
        print(f"🧹 Cleaning up cache files older than {args.cleanup} days...")
        cache = TTSCache(config.cache_dir)
        cache.cleanup_old_files(args.cleanup)
        return 0

    # Handle cleanup-unused mode
    if args.cleanup_unused:
        print(f"🧹 Cleaning up unused MP3 files...")
        generator = TTSGenerator(config)
        datasets = generator.load_config(args.config)

        if not datasets:
            print("❌ No datasets found in config file")
            return 1

        # Collect expected filenames from all datasets
        base_path = os.path.dirname(os.path.abspath(args.config))
        expected_filenames = set()

        for dataset in datasets:
            filename = dataset.get('file', '')
            file_format = dataset.get('format', 'ini')
            language_in = dataset.get('language-in', 'en')
            language_out = dataset.get('language-out', 'en')

            if filename:
                file_path = os.path.join(base_path, filename)

                # Parse vocabulary file
                if file_format.lower() == 'ini':
                    entries = VocabularyParser.parse_ini_file(file_path, language_in, language_out)
                elif file_format.lower() == 'json':
                    entries = VocabularyParser.parse_json_file(file_path, language_in, language_out)
                else:
                    continue

                # Collect expected filenames
                for entry in entries:
                    processed_foreign_text = preprocess_tts_text(entry.foreign)
                    chunks = generator.downloader.chunk_text(processed_foreign_text)
                    for chunk in chunks:
                        expected_filename = generator.downloader.cache.get_filename_only(chunk, language_in)
                        expected_filenames.add(expected_filename)

        # Perform cleanup
        generator.downloader.cache.cleanup_unused_files(expected_filenames)
        return 0

    # Handle test mode
    if args.test:
        print("🧪 Running internal tests...")
        return run_internal_tests()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file '{args.config}' not found")
        print(f"💡 Create a '{args.config}' file with dataset information")
        return 1

    try:
        # Generate TTS files
        generator = TTSGenerator(config)
        success_count = generator.generate_all(args.config)

        return 0 if success_count > 0 else 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
