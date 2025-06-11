#!/usr/bin/env python3
"""
Create Example Plugins for Mark-1 Universal Plugin System

This script creates several example plugins to demonstrate different types
of functionality and plugin patterns.
"""

import asyncio
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from mark1.plugins import PluginManager


async def create_text_analyzer_plugin(plugins_dir: Path) -> Path:
    """Create a text analysis plugin"""
    plugin_dir = plugins_dir / "text_analyzer_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Main script
    (plugin_dir / "analyze.py").write_text("""#!/usr/bin/env python3
import sys
import json
import re
from collections import Counter
import argparse

def analyze_text(text):
    \"\"\"Comprehensive text analysis\"\"\"
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len(text.split('\\n\\n')),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "most_common_words": dict(Counter(words).most_common(10)),
        "readability_score": calculate_readability(text),
        "sentiment": analyze_sentiment(text)
    }

def calculate_readability(text):
    \"\"\"Simple readability score\"\"\"
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    if not words or not sentences:
        return 0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple readability formula
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 100)
    return max(0, min(100, score))

def analyze_sentiment(text):
    \"\"\"Basic sentiment analysis\"\"\"
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def main():
    parser = argparse.ArgumentParser(description="Advanced text analysis tool")
    parser.add_argument("action", choices=["analyze", "sentiment", "readability", "stats"])
    parser.add_argument("--input", required=True, help="Input text to analyze")
    parser.add_argument("--format", choices=["json", "text"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    if args.action == "analyze":
        result = analyze_text(args.input)
    elif args.action == "sentiment":
        result = {"sentiment": analyze_sentiment(args.input)}
    elif args.action == "readability":
        result = {"readability_score": calculate_readability(args.input)}
    elif args.action == "stats":
        words = args.input.split()
        result = {
            "word_count": len(words),
            "character_count": len(args.input),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
""")
    
    # README
    (plugin_dir / "README.md").write_text("""
# Text Analyzer Plugin

A comprehensive text analysis tool that provides detailed insights into text content.

## Features

- **Word and character counting**
- **Sentence and paragraph analysis**
- **Readability scoring**
- **Sentiment analysis**
- **Most common words identification**
- **Average word length calculation**

## Usage

```bash
# Full analysis
python analyze.py analyze --input "Your text here"

# Sentiment analysis only
python analyze.py sentiment --input "I love this product!"

# Readability score
python analyze.py readability --input "Complex text with long sentences."

# Basic statistics
python analyze.py stats --input "Simple text for counting."
```

## Output

Returns JSON with comprehensive analysis including:
- Word count, character count, sentence count
- Readability score (0-100, higher is more readable)
- Sentiment (positive/negative/neutral)
- Most common words
- Average word length

## Dependencies

None - uses only Python standard library.
""")
    
    # Requirements (empty for this plugin)
    (plugin_dir / "requirements.txt").write_text("")
    
    return plugin_dir


async def create_file_processor_plugin(plugins_dir: Path) -> Path:
    """Create a file processing plugin"""
    plugin_dir = plugins_dir / "file_processor_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Main script
    (plugin_dir / "process.py").write_text("""#!/usr/bin/env python3
import sys
import json
import os
import hashlib
import mimetypes
from pathlib import Path
import argparse

def get_file_info(file_path):
    \"\"\"Get comprehensive file information\"\"\"
    path = Path(file_path)
    
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    stat = path.stat()
    
    # Calculate file hash
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
    except:
        file_hash = "unable_to_calculate"
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    
    return {
        "name": path.name,
        "path": str(path.absolute()),
        "size_bytes": stat.st_size,
        "size_human": format_bytes(stat.st_size),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "accessed": stat.st_atime,
        "mime_type": mime_type or "unknown",
        "extension": path.suffix,
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
        "md5_hash": file_hash,
        "permissions": oct(stat.st_mode)[-3:]
    }

def format_bytes(bytes_size):
    \"\"\"Format bytes in human readable format\"\"\"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def list_directory(dir_path, recursive=False):
    \"\"\"List directory contents\"\"\"
    path = Path(dir_path)
    
    if not path.exists():
        return {"error": f"Directory not found: {dir_path}"}
    
    if not path.is_dir():
        return {"error": f"Not a directory: {dir_path}"}
    
    files = []
    directories = []
    
    try:
        if recursive:
            for item in path.rglob("*"):
                info = get_file_info(item)
                if item.is_file():
                    files.append(info)
                elif item.is_dir():
                    directories.append(info)
        else:
            for item in path.iterdir():
                info = get_file_info(item)
                if item.is_file():
                    files.append(info)
                elif item.is_dir():
                    directories.append(info)
    except PermissionError:
        return {"error": f"Permission denied: {dir_path}"}
    
    return {
        "directory": str(path.absolute()),
        "total_files": len(files),
        "total_directories": len(directories),
        "files": files,
        "directories": directories
    }

def copy_file(source, destination):
    \"\"\"Copy file with verification\"\"\"
    import shutil
    
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            return {"error": f"Source file not found: {source}"}
        
        # Create destination directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source, destination)
        
        # Verify copy
        source_info = get_file_info(source)
        dest_info = get_file_info(destination)
        
        success = (source_info.get("md5_hash") == dest_info.get("md5_hash") and
                  source_info.get("size_bytes") == dest_info.get("size_bytes"))
        
        return {
            "success": success,
            "source": source_info,
            "destination": dest_info,
            "verified": success
        }
        
    except Exception as e:
        return {"error": f"Copy failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="File processing utility")
    parser.add_argument("action", choices=["info", "list", "copy", "hash"])
    parser.add_argument("--path", required=True, help="File or directory path")
    parser.add_argument("--destination", help="Destination path (for copy)")
    parser.add_argument("--recursive", action="store_true", help="Recursive operation")
    
    args = parser.parse_args()
    
    if args.action == "info":
        result = get_file_info(args.path)
    elif args.action == "list":
        result = list_directory(args.path, args.recursive)
    elif args.action == "copy":
        if not args.destination:
            result = {"error": "Destination path required for copy operation"}
        else:
            result = copy_file(args.path, args.destination)
    elif args.action == "hash":
        info = get_file_info(args.path)
        result = {"file": args.path, "md5_hash": info.get("md5_hash", "error")}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
""")
    
    # README
    (plugin_dir / "README.md").write_text("""
# File Processor Plugin

A comprehensive file and directory processing utility.

## Features

- **File information extraction** (size, dates, permissions, hash)
- **Directory listing** (recursive and non-recursive)
- **File copying with verification**
- **Hash calculation** (MD5)
- **Human-readable file sizes**
- **MIME type detection**

## Usage

```bash
# Get file information
python process.py info --path "/path/to/file.txt"

# List directory contents
python process.py list --path "/path/to/directory"

# List directory recursively
python process.py list --path "/path/to/directory" --recursive

# Copy file with verification
python process.py copy --path "/source/file.txt" --destination "/dest/file.txt"

# Calculate file hash
python process.py hash --path "/path/to/file.txt"
```

## Output

Returns JSON with detailed file/directory information including:
- File metadata (size, dates, permissions)
- MD5 hash for verification
- MIME type detection
- Human-readable formatting

## Dependencies

None - uses only Python standard library.
""")
    
    (plugin_dir / "requirements.txt").write_text("")
    
    return plugin_dir


async def create_data_converter_plugin(plugins_dir: Path) -> Path:
    """Create a data format converter plugin"""
    plugin_dir = plugins_dir / "data_converter_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Main script
    (plugin_dir / "convert.py").write_text("""#!/usr/bin/env python3
import sys
import json
import csv
import xml.etree.ElementTree as ET
from io import StringIO
import argparse

def json_to_csv(json_data):
    \"\"\"Convert JSON to CSV format\"\"\"
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()
        elif isinstance(data, dict):
            # Single dictionary
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
            return output.getvalue()
        else:
            return "Error: JSON must be a dictionary or list of dictionaries"
    except Exception as e:
        return f"Error converting JSON to CSV: {str(e)}"

def csv_to_json(csv_data):
    \"\"\"Convert CSV to JSON format\"\"\"
    try:
        reader = csv.DictReader(StringIO(csv_data))
        result = list(reader)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error converting CSV to JSON: {str(e)}"

def json_to_xml(json_data, root_name="root"):
    \"\"\"Convert JSON to XML format\"\"\"
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        def dict_to_xml(d, parent):
            for key, value in d.items():
                child = ET.SubElement(parent, str(key))
                if isinstance(value, dict):
                    dict_to_xml(value, child)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            item_elem = ET.SubElement(child, "item")
                            dict_to_xml(item, item_elem)
                        else:
                            item_elem = ET.SubElement(child, "item")
                            item_elem.text = str(item)
                else:
                    child.text = str(value)
        
        root = ET.Element(root_name)
        if isinstance(data, dict):
            dict_to_xml(data, root)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_elem = ET.SubElement(root, f"item_{i}")
                if isinstance(item, dict):
                    dict_to_xml(item, item_elem)
                else:
                    item_elem.text = str(item)
        
        return ET.tostring(root, encoding='unicode')
    except Exception as e:
        return f"Error converting JSON to XML: {str(e)}"

def xml_to_json(xml_data):
    \"\"\"Convert XML to JSON format\"\"\"
    try:
        def xml_to_dict(element):
            result = {}
            
            # Add attributes
            if element.attrib:
                result.update(element.attrib)
            
            # Add text content
            if element.text and element.text.strip():
                if len(element) == 0:  # No children
                    return element.text.strip()
                else:
                    result['text'] = element.text.strip()
            
            # Add children
            for child in element:
                child_data = xml_to_dict(child)
                if child.tag in result:
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
            
            return result
        
        root = ET.fromstring(xml_data)
        result = {root.tag: xml_to_dict(root)}
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error converting XML to JSON: {str(e)}"

def validate_json(json_data):
    \"\"\"Validate JSON format\"\"\"
    try:
        json.loads(json_data)
        return {"valid": True, "message": "Valid JSON"}
    except json.JSONDecodeError as e:
        return {"valid": False, "message": f"Invalid JSON: {str(e)}"}

def format_json(json_data, indent=2):
    \"\"\"Format/prettify JSON\"\"\"
    try:
        data = json.loads(json_data)
        return json.dumps(data, indent=indent, sort_keys=True)
    except Exception as e:
        return f"Error formatting JSON: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Data format converter")
    parser.add_argument("action", choices=["json2csv", "csv2json", "json2xml", "xml2json", "validate", "format"])
    parser.add_argument("--input", required=True, help="Input data")
    parser.add_argument("--root", default="root", help="Root element name for XML")
    parser.add_argument("--indent", type=int, default=2, help="Indentation for formatting")
    
    args = parser.parse_args()
    
    if args.action == "json2csv":
        result = json_to_csv(args.input)
    elif args.action == "csv2json":
        result = csv_to_json(args.input)
    elif args.action == "json2xml":
        result = json_to_xml(args.input, args.root)
    elif args.action == "xml2json":
        result = xml_to_json(args.input)
    elif args.action == "validate":
        result = json.dumps(validate_json(args.input), indent=2)
    elif args.action == "format":
        result = format_json(args.input, args.indent)
    
    print(result)

if __name__ == "__main__":
    main()
""")
    
    # README
    (plugin_dir / "README.md").write_text("""
# Data Converter Plugin

A versatile data format conversion utility supporting multiple formats.

## Features

- **JSON ↔ CSV conversion**
- **JSON ↔ XML conversion**
- **JSON validation**
- **JSON formatting/prettifying**
- **Error handling and validation**

## Usage

```bash
# Convert JSON to CSV
python convert.py json2csv --input '{"name":"John","age":30}'

# Convert CSV to JSON
python convert.py csv2json --input "name,age\\nJohn,30\\nJane,25"

# Convert JSON to XML
python convert.py json2xml --input '{"user":{"name":"John","age":30}}' --root "users"

# Convert XML to JSON
python convert.py xml2json --input '<users><user><name>John</name><age>30</age></user></users>'

# Validate JSON
python convert.py validate --input '{"valid": "json"}'

# Format/prettify JSON
python convert.py format --input '{"compact":"json"}' --indent 4
```

## Supported Formats

- **JSON**: JavaScript Object Notation
- **CSV**: Comma-Separated Values
- **XML**: eXtensible Markup Language

## Dependencies

None - uses only Python standard library.
""")
    
    (plugin_dir / "requirements.txt").write_text("")
    
    return plugin_dir


async def main():
    """Create all example plugins"""
    print("🔧 Creating Example Plugins for Mark-1")
    print("=" * 50)
    
    # Create plugins directory
    plugins_dir = Path.home() / ".mark1" / "example_plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plugins
    plugins = []
    
    print("📝 Creating Text Analyzer Plugin...")
    text_analyzer = await create_text_analyzer_plugin(plugins_dir)
    plugins.append(("Text Analyzer", text_analyzer))
    
    print("📁 Creating File Processor Plugin...")
    file_processor = await create_file_processor_plugin(plugins_dir)
    plugins.append(("File Processor", file_processor))
    
    print("🔄 Creating Data Converter Plugin...")
    data_converter = await create_data_converter_plugin(plugins_dir)
    plugins.append(("Data Converter", data_converter))
    
    print("\n✅ Example Plugins Created Successfully!")
    print("=" * 50)
    
    for name, path in plugins:
        print(f"📦 {name}: {path}")
        print(f"   Files: {len(list(path.glob('*')))} files created")
    
    print(f"\n📍 All plugins created in: {plugins_dir}")
    print("\n🚀 Next Steps:")
    print("1. Install plugins using: mark1 plugin install <plugin_directory>")
    print("2. Test plugins individually")
    print("3. Use in orchestration workflows")
    
    return plugins


if __name__ == "__main__":
    asyncio.run(main())
