#!/bin/bash
# Script-based Agent Example
# 
# This demonstrates how shell scripts can be integrated as agents.
# The script provides file processing and system information capabilities.

# Default values
INPUT_FILE=""
OUTPUT_FILE=""
OPERATION="info"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_JSON="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to log verbose messages
log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "INFO: $1" >&2
    fi
}

# Function to process file operations
process_file_operation() {
    local operation="$1"
    local file_path="$2"
    
    case $operation in
        "count_lines")
            if [ -f "$file_path" ]; then
                line_count=$(wc -l < "$file_path")
                echo "\"line_count\": $line_count"
            else
                echo "\"error\": \"File not found: $file_path\""
            fi
            ;;
        "file_size")
            if [ -f "$file_path" ]; then
                file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null)
                echo "\"file_size_bytes\": $file_size"
            else
                echo "\"error\": \"File not found: $file_path\""
            fi
            ;;
        "file_info")
            if [ -f "$file_path" ]; then
                file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null)
                line_count=$(wc -l < "$file_path")
                word_count=$(wc -w < "$file_path")
                echo "\"file_info\": {"
                echo "  \"path\": \"$file_path\","
                echo "  \"size_bytes\": $file_size,"
                echo "  \"line_count\": $line_count,"
                echo "  \"word_count\": $word_count"
                echo "}"
            else
                echo "\"error\": \"File not found: $file_path\""
            fi
            ;;
        *)
            echo "\"error\": \"Unknown file operation: $operation\""
            ;;
    esac
}

# Function to get system information
get_system_info() {
    echo "\"system_info\": {"
    echo "  \"hostname\": \"$(hostname)\","
    echo "  \"os\": \"$(uname -s)\","
    echo "  \"kernel\": \"$(uname -r)\","
    echo "  \"architecture\": \"$(uname -m)\","
    echo "  \"uptime\": \"$(uptime | awk '{print $3}' | sed 's/,//')\","
    echo "  \"current_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "}"
}

# Main processing function
main() {
    log_verbose "Starting script agent processing"
    
    # Parse JSON input if provided
    if [ -n "$INPUT_JSON" ]; then
        log_verbose "Parsing input JSON: $INPUT_JSON"
        
        # Extract values from JSON (simple parsing)
        OPERATION=$(echo "$INPUT_JSON" | grep -o '"operation":"[^"]*"' | cut -d'"' -f4)
        FILE_PATH=$(echo "$INPUT_JSON" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)
        
        if [ -z "$OPERATION" ]; then
            OPERATION="info"
        fi
        
        log_verbose "Extracted operation: $OPERATION"
        log_verbose "Extracted file_path: $FILE_PATH"
    fi
    
    # Start JSON response
    echo "{"
    echo "  \"agent_type\": \"script_based\","
    echo "  \"operation\": \"$OPERATION\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"success\": true,"
    
    # Process based on operation
    case $OPERATION in
        "system_info"|"info")
            get_system_info
            ;;
        "count_lines"|"file_size"|"file_info")
            if [ -n "$FILE_PATH" ]; then
                process_file_operation "$OPERATION" "$FILE_PATH"
            else
                echo "  \"error\": \"No file path provided for file operation\","
                echo "  \"success\": false"
            fi
            ;;
        "list_files")
            DIRECTORY=$(echo "$INPUT_JSON" | grep -o '"directory":"[^"]*"' | cut -d'"' -f4)
            if [ -z "$DIRECTORY" ]; then
                DIRECTORY="."
            fi
            
            if [ -d "$DIRECTORY" ]; then
                echo "  \"directory_listing\": ["
                first=true
                for file in "$DIRECTORY"/*; do
                    if [ "$first" = true ]; then
                        first=false
                    else
                        echo ","
                    fi
                    echo -n "    \"$(basename "$file")\""
                done
                echo ""
                echo "  ]"
            else
                echo "  \"error\": \"Directory not found: $DIRECTORY\","
                echo "  \"success\": false"
            fi
            ;;
        "environment")
            echo "  \"environment\": {"
            echo "    \"shell\": \"$SHELL\","
            echo "    \"user\": \"$USER\","
            echo "    \"home\": \"$HOME\","
            echo "    \"pwd\": \"$(pwd)\","
            echo "    \"path_count\": $(echo $PATH | tr ':' '\n' | wc -l)"
            echo "  }"
            ;;
        *)
            echo "  \"error\": \"Unknown operation: $OPERATION\","
            echo "  \"available_operations\": [\"system_info\", \"count_lines\", \"file_size\", \"file_info\", \"list_files\", \"environment\"],"
            echo "  \"success\": false"
            ;;
    esac
    
    # Close JSON response
    echo "}"
    
    log_verbose "Script agent processing completed"
}

# Run main function
main 