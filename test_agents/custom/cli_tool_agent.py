#!/usr/bin/env python3
"""
CLI Tool Agent Example

This demonstrates a command-line interface agent that can be integrated with Mark-1.
It shows how CLI-based agents can be wrapped and integrated into the system.
"""

import argparse
import json
import sys
import time
from typing import Dict, Any


def data_processor_agent(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main processing function for CLI agent
    
    This agent provides data processing capabilities:
    - Data transformation
    - Statistical analysis
    - Format conversion
    """
    
    operation = input_data.get("operation", "stats")
    data = input_data.get("data", [])
    
    processing_start = time.time()
    
    if operation == "stats":
        if not data:
            return {
                "error": "No data provided for statistics",
                "operation": operation,
                "success": False
            }
        
        # Calculate basic statistics
        try:
            numeric_data = [float(x) for x in data if str(x).replace('.', '').replace('-', '').isdigit()]
            
            if not numeric_data:
                return {
                    "error": "No numeric data found",
                    "operation": operation,
                    "success": False
                }
            
            stats = {
                "count": len(numeric_data),
                "sum": sum(numeric_data),
                "mean": sum(numeric_data) / len(numeric_data),
                "min": min(numeric_data),
                "max": max(numeric_data),
                "range": max(numeric_data) - min(numeric_data)
            }
            
            # Calculate median
            sorted_data = sorted(numeric_data)
            n = len(sorted_data)
            if n % 2 == 0:
                stats["median"] = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
            else:
                stats["median"] = sorted_data[n//2]
            
            return {
                "operation": "statistics",
                "statistics": stats,
                "processing_time": time.time() - processing_start,
                "success": True,
                "agent_type": "cli_tool"
            }
            
        except Exception as e:
            return {
                "error": f"Statistics calculation failed: {str(e)}",
                "operation": operation,
                "success": False
            }
    
    elif operation == "transform":
        transformation = input_data.get("transform_type", "uppercase")
        
        if transformation == "uppercase":
            result = [str(item).upper() for item in data]
        elif transformation == "lowercase":
            result = [str(item).lower() for item in data]
        elif transformation == "reverse":
            result = [str(item)[::-1] for item in data]
        elif transformation == "sort":
            result = sorted([str(item) for item in data])
        else:
            return {
                "error": f"Unknown transformation: {transformation}",
                "operation": operation,
                "success": False
            }
        
        return {
            "operation": "transform",
            "transformation": transformation,
            "original_data": data,
            "transformed_data": result,
            "processing_time": time.time() - processing_start,
            "success": True,
            "agent_type": "cli_tool"
        }
    
    elif operation == "format":
        output_format = input_data.get("format", "json")
        
        if output_format == "json":
            formatted_data = json.dumps(data, indent=2)
        elif output_format == "csv":
            # Simple CSV formatting
            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    # List of dictionaries
                    headers = list(data[0].keys())
                    csv_lines = [",".join(headers)]
                    for row in data:
                        csv_lines.append(",".join(str(row.get(h, "")) for h in headers))
                    formatted_data = "\n".join(csv_lines)
                else:
                    # Simple list
                    formatted_data = ",".join(str(item) for item in data)
            else:
                formatted_data = str(data)
        elif output_format == "text":
            formatted_data = "\n".join(str(item) for item in data)
        else:
            return {
                "error": f"Unknown format: {output_format}",
                "operation": operation,
                "success": False
            }
        
        return {
            "operation": "format",
            "format": output_format,
            "formatted_data": formatted_data,
            "processing_time": time.time() - processing_start,
            "success": True,
            "agent_type": "cli_tool"
        }
    
    else:
        return {
            "error": f"Unknown operation: {operation}",
            "available_operations": ["stats", "transform", "format"],
            "success": False
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="CLI Data Processing Agent")
    parser.add_argument("--input", type=str, required=True, help="Input data as JSON string")
    parser.add_argument("--output", type=str, help="Output file path (optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Parse input JSON
        input_data = json.loads(args.input)
        
        if args.verbose:
            print(f"Processing input: {json.dumps(input_data, indent=2)}", file=sys.stderr)
        
        # Process the data
        result = data_processor_agent(input_data)
        
        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            if args.verbose:
                print(f"Result written to: {args.output}", file=sys.stderr)
        else:
            print(json.dumps(result, indent=2))
    
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Invalid JSON input: {str(e)}",
            "success": False
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    
    except Exception as e:
        error_result = {
            "error": f"Processing failed: {str(e)}",
            "success": False
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main() 