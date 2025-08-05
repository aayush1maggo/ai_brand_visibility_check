#!/usr/bin/env python3

import ast
import sys

def check_syntax(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse the file
        ast.parse(source)
        print(f"✓ {filename} has valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {filename} has syntax error at line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False
    except Exception as e:
        print(f"✗ Error reading {filename}: {e}")
        return False

if __name__ == "__main__":
    success = check_syntax("app.py")
    sys.exit(0 if success else 1) 