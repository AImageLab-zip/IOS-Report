#!/bin/bash
# Setup script for auto-captioning system

set -e

echo "=================================="
echo "Auto-Captioning System Setup"
echo "=================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OpenAI API key:"
    echo "   nano .env"
    echo ""
else
    echo "✓ .env file already exists"
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p outputs
echo "✓ Output directory created"
echo ""

# Run verification tests
echo "Running verification tests..."
python test_setup.py

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
