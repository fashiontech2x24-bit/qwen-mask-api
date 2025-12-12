#!/bin/bash
# Download pre-trained SCHP models

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "Downloading SCHP pre-trained models..."
echo "=========================================="

# ATR Model (Recommended for fashion)
echo "Downloading ATR model..."
wget -O "$MODELS_DIR/exp-schp-201908301523-atr.pth" \
    "https://drive.google.com/uc?export=download&id=1ruJg4lqR_jgQPj-9K1j6XmTd3yisGXq7" || \
    echo "Warning: ATR model download failed. Please download manually from:"
    echo "  https://github.com/GoGoDuck912/Self-Correction-Human-Parsing"

# LIP Model
echo "Downloading LIP model..."
wget -O "$MODELS_DIR/exp-schp-201908261155-lip.pth" \
    "https://drive.google.com/uc?export=download&id=1_L8A1_1xc3j7k3oj3JYhR0vKD8t85Q9-" || \
    echo "Warning: LIP model download failed. Please download manually"

# Pascal-Person-Part Model
echo "Downloading Pascal-Person-Part model..."
wget -O "$MODELS_DIR/exp-schp-201908270938-pascal-person-part.pth" \
    "https://drive.google.com/uc?export=download&id=1E5YwNKW2VO3ay17aT2pU35p0P9l-2FR5" || \
    echo "Warning: Pascal model download failed. Please download manually"

echo ""
echo "=========================================="
echo "Model download complete!"
echo "=========================================="
echo ""
echo "Note: If downloads failed, please download models manually from:"
echo "  https://github.com/GoGoDuck912/Self-Correction-Human-Parsing"
echo ""
echo "Place them in the '$MODELS_DIR' directory"

