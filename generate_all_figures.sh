#!/bin/bash

# Script to generate all figures from the Iron Mind manuscript
# Usage: ./generate_all_figures.sh [path_to_run_data]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Iron Mind Figure Generation Script${NC}"
echo "=================================="

# Check if we're in the right directory
if [ ! -d "figures" ]; then
    echo -e "${RED}Error: figures/ directory not found. Please run this script from the repository root.${NC}"
    exit 1
fi

# Get run data path
RUN_DATA_PATH=""
if [ $# -eq 1 ]; then
    RUN_DATA_PATH="$1"
    echo -e "${GREEN}Using provided run data path: ${RUN_DATA_PATH}${NC}"
    
    # Verify the path exists
    if [ ! -d "$RUN_DATA_PATH" ]; then
        echo -e "${RED}Error: Run data path does not exist: ${RUN_DATA_PATH}${NC}"
        exit 1
    fi
elif [ $# -gt 1 ]; then
    echo -e "${RED}Error: Too many arguments. Usage: $0 [path_to_run_data]${NC}"
    exit 1
fi

cd figures

echo ""
echo -e "${BLUE}Generating figures...${NC}"
echo ""

# Figure 2 - Dataset objective histograms (no run data needed)
echo -e "${YELLOW}Generating Figure 2 - Dataset objective histograms...${NC}"
python figure_2.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figure 2 completed${NC}"
else
    echo -e "${RED}✗ Figure 2 failed${NC}"
    exit 1
fi
echo ""

# Figure 3 - Optimization complexity analysis (no run data needed)
echo -e "${YELLOW}Generating Figure 3 - Optimization complexity analysis...${NC}"
python figure_3.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figure 3 completed${NC}"
else
    echo -e "${RED}✗ Figure 3 failed${NC}"
    exit 1
fi
echo ""

# Figures that need run data
if [ -n "$RUN_DATA_PATH" ]; then
    # Figure 5 - LLM optimization performance
    echo -e "${YELLOW}Generating Figure 5 - LLM optimization performance...${NC}"
    python figure_5.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure 5 completed${NC}"
    else
        echo -e "${RED}✗ Figure 5 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure 6 - Duplicate suggestion analysis
    echo -e "${YELLOW}Generating Figure 6 - Duplicate suggestion analysis...${NC}"
    python figure_6.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure 6 completed${NC}"
    else
        echo -e "${RED}✗ Figure 6 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure 7 - Entropy analysis
    echo -e "${YELLOW}Generating Figure 7 - Entropy analysis...${NC}"
    python figure_7.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure 7 completed${NC}"
    else
        echo -e "${RED}✗ Figure 7 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure SI - Convergence analysis
    echo -e "${YELLOW}Generating Figure SI - Convergence analysis...${NC}"
    python figure_SI_convergence.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure SI completed${NC}"
    else
        echo -e "${RED}✗ Figure SI failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}No run data path provided. Skipping figures that require run data:${NC}"
    echo "  - Figure 5 (LLM optimization performance)"
    echo "  - Figure 6 (Duplicate suggestion analysis)"
    echo "  - Figure 7 (Entropy analysis)"
    echo "  - Figure SI (Convergence analysis)"
    echo ""
    echo -e "${BLUE}To generate all figures, run: $0 /path/to/run/data${NC}"
    echo ""
fi

cd ..

echo -e "${GREEN}Figure generation completed!${NC}"
echo ""
echo -e "${BLUE}Generated figures are saved in: figures/pngs/${NC}"

# List generated files
if [ -d "figures/pngs" ]; then
    echo ""
    echo "Generated files:"
    ls -la figures/pngs/*.png 2>/dev/null | while read line; do
        echo "  $line"
    done
fi

echo ""
echo -e "${GREEN}Done!${NC}"
