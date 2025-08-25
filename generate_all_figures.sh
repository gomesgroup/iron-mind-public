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

# Figures that don't need run data
echo -e "${BLUE}=== Figures that don't require run data ===${NC}"
echo ""

# Figure 2 - Dataset objective histograms
echo -e "${YELLOW}Generating Figure 2 - Dataset objective histograms...${NC}"
python figure_2.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figure 2 completed${NC}"
else
    echo -e "${RED}✗ Figure 2 failed${NC}"
    exit 1
fi
echo ""

# Figure 3 - Optimization complexity analysis
echo -e "${YELLOW}Generating Figure 3 - Optimization complexity analysis...${NC}"
python figure_3.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figure 3 completed${NC}"
else
    echo -e "${RED}✗ Figure 3 failed${NC}"
    exit 1
fi
echo ""

# Mock BO visualization
echo -e "${YELLOW}Generating Mock BO visualization...${NC}"
python mock_bo.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Mock BO completed${NC}"
else
    echo -e "${RED}✗ Mock BO failed${NC}"
    exit 1
fi
echo ""

# Figures that need run data
if [ -n "$RUN_DATA_PATH" ]; then
    echo -e "${BLUE}=== Figures that require run data ===${NC}"
    echo ""

    # Figure 5 & S12 - LLM optimization performance
    echo -e "${YELLOW}Generating Figure 5 & S12 - LLM optimization performance...${NC}"
    python figure_5_S12.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure 5 & S12 completed${NC}"
    else
        echo -e "${RED}✗ Figure 5 & S12 failed${NC}"
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

    # Figure 7 & S10 - Entropy analysis
    echo -e "${YELLOW}Generating Figure 7 & S10 - Entropy analysis...${NC}"
    python figure_7_S10.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure 7 & S10 completed${NC}"
    else
        echo -e "${RED}✗ Figure 7 & S10 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S6 & S7 - Dataset analysis
    echo -e "${YELLOW}Generating Figures S6 & S7 - Dataset analysis...${NC}"
    python figure_S6_S7.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figures S6 & S7 completed${NC}"
    else
        echo -e "${RED}✗ Figures S6 & S7 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S8 - Parameter exploration analysis
    echo -e "${YELLOW}Generating Figure S8 - Parameter exploration analysis...${NC}"
    python figure_S8.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure S8 completed${NC}"
    else
        echo -e "${RED}✗ Figure S8 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S1 & S2 - Additional analysis
    echo -e "${YELLOW}Generating Figures S1 & S2 - Additional analysis...${NC}"
    python figure_S1_S2.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figures S1 & S2 completed${NC}"
    else
        echo -e "${RED}✗ Figures S1 & S2 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S3 - Performance analysis
    echo -e "${YELLOW}Generating Figure S3 - Performance analysis...${NC}"
    python figure_S3.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure S3 completed${NC}"
    else
        echo -e "${RED}✗ Figure S3 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S4 & S5 - Time-to-threshold analysis
    echo -e "${YELLOW}Generating Figures S4 & S5 - Time-to-threshold analysis...${NC}"
    python figure_S4_S5.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figures S4 & S5 completed${NC}"
    else
        echo -e "${RED}✗ Figures S4 & S5 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S4 All - Complete time-to-threshold analysis
    echo -e "${YELLOW}Generating Figure S4 All - Complete time-to-threshold analysis...${NC}"
    python figure_S4_all.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure S4 All completed${NC}"
    else
        echo -e "${RED}✗ Figure S4 All failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S9 - Parameter analysis
    echo -e "${YELLOW}Generating Figure S9 - Parameter analysis...${NC}"
    python figure_S9.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure S9 completed${NC}"
    else
        echo -e "${RED}✗ Figure S9 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S11 - Additional convergence analysis
    echo -e "${YELLOW}Generating Figure S11 - Additional convergence analysis...${NC}"
    python figure_S11.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure S11 completed${NC}"
    else
        echo -e "${RED}✗ Figure S11 failed${NC}"
        exit 1
    fi
    echo ""

    # Figure S13 - Comprehensive analysis
    echo -e "${YELLOW}Generating Figure S13 - Comprehensive analysis...${NC}"
    python figure_S13.py "$RUN_DATA_PATH"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Figure S13 completed${NC}"
    else
        echo -e "${RED}✗ Figure S13 failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}No run data path provided. Skipping figures that require run data:${NC}"
    echo "  - Figure 5 & S12 (LLM optimization performance)"
    echo "  - Figure 6 (Duplicate suggestion analysis)"
    echo "  - Figure 7 & S10 (Entropy analysis)"
    echo "  - Figures S1 & S2 (Additional analysis)"
    echo "  - Figure S3 (Performance analysis)"
    echo "  - Figures S4 & S5 (Time-to-threshold analysis)"
    echo "  - Figure S4 All (Complete time-to-threshold analysis)"
    echo "  - Figure S9 (Parameter analysis)"
    echo "  - Figure S11 (Additional convergence analysis)"
    echo "  - Figure S13 (Comprehensive analysis)"
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
