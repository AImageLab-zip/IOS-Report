#!/bin/bash
# Complete workflow example for auto-captioning system

echo "=================================="
echo "AUTO-CAPTIONING WORKFLOW EXAMPLE"
echo "=================================="
echo ""

cd /work/grana_maxillo/IOS-DraftReport/src/auto-captioning

# Step 1: Initial setup
echo "Step 1: Initial Setup"
echo "---------------------"
./setup.sh
echo ""

# Step 2: Configure API key
echo "Step 2: Configure API Key"
echo "-------------------------"
echo "Edit .env file and add your OpenAI API key:"
echo "  nano .env"
echo ""
echo "Press Enter when done..."
read

# Step 3: Verify setup
echo "Step 3: Verify Setup"
echo "--------------------"
python test_setup.py
echo ""

# Step 4: Test on single patient
echo "Step 4: Test on Single Patient"
echo "-------------------------------"
echo "Testing with patient 201_1979..."
python auto_caption.py --patient 201_1979
echo ""

# Step 5: Review test result
echo "Step 5: Review Test Result"
echo "--------------------------"
python analyze_results.py show 201_1979
echo ""

# Step 6: Process all patients (ask for confirmation)
echo "Step 6: Process All Patients"
echo "----------------------------"
echo "⚠️  WARNING: This will process all patients and cost money!"
echo "Estimated cost: $10-20 with GPT-4o"
echo ""
echo "Do you want to proceed? (yes/no)"
read answer

if [ "$answer" = "yes" ]; then
    echo "Processing all patients..."
    python auto_caption.py
    
    # Step 7: Show statistics
    echo ""
    echo "Step 7: Show Statistics"
    echo "-----------------------"
    python analyze_results.py stats
    
    # Step 8: Export results
    echo ""
    echo "Step 8: Export Results"
    echo "----------------------"
    python analyze_results.py export --output final_descriptions.csv
    
    echo ""
    echo "✓ Complete! Results saved in outputs/ directory"
else
    echo "Skipped processing all patients"
fi

echo ""
echo "=================================="
echo "Workflow completed!"
echo "=================================="
