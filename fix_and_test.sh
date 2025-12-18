#!/bin/bash
# Script to apply RL fixes and test the system

set -e  # Exit on error

echo "=========================================="
echo "FIXING RL TRAINING ISSUES"
echo "=========================================="

# 3. Rebuild FMU
echo "3. Rebuilding ThermostatML.fmu..."
cd FMU/ThermostatML && 7z a -tzip -mx=1 ../../FMU/ThermostatML.fmu ./* > /dev/null && cd ../..
echo "   ✓ FMU rebuilt"

# 4. Clean previous runs
echo "4. Cleaning previous runs..."
rm -rf /var/tmp/learning_thermostat build
echo "   ✓ Cleaned"

# 5. Run training
echo "5. Starting training (BC → RL → EVAL)..."
echo "   This will take ~5 minutes..."
make build/stage2/outputs.csv 2>&1 | tee test.log

# 6. Quick analysis
echo ""
echo "=========================================="
echo "QUICK RESULTS ANALYSIS"
echo "=========================================="

python3 << 'EOF'
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('build/stage2/outputs.csv')
    temp = df['{Plant}.PlantInstance.T_bair_out'].values
    time = df['time'].values

    print("\nPhase Results:")
    print("-" * 60)
    
    for phase, (s, e) in [('BC', (0, 3000)), ('RL', (3000, 7000)), ('EVAL', (7000, 10000))]:
        mask = (time >= s) & (time < e)
        t = temp[mask]
        if len(t) == 0:
            continue
        
        rmse = np.sqrt(np.mean((t - 25.5) ** 2))
        violations = np.sum((t < 23.0) | (t > 28.0))
        violations_pct = 100 * violations / len(t)
        
        print(f"{phase:>4}: range=[{t.min():5.1f}, {t.max():5.1f}], "
              f"RMSE={rmse:5.2f}, violations={violations_pct:5.1f}%")
    
    # Improvement calculation
    bc_mask = time < 3000
    eval_mask = time >= 7000
    
    if np.sum(bc_mask) > 0 and np.sum(eval_mask) > 0:
        bc_rmse = np.sqrt(np.mean((temp[bc_mask] - 25.5) ** 2))
        eval_rmse = np.sqrt(np.mean((temp[eval_mask] - 25.5) ** 2))
        improvement = ((bc_rmse - eval_rmse) / bc_rmse) * 100
        
        print("\n" + "=" * 60)
        if improvement > 0:
            print(f"✓ IMPROVEMENT: {improvement:+.1f}% (RL better than BC)")
        else:
            print(f"✗ DEGRADATION: {improvement:+.1f}% (RL worse than BC)")
        
        # Safety check
        max_temp = temp[eval_mask].max()
        if max_temp > 35.0:
            print(f"⚠️  WARNING: Max temp in EVAL = {max_temp:.1f}°C (still too high!)")
        elif max_temp > 30.0:
            print(f"⚠️  CAUTION: Max temp in EVAL = {max_temp:.1f}°C (borderline)")
        else:
            print(f"✓ SAFE: Max temp in EVAL = {max_temp:.1f}°C (within bounds)")
        print("=" * 60)
        
except FileNotFoundError:
    print("ERROR: Could not find build/stage2/outputs.csv")
    print("Training may have failed. Check test.log for details.")
except Exception as e:
    print(f"ERROR during analysis: {e}")

EOF

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Check test.log for detailed training logs"
echo "2. Look for emergency override messages"
echo "3. If successful, proceed with: make dse"
echo "4. If still failing, we may need to:"
echo "   - Increase SAFETY_MARGIN"
echo "   - Decrease epsilon further"
echo "   - Shorten RL phase"
echo "=========================================="
