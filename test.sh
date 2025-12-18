#!/bin/bash
# Script to apply RL fixes and test the system with proper BC vs RL comparison

set -e  # Exit on error

echo "=========================================="
echo "FIXING RL TRAINING ISSUES"
echo "=========================================="


# 2. Rebuild FMU
echo "2. Rebuilding ThermostatML.fmu..."
cd FMU/ThermostatML && 7z a -tzip -mx=1 ../../FMU/ThermostatML.fmu ./* > /dev/null && cd ../..
echo "   ✓ FMU rebuilt"

# 3. Clean previous runs
echo "3. Cleaning previous runs..."
rm -rf /var/tmp/learning_thermostat build
echo "   ✓ Cleaned"

# 4. Run training
echo "4. Starting training (BC → RL → EVAL with RL policy)..."
echo "   This will take ~5 minutes..."
make build/stage2/outputs.csv 2>&1 | tee test.log

# 5. Quick analysis of training run
echo ""
echo "=========================================="
echo "TRAINING RUN ANALYSIS (with RL policy)"
echo "=========================================="

python3 << 'EOF'
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('build/stage2/outputs.csv')
    temp = df['{Plant}.PlantInstance.T_bair_out'].values
    time = df['time'].values
    
    # Try to get heater status, but continue if not available
    heater_available = False
    heater = None
    for col in ['{ThermostatML}.heater_on_out', '{Controller}.heater_on_out']:
        if col in df.columns:
            heater = df[col].values
            heater_available = True
            break
    
    print("\nPhase Results:")
    print("-" * 70)
    
    phases_data = {}
    for phase, (s, e) in [('BC', (0, 3000)), ('RL', (3000, 7000)), ('EVAL', (7000, time.max()))]:
        mask = (time >= s) & (time < e)
        t = temp[mask]
        
        if len(t) == 0:
            continue
        
        rmse = np.sqrt(np.mean((t - 25.5) ** 2))
        violations = np.sum((t < 23.0) | (t > 28.0))
        violations_pct = 100 * violations / len(t)
        
        result_str = (f"{phase:>4}: range=[{t.min():5.1f}, {t.max():5.1f}], "
                     f"RMSE={rmse:5.2f}, viol={violations_pct:5.1f}%")
        
        # Add energy metrics if heater data available
        if heater_available:
            h = heater[mask]
            switches = np.sum(np.abs(np.diff(h.astype(int))))
            heating_time = np.sum(h > 0.5)
            duty_cycle = 100 * heating_time / len(h)
            result_str += f", switches={switches:3d}, duty={duty_cycle:5.1f}%"
            
            phases_data[phase] = {
                'rmse': rmse,
                'range': (t.min(), t.max()),
                'violations_pct': violations_pct,
                'switches': switches,
                'duty_cycle': duty_cycle
            }
        else:
            phases_data[phase] = {
                'rmse': rmse,
                'range': (t.min(), t.max()),
                'violations_pct': violations_pct
            }
        
        print(result_str)
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    # Check if RL had issues
    if 'RL' in phases_data:
        rl_max = phases_data['RL']['range'][1]
        if rl_max > 30.0:
            print("⚠ WARNING: RL phase shows very high temperatures!")
            print("   This suggests the RL is unstable. Consider:")
            print("   - Reducing epsilon further (try 0.005)")
            print("   - Increasing beta_start (try 10.0)")
            print("   - Reducing RL_PHASE_END (try 5000)")
    
except FileNotFoundError:
    print("ERROR: Could not find build/stage2/outputs.csv")
    print("Training may have failed. Check test.log for details.")
    exit(1)
except Exception as e:
    print(f"ERROR during analysis: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

EOF

echo ""
echo "=========================================="
echo "ANALISI COMPLETA FASI"
echo "=========================================="

python3 << 'EOF'
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('build/stage2/outputs.csv')
    temp = df['{Plant}.PlantInstance.T_bair_out'].values
    time = df['time'].values
    
    print("\nConfronto BC vs RL (basato su fasi training):")
    print("-" * 70)
    
    # BC phase (steady state)
    bc_mask = (time >= 500) & (time < 3000)
    bc_temp = temp[bc_mask]
    bc_rmse = np.sqrt(np.mean((bc_temp - 25.5) ** 2))
    bc_viol = 100 * np.sum((bc_temp < 23.0) | (bc_temp > 28.0)) / len(bc_temp)
    
    print(f"\nBC Policy (steady state [500-3000s]):")
    print(f"  RMSE:       {bc_rmse:.3f}°C")
    print(f"  Range:      [{bc_temp.min():.1f}, {bc_temp.max():.1f}]°C")
    print(f"  Violations: {bc_viol:.1f}%")
    
    # RL phase (last 500s before EVAL - policy stabilized)
    rl_mask = (time >= 6500) & (time < 7000)
    rl_temp = temp[rl_mask]
    rl_rmse = np.sqrt(np.mean((rl_temp - 25.5) ** 2))
    rl_viol = 100 * np.sum((rl_temp < 23.0) | (rl_temp > 28.0)) / len(rl_temp)
    
    print(f"\nRL Policy (stabilized [6500-7000s]):")
    print(f"  RMSE:       {rl_rmse:.3f}°C")
    print(f"  Range:      [{rl_temp.min():.1f}, {rl_temp.max():.1f}]°C")
    print(f"  Violations: {rl_viol:.1f}%")
    
    # EVAL phase (final policy)
    eval_mask = time >= 7000
    eval_temp = temp[eval_mask]
    eval_rmse = np.sqrt(np.mean((eval_temp - 25.5) ** 2))
    eval_viol = 100 * np.sum((eval_temp < 23.0) | (eval_temp > 28.0)) / len(eval_temp)
    
    print(f"\nEVAL Phase (RL final [7000s+]):")
    print(f"  RMSE:       {eval_rmse:.3f}°C")
    print(f"  Range:      [{eval_temp.min():.1f}, {eval_temp.max():.1f}]°C")
    print(f"  Violations: {eval_viol:.1f}%")
    
    # Compare BC vs EVAL (best estimate)
    improvement = ((bc_rmse - eval_rmse) / bc_rmse) * 100
    viol_change = bc_viol - eval_viol
    
    print("\n" + "=" * 70)
    print("BC vs RL COMPARISON (best estimate):")
    print("=" * 70)
    
    if improvement > 0:
        print(f"✓ RMSE improvement: {improvement:+.1f}%")
    else:
        print(f"✗ RMSE degradation: {improvement:+.1f}%")
    
    if viol_change >= 0:
        print(f"✓ Violations:       {viol_change:+.1f}pp better")
    else:
        print(f"✗ Violations:       {viol_change:+.1f}pp worse")
    
    print("\n" + "=" * 70)
    if improvement > 5 and eval_rmse < 1.0:
        print("✓ OTTIMO: RL migliora BC significativamente")
        print("  → Usa RL per DSE")
    elif improvement > 0 and eval_viol < 5:
        print("✓ BUONO: RL migliora BC moderatamente")
        print("  → Usa RL per DSE")
    elif abs(improvement) < 5:
        print("≈ NEUTRALE: RL simile a BC")
        print("  → Usa BC per affidabilità (o RL se preferisci)")
    else:
        print("✗ PROBLEMATICO: RL peggiora BC")
        print("  → Usa BC per DSE")
    print("=" * 70)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

EOF

echo ""
echo "=========================================="
echo "COMPLETATO!"
echo "=========================================="
echo "Policies salvate in: /var/tmp/learning_thermostat/"
echo "  - policy_bc.pth"
echo "  - policy_rl.pth"
echo ""
echo "Prossimo step:"
echo "  make dse    (usa la policy migliore)"
echo "=========================================="