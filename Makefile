_dummy := $(shell mkdir -p build/cmp)

#ZIP_UTIL=zip -r
ZIP_UTIL=7z a -tzip -mx=1

MAESTRO_JAR ?= ~/Scaricati/maestro-4.0.0-jar-with-dependencies.jar
MAESTRO ?= java -classpath $(MAESTRO_JAR):/work/model/classpath org.intocps.maestro.Main
LIVE_LOSS ?= y # Requires Gnuplot

# Override like: make TMS=1 ...
TMS ?= 10
THZ ?= 1

ALL_STOCK_FMU = FMU/Controller.fmu FMU/KalmanFilter.fmu FMU/Plant.fmu FMU/Room.fmu FMU/Supervisor.fmu
GRAPHS = build/g_env.pdf build/g_loss.pdf build/g_act.pdf
DSE_TEMPS = $(shell cat temperatures)

all: build/report.csv $(GRAPHS) dse build/cmp/result.csv
dse: $(addprefix build/dse/,$(addsuffix /report.csv,$(DSE_TEMPS)))

.SUFFIXES:
.NOTPARALLEL:
.PHONY: all dse rl_dse_train rl_dse_eval clean
.PRECIOUS: build/dse/%/stage2/spec.mabl build/dse/%/spec.mabl build/dse/%/mm_param.json build/dse/%/

FMU/Controller.fmu: $(shell find FMU/Controller -print)
	@rm -f $@
	cd FMU/Controller && $(ZIP_UTIL) ../../$@ ./* > /dev/null

FMU/KalmanFilter.fmu: $(shell find FMU/KalmanFilter -print)
	@rm -f $@
	cd FMU/KalmanFilter && $(ZIP_UTIL) ../../$@ ./* > /dev/null

FMU/Plant.fmu: $(shell find FMU/Plant -print)
	@rm -f $@
	cd FMU/Plant && $(ZIP_UTIL) ../../$@ ./* > /dev/null

FMU/Room.fmu: $(shell find FMU/Room -print)
	@rm -f $@
	cd FMU/Room && $(ZIP_UTIL) ../../$@ ./*  > /dev/null

FMU/Supervisor.fmu: $(shell find FMU/Supervisor -print)
	@rm -f $@
	cd FMU/Supervisor && $(ZIP_UTIL) ../../$@ ./* > /dev/null

FMU/ThermostatML.fmu:
	@rm -f $@
	cd FMU/ThermostatML && $(ZIP_UTIL) ../../$@ ./* > /dev/null

$(GRAPHS): build/stage2/outputs.csv
	python3 plot.py

build/report.csv: build/stage2/outputs.csv build/outputs.csv
	touch build/report.csv

# MAIN (baseline stage1 -> transition stage2)
build/outputs.csv build/stage2/outputs.csv /var/tmp/learning_thermostat/thermostat_nn_model.pt: build/spec.mabl build/stage2/spec.mabl
	@[ "y" = $(LIVE_LOSS) ] && echo "======= LIVE PLOTTING _NOT_ IMPLEMENTED! ========="
	$(MAESTRO) interpret build/spec.mabl -tms $(TMS) -thz $(THZ) -transition build/stage2 -output build 2>&1 | tee build/out.txt

build/stage2/spec.mabl: $(ALL_STOCK_FMU) FMU/ThermostatML.fmu mm2.json simulation-config.json
	$(MAESTRO) import sg1 simulation-config.json mm2.json -fsp FMU -output build/stage2

build/spec.mabl: $(ALL_STOCK_FMU) mm1.json simulation-config.json
	$(MAESTRO) import sg1 simulation-config.json mm1.json -fsp FMU -output build

# ===========================
# DSE (baseline, uses transfer transition)
# ===========================

build/dse/%/report.csv: build/dse/%/ | build/dse/%/spec.mabl build/dse/%/stage2/spec.mabl
	@echo "Running simulation for $*"
	$(MAESTRO) interpret build/dse/$*/spec.mabl -tms $(TMS) -thz $(THZ) -transition build/dse/$*/stage2 -output build/dse/$*/ 2>&1 | tee build/dse/$*/out.txt
	touch $@

build/dse/%/:
	mkdir -p $@

build/dse/%/mm_param.json:
	sed "s/%TEMP%/$*/g" mm_dse.template.json > build/dse/$*/mm_param.json

build/dse/%/stage2/spec.mabl: $(ALL_STOCK_FMU) FMU/ThermostatML.fmu build/dse/%/mm_param.json mm2.json simulation-config.json
	$(MAESTRO) import sg1 build/dse/$*/mm_param.json simulation-config.json mm2.json -fsp FMU -output build/dse/$*/stage2

build/dse/%/spec.mabl: $(ALL_STOCK_FMU) build/dse/%/mm_param.json mm1.json simulation-config.json
	$(MAESTRO) import sg1 build/dse/$*/mm_param.json simulation-config.json mm1.json -fsp FMU -output build/dse/$*

# ==================================================
# RL: standalone multi-model (no transfers / no swaps)
# ==================================================

# Generate mm_rl.json from mm2.json:
# - removes modelSwaps/modelTransfers (so no transfer spec, no START_TIME issues)
# - rewires Plant/Supervisor/KF heater_on_in to ThermostatML.heater_on_out
# - keeps Controller.heater_on_out only as teacher into ThermostatML.heater_on_in
mm_rl.json: mm2.json
	python3 -c 'import json,sys; src=sys.argv[1]; dst=sys.argv[2]; mm=json.load(open(src)); mm.pop("modelSwaps",None); mm.pop("modelTransfers",None); conns=mm.get("connections",{}); ctrl_out="{Controller}.ControllerInstance.heater_on_out"; ml_out="{ThermostatML}.ThermostatMLInstance.heater_on_out"; ml_teacher_in="{ThermostatML}.ThermostatMLInstance.heater_on_in"; plant_in="{Plant}.PlantInstance.heater_on_in"; sup_in="{Supervisor}.SupervisorInstance.heater_on_in"; kf_in="{KalmanFilter}.KalmanFilterInstance.heater_on_in"; remove=set([plant_in,sup_in,kf_in]); t=list(conns.get(ctrl_out,[])); t=[x for x in t if x not in remove]; (ml_teacher_in in t) or t.append(ml_teacher_in); conns[ctrl_out]=t; u=list(conns.get(ml_out,[])); [u.append(x) for x in [plant_in,sup_in,kf_in] if x not in u]; conns[ml_out]=u; extra={"{Supervisor}.SupervisorInstance.LL_out":"{ThermostatML}.ThermostatMLInstance.LL_in","{Supervisor}.SupervisorInstance.UL_out":"{ThermostatML}.ThermostatMLInstance.UL_in","{Supervisor}.SupervisorInstance.H_out":"{ThermostatML}.ThermostatMLInstance.H_in","{Supervisor}.SupervisorInstance.C_out":"{ThermostatML}.ThermostatMLInstance.C_in"}; [conns.setdefault(s,[]).append(tgt) for s,tgt in extra.items() if tgt not in conns.setdefault(s,[])]; mm["connections"]=conns; json.dump(mm,open(dst,"w"),indent=2,sort_keys=True); print("Wrote",dst)' $< $@

# ===========================
# DSE with RL (SAC) online
# ===========================

RL_DSE_TEMPS = $(DSE_TEMPS)

# RL TRAIN
rl_dse_train: mm_rl.json $(addprefix build/rl_dse_train/,$(addsuffix /report.csv,$(RL_DSE_TEMPS)))

build/rl_dse_train/%/:
	mkdir -p $@

build/rl_dse_train/%/mm_param.json:
	sed "s/%TEMP%/$*/g" mm_dse_rl.template.json > build/rl_dse_train/$*/mm_param.json

build/rl_dse_train/%/spec.mabl: $(ALL_STOCK_FMU) FMU/ThermostatML.fmu build/rl_dse_train/%/mm_param.json mm_rl.json simulation-config.json
	$(MAESTRO) import sg1 build/rl_dse_train/$*/mm_param.json simulation-config.json mm_rl.json -fsp FMU -output build/rl_dse_train/$*

build/rl_dse_train/%/report.csv: build/rl_dse_train/%/ | build/rl_dse_train/%/spec.mabl
	@echo "Running RL TRAIN simulation for $*"
	$(MAESTRO) interpret build/rl_dse_train/$*/spec.mabl -tms $(TMS) -thz $(THZ) -output build/rl_dse_train/$*/ 2>&1 | tee build/rl_dse_train/$*/out.txt
	touch $@

# RL EVAL
rl_dse_eval: mm_rl.json $(addprefix build/rl_dse_eval/,$(addsuffix /report.csv,$(RL_DSE_TEMPS)))

build/rl_dse_eval/%/:
	mkdir -p $@

build/rl_dse_eval/%/mm_param.json:
	sed "s/%TEMP%/$*/g" mm_dse_eval.template.json > build/rl_dse_eval/$*/mm_param.json

build/rl_dse_eval/%/spec.mabl: $(ALL_STOCK_FMU) FMU/ThermostatML.fmu build/rl_dse_eval/%/mm_param.json mm_rl.json simulation-config.json
	$(MAESTRO) import sg1 build/rl_dse_eval/$*/mm_param.json simulation-config.json mm_rl.json -fsp FMU -output build/rl_dse_eval/$*

build/rl_dse_eval/%/report.csv: build/rl_dse_eval/%/ | build/rl_dse_eval/%/spec.mabl
	@echo "Running RL EVAL simulation for $*"
	$(MAESTRO) interpret build/rl_dse_eval/$*/spec.mabl -tms $(TMS) -thz $(THZ) -output build/rl_dse_eval/$*/ 2>&1 | tee build/rl_dse_eval/$*/out.txt
	touch $@

# ===========================
# targets for comparison
# ===========================

build/cmp/baseline/spec.mabl: mm_cmp_baseline.json simulation-config-cmp.json $(ALL_STOCK_FMU)
	mkdir -p build/cmp/baseline
	$(MAESTRO) import sg1 simulation-config-cmp.json $< -fsp FMU -output build/cmp/baseline

build/cmp/baseline/outputs.csv: build/cmp/baseline/spec.mabl
	@[ "y" = $(LIVE_LOSS) ] && echo "======= LIVE PLOTTING _NOT_ IMPLEMENTED! ========="
	$(MAESTRO) interpret $< -tms $(TMS) -output build/cmp/baseline 2>&1 | tee build/cmp/baseline/out.txt

build/cmp/ml/spec.mabl: mm_cmp_ml.json simulation-config-cmp.json $(ALL_STOCK_FMU)
	mkdir -p build/cmp/ml
	$(MAESTRO) import sg1 simulation-config-cmp.json $< -fsp FMU -output build/cmp/ml

build/cmp/ml/outputs.csv: build/cmp/ml/spec.mabl /var/tmp/learning_thermostat/thermostat_nn_model.pt
	@[ "y" = $(LIVE_LOSS) ] && echo "======= LIVE PLOTTING _NOT_ IMPLEMENTED! ========="
	$(MAESTRO) interpret $< -tms $(TMS) -output build/cmp/ml 2>&1 | tee build/cmp/ml/out.txt

build/cmp/result.csv: build/cmp/ml/outputs.csv build/cmp/baseline/outputs.csv merge_cmp.py
	python3 merge_cmp.py build/cmp/baseline/outputs.csv build/cmp/ml/outputs.csv $@

clean:
	rm -rf build
	rm -f FMU/*.fmu
	rm -f mm_rl.json
	rm -rf /var/tmp/learning_thermostat
