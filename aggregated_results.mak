.SECONDARY=

MAESTRO_JAR ?= ~/Scaricati/maestro-4.0.0-jar-with-dependencies.jar
MAESTRO ?= java -classpath $(MAESTRO_JAR):/work/model/classpath org.intocps.maestro.Main

ALL_STOCK_FMU = FMU/Controller.fmu FMU/KalmanFilter.fmu FMU/Plant.fmu FMU/Room.fmu FMU/Supervisor.fmu

TEMPS := $(shell seq 30 1 42)
.PHONY = all clean

all: build2/results.csv

build2/results.csv: $(addprefix build2/ml/,$(addsuffix /outputs.csv,$(TEMPS))) $(addprefix build2/baseline/,$(addsuffix /outputs.csv,$(TEMPS))) 
	@echo "TODO I'LL MERGE" $^
	python aggregate.py $^

build2/baseline/%/outputs.csv: mm_cmp_baseline.json simulation-config-cmp.json $(ALL_STOCK_FMU)
	mkdir -p $(dir $@)
	jq '.parameters."{Controller}.ControllerInstance.T_desired" = $* | .parameters."{ThermostatML}.ThermostatMLInstance.T_desired" = $*' $< > $(dir $@)/mm.json
	$(MAESTRO) import sg1 simulation-config-cmp.json $(dir $@)/mm.json -fsp FMU -output $(dir $@)
	$(MAESTRO) interpret $(dir $@)/spec.mabl -tms 10 -output $(dir $@) 2>&1 | tee $(dir $@)/out.txt

build2/ml/%/outputs.csv: mm_cmp_ml.json simulation-config-cmp.json $(ALL_STOCK_FMU)
	mkdir -p $(dir $@)
	jq '.parameters."{Controller}.ControllerInstance.T_desired" = $* | .parameters."{ThermostatML}.ThermostatMLInstance.T_desired" = $*' $< > $(dir $@)/mm.json
	$(MAESTRO) import sg1 simulation-config-cmp.json $(dir $@)/mm.json -fsp FMU -output $(dir $@)
	$(MAESTRO) interpret $(dir $@)/spec.mabl -tms 10 -output $(dir $@) 2>&1 | tee $(dir $@)/out.txt

clean:
	-rm -r build2/*
