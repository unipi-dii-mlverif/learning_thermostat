_dummy := $(shell mkdir -p build)

#ZIP_UTIL=zip -r
ZIP_UTIL=7z a -tzip -mx=1

MAESTRO ?= java -classpath ~/Scaricati/maestro-4.0.0-jar-with-dependencies.jar:/work/model/classpath org.intocps.maestro.Main
LIVE_LOSS ?= y # Requires Gnuplot

ALL_STOCK_FMU = FMU/Controller.fmu FMU/KalmanFilter.fmu FMU/Plant.fmu FMU/Room.fmu FMU/Supervisor.fmu
GRAPHS = build/g_env.pdf build/g_loss.pdf build/g_act.pdf

all: build/report.csv $(GRAPHS)

.SUFFIXES:
.PHONY: all clean

FMU/ThermostatML.fmu: $(shell find FMU/ThermostatML -print)
	@rm -f $@ 
	cd FMU/ThermostatML && $(ZIP_UTIL) ../../$@ ./* > /dev/null

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

$(GRAPHS): build/stage2/outputs.csv
	python3 plot.py

build/report.csv: build/stage2/outputs.csv build/outputs.csv
# TODO merge csv?
	touch build/report.csv

build/outputs.csv build/stage2/outputs.csv: build/spec.mabl build/stage2/spec.mabl
	@[ "y" = $(LIVE_LOSS) ] && echo "======= LIVE PLOTTING _NOT_ IMPLEMENTED! =========" 
	$(MAESTRO) interpret build/spec.mabl -tms 10 -thz 1 -transition build/stage2 -output build 2>&1 | tee build/out.txt

build/stage2/spec.mabl: $(ALL_STOCK_FMU) FMU/ThermostatML.fmu mm2.json simulation-config.json
	$(MAESTRO) import sg1 simulation-config.json mm2.json -fsp FMU -output build/stage2

build/spec.mabl: $(ALL_STOCK_FMU) mm1.json simulation-config.json
	$(MAESTRO) import sg1 simulation-config.json mm1.json -fsp FMU -output build

clean:
	rm -rf build
	rm -f FMU/*.fmu

