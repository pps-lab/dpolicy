
help:
	@echo 'DPolicy Components'
	@echo '  make create-workloads                           - run the workload generator'
	@echo 'DPolicy Evaluation Plots'
	@echo '  make plot-all                                   - recreate all plots from the paper'
	@echo '  make plot-context                               - recreate the subsampling plot from the paper'
	@echo '  make plot-scope                                 - recreate the unlocking plot from the paper'
	@echo '  make plot-timeunit                              - recreate the comparison plot from the paper'
	@echo 'Start Reproducing the Evaluation (using DoE-Suite on your remote machines e.g., AWS)'
	@echo '  make run-context                                - start the REMOTE experiments underlying the context plot'
	@echo '  make run-scope                                  - start the REMOTE experiments underlying the scope plots'
	@echo '  make run-timeunit                               - start the REMOTE experiments underlying the time unit plot'
	@echo 'List Experiment Commands'
	@echo '  make cmd-context                                - list all commands for the context plot'
	@echo '  make cmd-scope                                  - list all commands for the scope plot'
	@echo '  make cmd-timeunit                               - list all commands for the time plot'


##################################################
#   ___                                  _
#  / __|___ _ __  _ __  ___ _ _  ___ _ _| |_ ___
# | (__/ _ \ '  \| '_ \/ _ \ ' \/ -_) ' \  _(_-<
#  \___\___/_|_|_| .__/\___/_||_\___|_||_\__/__/
#                |_|
##################################################

create-workloads:
	@cd workload-simulator && poetry install && poetry run python workload_simulator/main.py --output-dir $(out)/workloads

#############################

# Get the directory where the Makefile resides
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

out:=$(abspath $(MAKEFILE_DIR)/doe-suite-results-super-reproduce)


# results from the paper
context_default="trap-relax=1731504884"
scope_default="trap=1731489444"
timeunit_default="trap-time-threads=1731532838"
perf_default="$(context_default) $(scope_default) $(timeunit_default)"


##############################################################################
#    ___              _         ___ _     _
#   / __|_ _ ___ __ _| |_ ___  | _ \ |___| |_ ___
#  | (__| '_/ -_) _` |  _/ -_) |  _/ / _ \  _(_-<
#   \___|_| \___\__,_|\__\___| |_| |_\___/\__/__/
#
##############################################################################

plot-all: plot-context plot-scope plot-timeunit plot-perf

# TODO: At the moment, need to have all designs for running the super etls. We could split this

plot-context:
	@echo "Building Context Figure 3 (left)"
	@$(MAKE) run_super_etl default=$(perf_default) config=trap-combined pipelines="preprocess context"
	@echo "Finished creating context plot: $(out)"
#	@$(MAKE) latex-pdf-unlocking-fig6

plot-scope:
	@echo "Building Scope Figure 3 (center) + Figure 4"
	@$(MAKE) run_super_etl default=$(perf_default) config=trap-combined pipelines="preprocess scope_category scope_attribute"
	@echo "Finished creating scope plots: $(out)"
#	@$(MAKE) latex-pdf-comparison-fig7


plot-timeunit:
	@echo "Building time-based Privacy Unit Figure 3 (right)"
	@$(MAKE) run_super_etl default=$(perf_default) config=trap-combined pipelines="preprocess time_privacy_unit"
	@echo "Finished creating time privacy unit plot: $(out)"
#	@$(MAKE) latex-pdf-unlocking-fig6


plot-perf: # TODO: could improve visualization
	echo "Building Performance Data"
	@$(MAKE) run_super_etl default=$(perf_default) config=trap-perf
	echo "Finished creating perf plot: $(out)"

############################

# uses the doe-suite super etl functionality to build the plots based on the results
run_super_etl:
	@echo "Enter a result id:"; \
    read -p '  [default:$(default)] ' custom; \
	cd doe-suite && $(MAKE) etl-super config=$(config) out=$(out) custom-suite-id="$${custom:-$(default)}";

# Builds the figure as present in the paper
#latex-pdf-%:
#	cp doe-suite-config/super_etl/aux/$*.tex $(out)
#	cd $(out) && latexmk -pdf && latexmk -c
#	find $(out) -type f -name '*.fdb_latexmk' -delete
#	find $(out) -type f -name '*.synctex.gz' -delete
#	rm $(out)/$*.tex


###########################################################################
#   ___             ___                   _               _
#  | _ \_  _ _ _   | __|_ ___ __  ___ _ _(_)_ __  ___ _ _| |_ ___
#  |   / || | ' \  | _|\ \ / '_ \/ -_) '_| | '  \/ -_) ' \  _(_-<
#  |_|_\\_,_|_||_| |___/_\_\ .__/\___|_| |_|_|_|_\___|_||_\__/__/
#                          |_|
###########################################################################


suite_id?=new # alternative: last
RUN=cd doe-suite && $(MAKE) run suite=$(suite) id=$(suite_id)

run-context:  # option to continue   (use make figure5-run suite_id=last to fetch the last run)
	echo "Starting Experiments for Context (Figure 3 left)"
	$(eval suite := trap-relax)
	$(RUN)

run-scope:
	echo "Starting Experiments for Scope (Figure 3 middle + Figure 4)"
	$(eval suite := trap)
	$(RUN)

run-timeunit:
	echo "Starting Experiments for time-based Privacy Unit (Figure 3 right)"
	$(eval suite := trap-time-threads)
	$(RUN)


#################################################
#   ___                              _
#  / __|___ _ __  _ __  __ _ _ _  __| |___
# | (__/ _ \ '  \| '  \/ _` | ' \/ _` (_-<
#  \___\___/_|_|_|_|_|_\__,_|_||_\__,_/__/
#
#################################################


CMD=cd doe-suite && $(MAKE) design suite=$(suite)

cmd-context:
	$(eval suite := trap-relax)
	$(CMD)

cmd-scope:
	$(eval suite := trap)
	$(CMD)

cmd-timeunit:
	$(eval suite := trap-time-threads)
	$(CMD)

############################