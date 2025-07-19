# Rutas de include
EIGEN_INC = /usr/include/eigen3
MYINC     = include

# Compilador y flags
CXX       = g++
CXXFLAGS  = -std=c++17 -I$(EIGEN_INC) -I$(MYINC)

# Graficar solo ls frontera
BOUNDARY_SRC = plot_boundary/dump_boundary.cpp
BOUNDARY_BIN = dump_boundary
BOUNDARY_DAT = boundary.dat
BOUNDARY_PY  = plot_boundary/plt_boundary.py

# Graficas de densidad de probabilidad y distribución de fase
COMPUTE_SRC  = plot_dprob_dfase/compute_billiard.cpp
COMPUTE_BIN  = compute_billiard
DENSITY_DAT  = density.dat
PHASE_DAT    = phase.dat
PLOTDP_PY    = plot_dprob_dfase/plot_results.py

# Parámetros por defecto (se pueden sobrescribir en la invocación)
xi0  ?= 3.0
eta0 ?= 2.0
k2   ?= 0.406

.PHONY: all plotb plot clean

# Por defecto solo compila los binarios y vuelca la frontera
all: $(BOUNDARY_BIN) $(BOUNDARY_DAT) $(COMPUTE_BIN)

# 1) Frontera
$(BOUNDARY_BIN): $(BOUNDARY_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Volcar la frontera pasando xi0 y eta0
$(BOUNDARY_DAT): $(BOUNDARY_BIN)
	./$(BOUNDARY_BIN) $(xi0) $(eta0) > $(BOUNDARY_DAT)

# 2) Compute (densidad/fase)
$(COMPUTE_BIN): $(COMPUTE_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Graficar solo frontera. Ej: make plotb xi0=3.0 eta0=2.0
plotb: $(BOUNDARY_DAT)
	python3 $(BOUNDARY_PY)

# Graficar densidad y fase. Ej: make plot xi0=3.0 eta0=2.0 k2=0.406
plot: all
	./$(COMPUTE_BIN) $(xi0) $(eta0) $(k2)
	python3 $(PLOTDP_PY)

# Graficar espectro de T(k)
SPEC_SRC       = plot_spectrum/scan_spectrum.cpp
SPEC_BIN       = scan_spectrum
PLOTSPEC_PY    = plot_spectrum/plot_spectrum.py

$(SPEC_BIN): $(SPEC_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

plotspec: $(SPEC_BIN) #Ej: make plotspec xi0=3.0 eta0=2.0
	./$(SPEC_BIN) $(xi0) $(eta0)
	python3 $(PLOTSPEC_PY)

clean:
	rm -f $(BOUNDARY_BIN) $(BOUNDARY_DAT) \
	      $(COMPUTE_BIN) $(DENSITY_DAT) $(PHASE_DAT) \
	      $(SPEC_BIN) spectrum.dat resonances.dat

