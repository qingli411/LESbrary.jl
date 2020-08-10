# # Free convection

# This script runs a simulation of convection driven by cooling at the 
# surface of an idealized, stratified, rotating ocean surface boundary layer.

using LESbrary, Random, Printf, Statistics

LESbrary.Utils.select_device!(4)

# Domain

using Oceananigans.Grids

grid = RegularCartesianGrid(size=(256, 256, 256), x=(0, 256), y=(0, 256), z=(-128, 0))

# Buoyancy and boundary conditions

using Oceananigans.Buoyancy, Oceananigans.BoundaryConditions

Qᵇ = 1e-7
N² = 1e-5

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

  Qᶿ = Qᵇ / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)
dθdz = N² / (buoyancy.gravitational_acceleration * buoyancy.equation_of_state.α)

θ_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dθdz))

# LES Model

# Wall-aware AMD model constant which is 'enhanced' near the upper boundary.
# Necessary to obtain a smooth temperature distribution.
using LESbrary.NearSurfaceTurbulenceModels: SurfaceEnhancedModelConstant

Cᴬᴹᴰ = SurfaceEnhancedModelConstant(grid.Δz, C₀ = 1/12, enhancement = 7, decay_scale = 4 * grid.Δz)

# Instantiate Oceananigans.IncompressibleModel

using Oceananigans

using CUDA: has_cuda

model = IncompressibleModel(architecture = has_cuda() ? GPU() : CPU(),
                                    grid = grid,
                                 tracers = (:T,),
                                buoyancy = buoyancy,
                                coriolis = FPlane(f=1e-4),
                                 closure = AnisotropicMinimumDissipation(C=Cᴬᴹᴰ),
                     boundary_conditions = (T=θ_bcs,))

# # Initial condition

Ξ(z) = randn() * exp(z / 8)

θᵢ(x, y, z) = dθdz * z + 1e-6 * Ξ(z) * dθdz * grid.Lz

set!(model, T=θᵢ)

# # Prepare the simulation

using Oceananigans.Utils: hour, minute
using LESbrary.Utils: SimulationProgressMessenger

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=0.2, Δt=1e-1, max_change=1.1, max_Δt=10.0)

# Calculate stop time as time when boundary layer depth is h = Lz/2.
# Uses a conservative estimate based on
#
#   h ∼ √(2 * Qᵇ * stop_time / N²)

h = grid.Lz/2 # end boundary layer depth is half domain depth
stop_time = 1/2 * h^2 * N² / Qᵇ

simulation = Simulation(model, Δt=wizard, stop_time=stop_time, iteration_interval=100, 
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

using Oceananigans.Utils: GiB
using Oceananigans.OutputWriters: FieldOutputs, JLD2OutputWriter
using LESbrary.Statistics: horizontal_averages

prefix = @sprintf("free_convection_Qb%.1e_Nsq%.1e_Nz%d", Qᵇ, N², grid.Nz)

data_directory = joinpath(@__DIR__, "..", "data", prefix) # save data in /data/prefix

# Copy this file into the directory with data
mkpath(data_directory)
cp(@__FILE__, joinpath(data_directory, basename(@__FILE__)), force=true)

# Three-dimensional field output
fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, FieldOutputs(merge(model.velocities, model.tracers)); 
                            force = true,
                    time_interval = 4hour, # every quarter period
                     max_filesize = 2GiB,
                              dir = data_directory,
                           prefix = prefix * "_fields")

# Horizontal averages
simulation.output_writers[:averages] =
    JLD2OutputWriter(model, LESbrary.Statistics.first_through_third_order(model); 
                             force = true, 
                     time_interval = 10minute, # every quarter period
                               dir = data_directory,
                            prefix = prefix * "_statistics")

# # Run

LESbrary.Utils.print_banner(simulation)

run!(simulation)

exit() # Release GPU memory
