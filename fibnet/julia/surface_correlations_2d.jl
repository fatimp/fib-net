using CorrelationFunctions

const D = CorrelationFunctions.Directional
const U = CorrelationFunctions.Utilities

length(ARGS) == 9 || error(
    "usage: surface_correlations_2d.jl INPUT HEIGHT WIDTH MAX_DISTANCE STEP AXES BOUNDARY FILTER OUTPUT",
)

input_path = ARGS[1]
dims = parse.(Int, ARGS[2:3])
max_distance = parse(Int, ARGS[4])
step = parse(Int, ARGS[5])
axes = split(ARGS[6], ",")
boundary = ARGS[7]
filter_width = parse(Int, ARGS[8])
output_path = ARGS[9]

bytes = read(input_path)
length(bytes) == prod(dims) || error(
    "raw input has $(length(bytes)) bytes; expected $(prod(dims)) for shape $(Tuple(dims))",
)

# Public phase convention: zero is pore/void and one is solid. surf2 receives
# the solid phase explicitly; surfvoid defines void internally as array .== 0.
solid = reshape(bytes, dims...)
mode = boundary == "periodic" ? U.Periodic() : U.NonPeriodic()
filter = U.ConvKernel(filter_width)
directions = Dict(
    "y" => (U.DirX(), 1),
    "x" => (U.DirY(), 2),
)

open(output_path, "w") do output
    println(output, "axis\tdistance\tsample_count\tfss\tfsv")
    for axis in axes
        direction, dimension = directions[axis]
        axis_max = min(max_distance, dims[dimension] - 1)
        result_length = axis_max + 1
        fss = D.surf2(
            solid, UInt8(1), direction;
            len=result_length, mode=mode, filter=filter,
        )
        fsv = D.surfvoid(
            solid, UInt8(1), direction;
            len=result_length, mode=mode, filter=filter,
        )
        for distance in 0:step:axis_max
            samples = if boundary == "periodic"
                prod(dims)
            else
                prod(dims) ÷ dims[dimension] * (dims[dimension] - distance)
            end
            println(output, join((
                axis,
                distance,
                samples,
                repr(fss[distance + 1]),
                repr(fsv[distance + 1]),
            ), '\t'))
        end
    end
end
