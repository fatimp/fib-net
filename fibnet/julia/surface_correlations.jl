using CorrelationFunctions

const D = CorrelationFunctions.Directional
const U = CorrelationFunctions.Utilities

length(ARGS) == 10 || error(
    "usage: surface_correlations.jl INPUT Z Y X MAX_DISTANCE STEP AXES BOUNDARY FILTER OUTPUT",
)

input_path = ARGS[1]
dims = parse.(Int, ARGS[2:4])
max_distance = parse(Int, ARGS[5])
step = parse(Int, ARGS[6])
axes = split(ARGS[7], ",")
boundary = ARGS[8]
filter_width = parse(Int, ARGS[9])
output_path = ARGS[10]

expected_length = prod(dims)
bytes = read(input_path)
length(bytes) == expected_length || error(
    "raw input has $(length(bytes)) bytes; expected $expected_length for shape $(Tuple(dims))",
)
solid = reshape(bytes, dims...)
mode = boundary == "periodic" ? U.Periodic() : U.NonPeriodic()
filter = U.ConvKernel(filter_width)

directions = Dict(
    "z" => (U.DirX(), 1),
    "y" => (U.DirY(), 2),
    "x" => (U.DirZ(), 3),
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
                div(prod(dims), dims[dimension]) * (dims[dimension] - distance)
            end
            println(
                output,
                join(
                    (axis, distance, samples, repr(fss[distance + 1]), repr(fsv[distance + 1])),
                    "\t",
                ),
            )
        end
    end
end
