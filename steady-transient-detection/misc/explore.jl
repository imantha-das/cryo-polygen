begin 
    using Pkg 
    Pkg.activate(
    "julia-env/steady-trans"
    )
end 

# Load Packages
begin 
    import Cairo
    using Gadfly:plot, Geom, draw, SVG, set_default_plot_size, Theme, layer,mm, style
    using XLSX
    using DataFrames
    using StatsBase:autocor
    using GLM:fit, LinearModel, @formula, predict, coeftable
end

set_default_plot_size(25cm,15cm)

# Read Data
xf = XLSX.readxlsx("data/power_minute_data.xlsx")
data = xf["Sheet1"]["A2:A43201"]
df = DataFrame(data, :auto)
rename!(df, "x1" => :power_consumption)

p1 = plot(
    x = 1:nrow(df),
    y = df.power_consumption,
    Geom.line
)

p2 = plot(
    x = df.power_consumption,
    Geom.density
)

#draw(SVG("figs/power_consume.svg", 4inch, 3inch), p)

# Power consumption values
X = convert(Vector{Float64}, df.power_consumption)

mutable struct lr_results
    t_value::Vector{Float64}
    t_test::Vector{Float64}
end


function sliding_window(X, ws)
    t_value = Vector{Float64}([])
    t_test = Vector{Float64}([])

    for i = 1+ws:length(X) - ws
        # Get window slice
        X_slc = get_slice(X,idx = i, ws = ws)
        # Construct a fataframe for prediction
        df = DataFrame(Dict(:x => 1:length(X_slc), :y => X_slc))
        # Fit linear Linear 
        lr = fit(LinearModel, @formula(y ~ x), df)
        # Make Prediction if necessary
        #ŷ = predict(lr)
        # Get tvalue
        push!(t_value, coeftable(lr).cols[:3][2])
        push!(t_test,coeftable(lr).cols[:4][2])    
    end

    return t_value, t_test
end

@doc """ 
Returns a slice of the input data
""" -> 
get_slice(X::Vector{Float64};idx::Int64,ws::Int64)::Vector{Float64} = X[idx:idx+ws]


@doc """
Plotting function 
""" ->
function plot_slice(y::Vector{Float64},ŷ::Vector{Float64})::Plot 
    plot(
        layer(x = 1:length(y), y = y, Geom.line, Theme(default_color = "dodgerblue",line_width = 1mm)),
        layer(x = 1:length(ŷ), y = ŷ, Geom.line, Theme(default_color = "indianred", line_width = 1mm))
    )
end

# Applying linear regression on sliding window and compute t_test
t_value, t_test = sliding_window(X, 20)


p1 = plot(x = 1: length(X), y = X, Geom.line)
p2 = plot(x = 1:length(t_test),y = t_test,Geom.line)

p3 = vstack(p1,p2)
p4 = plot(x = t_test,Geom.density)










