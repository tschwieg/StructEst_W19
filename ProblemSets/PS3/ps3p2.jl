using CSVFiles
using DataFrames
using Plots
using Distributions
using LinearAlgebra
using Optim
pyplot()

macroData = DataFrame(load("data/MacroSeries.csv", header_exists=false, colnames=["C", "K", "W", "R"]))

function ConstructMoments( moments::Vector{Real}, α::Real, ρ::Real,
                           μ::Real, β::Float64, c::Vector{Float64},
                           k::Vector{Float64}, w::Vector{Float64},
                           r::Vector{Float64}, W::Matrix{Real})
    # $r_t - \alpha \exp ( z_t ) k_t^{\alpha - 1} = 0$
    # $\log r_t = \log \alpha + z_t + (\alpha - 1) \log k_t$
    # $z_t = \log r_t - \log \alpha - (\alpha - 1) \log k_t$
    N = 100
    z = Vector{Real}(N)
    for i in 1:N
        z[i] = log( r[i]) - log(α) - (α - 1.0)*log( k[i] )
    end

    moments[1] = mean( z[i+1]- ρ*z[i] - (1-ρ)*μ for i in 1:99)
    moments[2] = mean( (z[i+1]- ρ*z[i] - (1-ρ)*μ)*z[i] for i in 1:99)
    moments[3] = mean( β * α * exp(z[i+1]) * k[i+1]^(α-1.0) * (c[i]/c[i+1]) - 1 for i in 1:99)
    moments[4] = mean( (β*α*exp(z[i+1])*k[i+1]^(α-1.0) * (c[i]/c[i+1]) - 1)*w[i] for i in 1:99 )

    return sum(moments[i]*moments[i] for i in 1:4)
end

W = convert(Matrix{Real}, Diagonal([1.0,1.0,1.0,1.0]))

c = convert( Vector{Float64}, macroData[:C] )
w = convert( Vector{Float64}, macroData[:W] )
k = convert( Vector{Float64}, macroData[:K] )
r = convert( Vector{Float64}, macroData[:R] )

mom = Vector{Real}(4)

function limitedLogistic( unbounded::Real )
    return ((exp(unbounded)) / ( 1 + exp(unbounded)))*.99 + .005
end

function invertLogistic( x::Real )
    return log( (1.0-200.0*x)/ (200.0*x - 199.0))
end


#Initialize this guy with the stuff we got the first time around
alphaStart = invertLogistic(.70216)
rhoStart = atanh(.47972)
muStart = log(5.0729)
betaStart = invertLogistic(.99)


θ = [alphaStart, rhoStart, muStart]

f(x::Vector) = ConstructMoments( mom, limitedLogistic(x[1]), tanh(x[2]), exp(x[3]), .99, c, k, w, r, W )

result = optimize( f, θ, Newton(), autodiff = :forward)

alphaHat = limitedLogistic( result.minimizer[1])
rhoHat = tanh( result.minimizer[2])
muHat = exp( result.minimizer[3])



