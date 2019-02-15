using CSVFiles
using DataFrames
#using Plots
using Distributions
using StatsBase
using LinearAlgebra
using Random
using Optim
using ForwardDiff
#pyplot()

Random.seed!(235711)

macroData = DataFrame(load("data/NewMacroSeries.csv", header_exists=false, colnames=["C", "K", "W", "R", "Y"]))

S = 1000
T = 100
u = rand(Uniform(0,1),S,T)

ϵ = Matrix{Real}(undef,S,T)
z = Matrix{Real}(undef,S,T+1)
k = Matrix{Real}(undef,S,T+1)
w = Matrix{Real}(undef,S,T)
r = Matrix{Real}(undef,S,T)
c = Matrix{Real}(undef,S,T)
y = Matrix{Real}(undef,S,T)

function BuildSim( μ::Real, α::Real, ρ::Real, σ::Real, β::Real,
                   S::Int64, T::Int64, initK::Real, c::Matrix{Real},
                   k::Matrix{Real}, w::Matrix{Real},
                   r::Matrix{Real}, y::Matrix{Real}, z::Matrix{Real},
                   u::Matrix{Float64})
    ϵ = σ*quantile.( Normal(), u)

    for s in 1:S
        z[s,1] = μ
        k[s,1] = initK#mean(macroData[:K])
        for t in 1:T
            #We have z shift up by one to deal with the fact we are
            #1-indexed.
            #z[1] = $z_0$
            z[s,t+1] = ρ*z[s,t] + (1-ρ)*μ + ϵ[s,t]
            k[s,t+1] = α*β*exp(z[s,t+1])*k[s,t]^α
            w[s,t] = (1-α)*exp(z[s,t+1])*k[s,t]^α
            r[s,t] = α*exp(z[s,t+1])*k[s,t]^(α-1)
            c[s,t] = r[s,t]*k[s,t] + w[s,t] - k[s,t+1]
            y[s,t] = exp(z[s,t+1])*k[s,t]^α
            #println(t)
        end
    end
end

function myVar( x::Vector{Real})
    return (sum( x[i]*x[i] for i in 1:100 ) - sum(x)*sum(x) / 100.0) / 99.0
end



function BuildMoments( dC::Vector{Float64}, dK::Vector{Float64},
                       dW::Vector{Float64}, dR::Vector{Float64},
                       dY::Vector{Float64}, sC::Matrix{Real},
                       sK::Matrix{Real}, sW::Matrix{Real},
                       sR::Matrix{Real}, sY::Matrix{Real},
                       momentBox::Vector{Real},S::Int64 )
    #momentBox[1] = mean(dC) - mean(mean(sC,2))
    #println("test")
    momentBox[1] = ( mean(mean(sC[i,:] for i in 1:S)) - mean(dC)) / mean(dC)
    #println("1")
    momentBox[2] = (mean(mean(sK[i,:] for i in 1:S))- mean(dK) ) / mean(dK)
    #println("2")
    momentBox[3] = (mean( mean(sC[i,:] ./ sY[i,:] for i in 1:S)) - mean( dC ./ dY) ) / mean( dC ./ dY)
    momentBox[4] = (mean([myVar(sY[i,:]) for i in 1:S]) - var( dY) ) / var(dY)
    #momentBox[5] = autocor(dC)[2] - mean([autocor(sC[i,:])[2] for i in 1:S])
    momentBox[5] = ( mean([cor(sC[i,1:99],sC[i,2:100]) for i in 1:S]) - cor(dC[1:99],dC[2:100])) / cor(dC[1:99],dC[2:100])
    momentBox[6] = (mean( [cor(sC[i,:],sK[i,1:100]) for i in 1:S] ) - cor(dC,dK)) / cor(dC,dK)
end




function objective(μ::Real, α::Real, ρ::Real, σ::Real, β::Real,
                   S::Int64, T::Int64, initK::Real, c::Matrix{Real},
                   k::Matrix{Real}, w::Matrix{Real},
                   r::Matrix{Real}, y::Matrix{Real}, z::Matrix{Real},
                   u::Matrix{Float64}, dC::Vector{Float64}, dK::Vector{Float64},
                   dW::Vector{Float64}, dR::Vector{Float64},
                   dY::Vector{Float64}, W::Matrix{Real} )

    m = Moments( μ, α, ρ, σ, β, S, T, initK, c, k, w, r, y, z, u, dC, dK, dW, dR, dY, W )
    return dot( m, W*m)#sum( m[i]*m[i] for i in 1:6)
end

function Moments(μ::Real, α::Real, ρ::Real, σ::Real, β::Real,
                   S::Int64, T::Int64, initK::Real, c::Matrix{Real},
                   k::Matrix{Real}, w::Matrix{Real},
                   r::Matrix{Real}, y::Matrix{Real}, z::Matrix{Real},
                   u::Matrix{Float64}, dC::Vector{Float64}, dK::Vector{Float64},
                   dW::Vector{Float64}, dR::Vector{Float64},
                 dY::Vector{Float64}, W::Matrix{Real} )
    BuildSim( μ, α, ρ, σ, β, S, T, initK, c, k, w, r, y, z, u)
    m = Vector{Real}(undef,6)
    BuildMoments( dC, dK, dW, dR, dY, c, k, w, r, y, m, S)
    return m
end

    
    

function LogitTransform( x::Real, upper::Float64, lower::Float64 )
    return lower + (exp(x) / (1.0 + exp(x)))*(upper-lower)
end

function InverLogit( x::Real, upper::Float64, lower::Float64)
    return log( (x - lower) / (upper - x))
end

W = Matrix{Real}(undef,6,6)
W .= 0
for i in 1:6
    W[i,i] = 1.0
end



f(x) = objective( LogitTransform(x[1],5.0, 14.0),
                  LogitTransform(x[2], .01, .99),
                  LogitTransform(x[3], -.99, .99),
                  LogitTransform(x[4], 0.01, 1.1),
                  .99, S, T, mean(macroData[:K]), c, k, w, r, y, z, u, macroData[:C], macroData[:K], macroData[:W], macroData[:R], macroData[:Y], W)

# inner_optimizer = BFGS()
# results = optimize(f, lx, ux, [5.0729,.70216,.47972,.05], Fminbox(inner_optimizer), autodiff=:forward)

θ = [ InverLogit(5.0729,5.0, 14.0),
      InverLogit(.70216, .01, .99),
      InverLogit(.47972, -.99, .99),
      InverLogit(.05, 0.01, 1.1) ]

results = optimize(f, θ, Newton(), autodiff=:forward)

x = results.minimizer

#μ, α, ρ, σ
answer = [LogitTransform(x[1],5.0, 14.0),
          LogitTransform(x[2], .01, .99),
          LogitTransform(x[3], -.99, .99),
          LogitTransform(x[4], 0.01, 1.1)]

m(x) = Moments( x[1], x[2], x[3], x[4], .99, S, T,  mean(macroData[:K]), c, k, w, r, y, z, u, macroData[:C], macroData[:K], macroData[:W], macroData[:R], macroData[:Y], W)

mom = m(answer)

J = ForwardDiff.jacobian( m, answer )

varMat = (1/S)*inv( J' * W*J)
stdErrors = [sqrt(varMat[i,i]) for i in 1:4]

dC = macroData[:C]
dK = macroData[:K]
dW = macroData[:W]
dR = macroData[:R]
dY = macroData[:Y]

E = Matrix{Real}(undef,6,S)
for i in 1:S
    #println(i)
    E[1,i] = (mean(c[i,:]) - mean(dC)) / mean(dC)
    E[2,i] = (mean(k[i,:])- mean(dK) ) / mean(dK)
    E[3,i] = (mean(c[i,:] ./ y[i,:] ) - mean( dC ./ dY) ) / mean( dC ./ dY)
    E[4,i] = (myVar(y[i,:]) - var( dY) ) / var(dY)
    E[5,i] = ( cor(c[i,1:99],c[i,2:100])-cor(dC[1:99],dC[2:100])) / cor(dC[1:99],dC[2:100])
    E[6,i] = (cor(c[i,:],k[i,1:100]) - cor(dC,dK)) / cor(dC,dK)
end


wHat = convert( Matrix{Real},inv((1/S)*sum( E[:,i]*E[:,i]' for i in 1:S)))

fOpt(x) = objective( LogitTransform(x[1],5.0, 14.0),
                  LogitTransform(x[2], .01, .99),
                  LogitTransform(x[3], -.99, .99),
                  LogitTransform(x[4], 0.01, 1.1),
                  .99, S, T, mean(macroData[:K]), c, k, w, r, y, z, u, macroData[:C], macroData[:K], macroData[:W], macroData[:R], macroData[:Y], wHat)
resultsOpt = optimize(fOpt, x, Newton(), autodiff=:forward)

xOpt = results.minimizer

#μ, α, ρ, σ
answerOpt = [LogitTransform(xOpt[1],5.0, 14.0),
          LogitTransform(xOpt[2], .01, .99),
          LogitTransform(xOpt[3], -.99, .99),
          LogitTransform(xOpt[4], 0.01, 1.1)]

mOpt(x) = Moments( x[1], x[2], x[3], x[4], .99, S, T,  mean(macroData[:K]), c, k, w, r, y, z, u, macroData[:C], macroData[:K], macroData[:W], macroData[:R], macroData[:Y], wHat)

momOpt = mOpt(answerOpt)

JOpt = ForwardDiff.jacobian( m, answerOpt )

varMatOpt = (1/S)*inv( JOpt' * wHat*JOpt )

stdErrorsOpt = [sqrt(varMatOpt[i,i]) for i in 1:4]
