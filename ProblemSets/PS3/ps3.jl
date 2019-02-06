using CSV
using DataFrames
using Plots
using Distributions
using LinearAlgebra
using Optim
pyplot()

incomes = DataFrame( CSV.File( "data/usincmoms.csv", delim='\t',header=[:percent,:midpoint] ) )
#,colnames=["Percent", "midpoint"]

heights = copy(incomes[:percent])
heights[41] /= 10.0
heights[42] /= 20.0

bins = Vector{Int64}(undef,length(incomes[:midpoint])+1)
bins[1] = 0
for i in 1:length(incomes[:midpoint])
    bins[i+1] = bins[i] + ((incomes[:midpoint][i]) - bins[i])*2
end

plotBins = copy(bins)
plotBins ./= 1000.0

plot( plotBins, heights, bins = bins, seriestype=:barbins, xlabel="Thousands of Dollars", label="US Incomes", ylabel="density")

savefig("histogram.pdf")

# #We're gonna simulate some stuff and match moments for our initial
# #guess
# dataProbs = cumsum( incomes[:percent])
# dataProbs[length(dataProbs)] = 1.0
# N = 100000
# M = length(dataProbs)
# simulation = rand( Uniform(), N)
# for i in 1:N
#     for j in 1:M
#         #if( dataProbs[j] >= simulation[i])
#         if( simulation[i] < dataProbs[j])
#             simulation[i] = incomes[:midpoint][j]
#             break
#         end
#     end
# end

simMean = mean(incomes[:percent].*incomes[:midpoint])
simVar = mean(incomes[:percent.*(incomes[:midpoint].^2)]) - simMean^2#var(simulation)
#From the lognormal we know that

#simMean = exp( mu + sig / 2)
#simVar = (exp(sig) - 1)exp( 2mu + sig)

#mu = log(simMean) - sig / 2
#simVar = (exp(sig) - 1) simMean^2
#log (simVar / simMean^2 + 1) = sig

sigGuess = log( simVar / (simMean*simMean) + 1)
muGuess = log(simMean) - sigGuess / 2.0


container = Vector{Float64}(undef,42)


function GenerateMoments( container::Vector{Float64},bins::Vector{Float64},
                                   distribution, params::Vector)#μ::Float64, σ::Float64 )
    cdfs = [cdf( distribution(params...), x) for x in bins]

    for i in 1:42
        container[i] = cdfs[i+1] - cdfs[i]
    end
    #container[42] = 1.0 - cdfs[42]
    return container
end

#W = Matrix{Real}(undef,42,42)
W = convert( Matrix{Float64}, Diagonal(convert(Vector{Float64}, incomes[:percent])) )
# for i in 1:42
#     for j in 1:42
#         W[i,j] = Wfloat[i,j]
#     end
# end


function GMMCrit( W::Matrix{Float64}, dataMoments::Vector{Float64},
                  container::Vector{Float64}, bins::Vector{Float64},
                  distributions, params::Vector)
    e = GenerateMoments( container, bins, distributions, params ) - dataMoments
    return dot(e, W*e)#e'*W*e#(ones(42)')*W*e#(dot(W*e))
end

dataStuff = convert(Vector{Float64}, incomes[:percent])
cdfBins = convert(Vector{Float64}, bins)

θ = [muGuess, sqrt(sigGuess)]
fun(x::Vector) = GMMCrit( W, dataStuff, container, cdfBins, LogNormal, x  )

logResult = optimize( fun, θ, BFGS())#, Optim.Options(g_tol = 1e-18))

mu = logResult.minimizer[1]
sigma = logResult.minimizer[2]

estHeightsLoggyBoy = GenerateMoments( container, cdfBins, LogNormal, [mu, sigma] )
estHeightsLoggyBoy[41] /= 10.0
estHeightsLoggyBoy[42] /= 20.0

plot!(incomes[:midpoint] / 1000.0, estHeightsLoggyBoy, label="LogNormal GMM Estimate")
savefig("loggyboy.pdf")

gamContainer = copy(container)
θ = [3.0, 25000.0]
betaFun(x::Vector) = GMMCrit( W, dataStuff, gamContainer, cdfBins, Gamma, x  )
gamResult = optimize( betaFun, θ, BFGS())

gamAlpha = gamResult.minimizer[1]
gamBeta = gamResult.minimizer[2]

estHeightsGammaMan = GenerateMoments( gamContainer, cdfBins, Gamma, [gamAlpha, gamBeta] )
estHeightsGammaMan[41] /= 10.0
estHeightsGammaMan[42] /= 20.0

plot!(incomes[:midpoint] / 1000.0, estHeightsGammaMan, label="Gamma GMM Estimate")
savefig("gammaMan.pdf" )

#Compare the fit using the GMM weight function
gamResult.minimum
logResult.minimum

#Should we compare the distribution using distances like total
#variation or kullback-leibler?
#Total variation says tehres no difference b/w the distributions.
sum(abs(x - y) for (x,y) in zip( estHeightsGammaMan, dataStuff)) / 2

#THis isn't dependent upon the W used, while the minimum is.
sum(abs(x - y) for (x,y) in zip( estHeightsLoggyBoy, dataStuff)) / 2


# We have a severe dearth of data, so I am going to simulate more, if
# we believe we have an asymptotically correct estimate, then these
# simulation moments should converge to the truth.

S = 100
simulation = Vector{Vector{Float64}}(undef,S)
simMoments = Matrix{Float64}(undef, S, 42)
simMoments .= 0
for s in 1:S
    simulation[s] = rand(Gamma( gamAlpha, gamBeta), N)
    for i in 1:N
        for j in 1:42
            if( simulation[s][i] >= bins[j] && simulation[s][i] < bins[j+1]  )
                simMoments[s,j] += 1
            end
        end
    end
end

simMoments ./= N

#Theres a problem with this simulating over the upper bound of the
#bins, so we just make it sum to one
for s in 1:S
    simMoments[s,42] = 1.0 - sum(simMoments[s,1:41])
    #simMoments[s,:] = estHeightsGammaMan - simMoments[s,:]
end

simErrors = Matrix{Float64}(undef, S, 42)
for s in 1:S
    simErrors[s,:] = estHeightsGammaMan - simMoments[s,:]
end


F = cholesky(mean( simMoments[s,:]*simMoments[s,:]' for s in 1:S ))
#F.L * F.U
wHat = F.U \ (F.L \ I)

twoStageContainer = copy(container)
θ = [gamAlpha, gamBeta]
gamFun(x::Vector) = GMMCrit( wHat, dataStuff, twoStageContainer, cdfBins, Gamma, x  )
gamResult = optimize( gamFun, θ, NelderMead())

twoStageAlpha = gamResult.minimizer[1]
twoStageBeta = gamResult.minimizer[2]

estHeightsTwoStage = GenerateMoments( twoStageContainer, cdfBins, Gamma, [twoStageAlpha, twoStageBeta] )
estHeightsTwoStage[41] /= 10.0
estHeightsTwoStage[42] /= 20.0

plot!(incomes[:midpoint] / 1000.0, estHeightsTwoStage, label="Two-Stage Gamma GMM Estimate")
savefig("twoStageLad.pdf")


sum(abs(x - y) for (x,y) in zip( estHeightsTwoStage, dataStuff)) / 2
