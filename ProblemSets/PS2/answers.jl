using Plots
using DataFrames
using CSVFiles
using ForwardDiff
using Distributions
using SpecialFunctions
using Optim
using LinearAlgebra
using QuadGK
using Printf

function cln( x::Float64 )
    return replace(@sprintf("%.5g",x), r"e[\+]?([\-0123456789]+)" => s" \\times 10^{\1}")  
end
pyplot()

healthClaims = DataFrame(load("clms.csv", header_exists=false, colnames=["A"]))

results = [["mean", "min", "median", "max", "StdDev"] cln.([mean(healthClaims[:A]), minimum(healthClaims[:A]), median(healthClaims[:A]), maximum(healthClaims[:A]), std(healthClaims[:A])] )]

histogram( healthClaims[:A], bins=1000, normalize = true, label="Health Claims")
savefig("histOne.png")

#We force all bins to have length 8, and allow for 100 of them.
histogram( healthClaims[:A], bins=0:8:800, normalize=true, xlims=(0,800),label="Health Claims \$\\leq 800\$")
savefig("histTwo.png")

function GammaLogLikelihood( x::Vector{Float64}, α::Float64, β::Float64)
    #Yes I know I could get this using Distributions.jl which could
    #even do the MLE estimate But thats pretty much cheating, and
    #gamma is in the exponential family so using Newton's method will
    #cause no issues.

    #Pdf is: $\frac{ 1 }{\Gamma(\alpha) \beta^{\alpha}} x^{\alpha - 1} \exp\left( - \frac{x}{\beta} \right)$
    #Log-likelihood is: $- \alpha \log ( \beta) - \log( \Gamma (\alpha)) + (\alpha - 1) \log x - \frac{x}{\beta}$

    return -α*log( β) - lgamma(α) + (α - 1)*mean(log.(x)) - mean(x) / β
end

function GammaGradient( x::Vector{Float64}, α::Float64, β::Float64)
    delA = -log(β) - digamma(α) + mean(log.(x))
    #delB = mean(x) / β - α
    delB = mean(x) / β^2 - α / β
    return [delA,delB]
end

function GammaHessian( x::Vector{Float64}, α::Float64, β::Float64)
    delAA = -trigamma(α)
    delAB = -1 / β
    delBB =( α / (β*β)) - ((2* mean(x)) / (β*β*β))
    return [delAA delAB; delAB delBB]
end

function GammaPDF( α::Float64, β::Float64, x::Float64)
    return  (1 / (gamma(α)*β^α))*x^(α-1)*exp( -x/β)
end

function EstimateGammaParameters( data::Vector{Float64}, guess::Vector{Float64}, gradientFun, hessianFun)

    θ = guess
    tol = 1e-10
    maxLoops = 100

    grad = gradientFun( data, θ... )
    hess = hessianFun( data, θ... )

    loopCounter = 0
    while( loopCounter < maxLoops && norm(grad) >= tol)
        θ = θ - hess \ grad
        grad = gradientFun( data, θ... )
        hess = hessianFun( data, θ... )

        loopCounter += 1
        # println( norm(grad))
        # println( θ)
        # println( " ")
    end
    # println( loopCounter)
    return θ
end
healthCosts = convert(  Vector{Float64}, healthClaims[:A] )

β₀ =  var(healthCosts) / mean(healthCosts)
α₀ = mean(healthCosts) / β₀

(Gamma_̂α, Gamma_̂β) = EstimateGammaParameters( healthCosts, [α₀, β₀], GammaGradient, GammaHessian)

likelihood = GammaLogLikelihood(  healthCosts, Gamma_̂α, Gamma_̂β)

result = [["\$\\est{\\alpha}\$: ", "\$\\est{\\beta}\$: ", "Likelihood: " ] cln.([ Gamma_̂α,  Gamma_̂β, likelihood])]

histogram( healthClaims[:A], bins=0:8:800, normalize=true, xlims=(0,800),label="Health Claims \$\\leq 800\$")
pdfXVal = range( 0.0, 800.0)

pdfYVal = [GammaPDF( Gamma_̂α, Gamma_̂β, x ) for x in pdfXVal]

plot!( pdfXVal, pdfYVal, label="Gamma Estimate" )
savefig("histPDF_Gamma.png")

# $\text{(GG):}\quad f(x;\alpha,\beta,m) = \frac{m}{\beta^\alpha \Gamma\left(\frac{\alpha}{m}\right)}x^{\alpha-1}e^{-\left(\frac{x}{\beta}\right)^m},\quad x\in[0,\infty), \:\alpha,\beta,m>0$
function GGammaPDF( α::Float64, β::Float64, m::Float64, x::Float64)
    return ( (m / β^α) * x^(α-1) * exp( - (x / β)^m) ) / gamma( α / m)
end


function GGammaLikelihood( x::Vector{Float64}, α::Real, β::Real, m::Real)
    return log(m) - α*log(β) + (α - 1)*mean(log.(x)) - mean( (x ./ β).^m  ) - lgamma( α / m )    
end

function EstimateGG( data::Vector{Float64}, guess::Vector{Float64})
    #To hard enforce that all of our parameters are positive, we
    #exponentiate them. Limit them to .1 as the lower bound for
    #numerics sake
    θ = log.(guess .- .1)
    fun(x::Vector) = -GGammaLikelihood( data, (exp.(x).+ .1)... )



    result = optimize(fun, θ, Newton(), autodiff=:forward)
end


sln = EstimateGG( healthCosts, [Gamma_̂α, Gamma_̂β, 1.0])

GG_̂α = exp(sln.minimizer[1]) + .1
GG_̂β = exp(sln.minimizer[2]) + .1
GG_̂m = exp(sln.minimizer[3]) + .1
GG_LogLikelihood = -sln.minimum

println( "GG ̂α = ", GG_̂α)
println( "GG ̂β = ", GG_̂β )
println( "GG ̂m = ", GG_̂m )
println( "Likelihood Value: ", GG_LogLikelihood )

result = [["GG \$\\est{\\alpha}\$: ", "GG \$\\est{\\beta}\$: ", "GG \$\\est{m}\$: ","GG Likelihood: " ] cln.([ GG_̂α,  GG_̂β,  GG_̂m, GG_LogLikelihood])]

histogram( healthClaims[:A], bins=0:8:800, normalize=true, xlims=(0,800),label="Health Claims \$\\leq 800\$")
pdfXVal = range(0.0, 800.0)
#pdfXVal = linspace( minimum(truncatedHealthClaims), maximum(truncatedHealthClaims))
pdfYVal = [GGammaPDF( GG_̂α, GG_̂β, GG_̂m, x ) for x in pdfXVal]

plot!( pdfXVal, pdfYVal, label="Generalized Gamma Estimate" )
savefig( "histPDF_GG.png" )

function GBetaTwoPDF( x::Float64, a::Real, b::Real, p::Real, q::Real)
    #We require all parameters to be positive, so abs(a) = a
    return a*x^(a*p -1) / (b^(a*p) *beta(p,q)*(1+(x/b)^a)^(p+q))
end

function GBetaTwoLikelihood( x::Vector{Float64}, a::Real, b::Real, p::Real, q::Real)
    return log( a) + (a*p -1)*mean(log.(x)) - (a*p)*log(b) - log(beta(p,q)) - (p+q)*mean( log.( 1 .+(x ./ b).^a ))
end

function EstimateGBetaTwo( data::Vector{Float64}, guess::Vector{Float64})
      #To hard enforce that all of our parameters are positive, we
      #exponentiate them
    θ = log.(guess .- .1)
    #θ = guess
    fun(x::Vector) = -GBetaTwoLikelihood( data, (exp.(x) .+ .1)... )


    #This guy is being fickle, Newton() struggles a little bit, but
    #NewtonTrust seems to outperform LBFGS
    result = optimize(fun, θ, NewtonTrustRegion(), autodiff=:forward, Optim.Options(iterations=2000) )
end

#$GG(\alpha,\beta,m) = \lim_{q\rightarrow\infty}GB2\left(a=m,b=q^{1/m}\beta,p=\frac{\alpha}{m},q\right)$
sln = EstimateGBetaTwo( healthCosts, [GG_̂m, 10000^(1 / GG_̂m) * GG_̂β, GG_̂α / GG_̂m, 10000])

GB2_̂α = exp( sln.minimizer[1]) + .1
GB2_̂β = exp( sln.minimizer[2]) + .1
GB2_̂p = exp( sln.minimizer[3]) + .1
GB2_̂q = exp( sln.minimizer[4]) + .1
GB2_LogLikelihood = -sln.minimum

result = [["GB2 \$\\est{\\alpha}\$: ", "GB2 \$\\est{\\beta}\$: ", "GB2 \$\\est{p}\$: ","GB2 \$\\est{q}\$: ","GB2 Likelihood: " ] cln.([GB2_̂α, GB2_̂β,  GB2_̂p,  GB2_̂q, -sln.minimum])]

histogram( healthClaims[:A], bins=0:8:800, normalize=true, xlims=(0,800),label="Health Claims \$\\leq 800\$")
pdfXVal = range( 0.0, 800.0)
#pdfXVal = linspace( minimum(truncatedHealthClaims), maximum(truncatedHealthClaims))
pdfYVal = [GBetaTwoPDF( x, GB2_̂α, GB2_̂β, GB2_̂p, GB2_̂q ) for x in pdfXVal]

plot!( pdfXVal, pdfYVal, label="Generalized Beta 2 Estimate" )
savefig( "histPDF_GB2.png" )

# Gamma Has Two restrictions
tStatGamma = 2*N*(GB2_LogLikelihood - likelihood)
# Generalized Gamma Has One Restriction
tStatGG = 2*N*(GB2_LogLikelihood - GG_LogLikelihood)

results = [["", "Gamma", "Generalized Gamma"] [ "\$\\chi^{2}\$", cln(tStatGamma), cln(tStatGG)] ["p-value",  cln(1.0 - cdf(Chisq(4),tStatGamma)), cln(1.0 - cdf( Chisq(4),tStatGG)) ] ]

f(x) = GBetaTwoPDF( x, GB2_̂α, GB2_̂β, GB2_̂p, GB2_̂q )
area = quadgk( f, 0, 1000 )[1]
output = ["Probability of Having > 1000: " cln(1-area)]

f(x) = GammaPDF( Gamma_̂α, Gamma_̂β, x )
area = quadgk(f, 0, 1000)[1]
output = ["Gamma Probability of Having > 1000: " cln(1-area)]

#$\normal\left( \rho\left( \log w_{t-1} - \log( 1- \alpha) -(\alpha-1) \log k_{t-1} \right) + (1-\rho)\mu, \sigma^2 \right)$

#Clean it up when it exists, comes in the order: (c, k, w, r)
macroData = DataFrame(load("MacroSeries.csv", header_exists=false, colnames=["C", "K", "W", "R"]))

w = convert( Vector{Float64}, macroData[:W] )
k = convert( Vector{Float64}, macroData[:K] )

function LogLikelihood( N, w::Vector{Float64}, k::Vector{Float64}, α::Real, ρ::Real, μ::Real, σ²::Real  )
    #The pdf of a normal: $\frac{1}{\sqrt{2 \pi \sigma^2}} \exp( - \frac{ (x-\mu)^2}{2 \sigma^2})$
    #Log Likelihood: $- \frac{1}{2} \log \sigma^2 - \frac{ (x-\mu)^2}{ 2 \sigma^2}$

    logLik = -.5*log(σ²)- (( log(w[1]) - log(1-α) - (α)*log(k[1]) - μ)^2 / (2*σ²))

    #Note we do not have the -.5*log(2*pi)
    #Because that does not matter at all for MLE estimation.
    for i in 2:N
        mean = ρ*(log(w[i-1]) - log( 1 - α)  - (α)*log( k[i-1])) + (1-ρ)*μ
        logLik += -.5*log( σ² ) - (  (log(w[i]) - log(1-α) - (α)*log(k[i]) - mean)^2 / (2*σ²))
    end
    return logLik
end


N = length(w)

α₀ = .5
β = .99
μ₀ = .5
σ₀ = .5
ρ₀ = 0.0

#We parameterize each of the variables so that they meet their constraints.
# tanh is used to ensure that $\rho \in (-1,1)$
θ = zeros(4)
θ[1] = log( α₀ / ( 1 - α₀) )
θ[2] = atanh( ρ₀)
θ[3] = log( μ₀ )
θ[4] = log( σ₀)


fun(x::Vector) = -LogLikelihood( N, w, k, exp(x[1]) / (1 + exp(x[1])), tanh(x[2]), exp(x[3]), exp(x[4])  )

result = optimize(fun, θ, Newton(), autodiff=:forward)

model_̂θ = result.minimizer

model_̂α = exp(model_̂θ[1]) / (1 + exp(model_̂θ[1]))
model_̂ρ = tanh(model_̂θ[2])
model_̂μ = exp(model_̂θ[3])
model_̂σ = exp(model_̂θ[4])

output = [["\$\\est{\\alpha}\$:", "\$\\est{\\rho}\$:", "\$\\est{\\mu}\$:", "\$\\est{\\sigma^{2}}\$:"]  cln.([model_̂α, model_̂ρ, model_̂μ, model_̂σ])]

#Sadly Optim.jl does not automatically report the hessian, though I am
#sure it is obtainable. So we will use forward-mode automatic
#differentiation to obtain this hessian. However it does not always
#return symmetric matrices, so we will make the matrix symmetric then
#invert it using the cholesky decomposition to be numerically stable.
hess = ForwardDiff.hessian(fun, result.minimizer)

F = cholesky(Hermitian(hess))
F.L * F.U = H
hessInv = cln.(F.U \ (F.L \ I))
#This is for version .6 rather than the 1.0 running above.
#F = chol(Hermitian(hess))
#hessInv = cln.(F \ (F' \ I))
result = hessInv

r = convert( Vector{Float64}, macroData[:R] )
k = convert( Vector{Float64}, macroData[:K] )

#$\log r_t - \log \alpha - z_t - (\alpha - 1 ) \log k_t = 0$
function LogLikelihood( N, r::Vector{Float64}, k::Vector{Float64}, α::Real, ρ::Real, μ::Real, σ²::Real  )
    #The pdf of a normal: $\frac{1}{\sqrt{2 \pi \sigma^2}} \exp( - \frac{ (x-\mu)^2}{2 \sigma^2})$
    #Log Likelihood: $- \frac{1}{2} \log \sigma^2 - \frac{ (x-\mu)^2}{ 2 \sigma^2}$

    logLik = -.5*log(σ²)- (( log(r[1]) - log(α) - (α-1)*log(k[1]) - μ)^2 / (2*σ²))

    #Note we do not have the -.5*log(2*pi)
    #Because that does not matter at all for MLE estimation.
    for i in 2:N
        mean = ρ*(log(r[i-1]) - log( α)  - (α-1)*log( k[i-1])) + (1-ρ)*μ
        logLik += -.5*log( σ² ) - (  (log(r[i]) - log(α) - (α-1)*log(k[i]) - mean)^2 / (2*σ²))
    end

    return logLik
end

N = size(macroData)[1]

α₀ = .5
β = .99
μ₀ = .5
σ₀ = .5
ρ₀ = 0.0

#We parameterize each of the variables so that they meet their
# constraints.  tanh is used to ensure that $\rho \in (-1,1)$
θ = zeros(4)
θ[1] = log( α₀ / ( 1 - α₀) )
θ[2] = atanh( ρ₀)
θ[3] = log( μ₀ )
θ[4] = log( σ₀)

function limitedLogistic( unbounded::Real )
    return ((exp(unbounded)) / ( 1 + exp(unbounded)))*.99 + .005
end

#This clamp on the logistic function is quite the hack, since this
#function shouldn't get to 0 or 1, but it was getting stuck at 1
fun(x::Vector) = -LogLikelihood( N, r, k, limitedLogistic(x[1]), tanh(x[2]), exp(x[3]), exp(x[4])  )

result = optimize(fun, θ, Newton(), autodiff=:forward)

bmodel_̂θ = result.minimizer

bmodel_̂α = limitedLogistic(bmodel_̂θ[1])
bmodel_̂ρ = tanh(bmodel_̂θ[2])
bmodel_̂μ = exp(bmodel_̂θ[3])
bmodel_̂σ = exp(bmodel_̂θ[4])

output = [["\$\\est{\\alpha}\$:", "\$\\est{\\rho}\$:", "\$\\est{\\mu}\$:", "\$\\est{\\sigma^{2}}\$:"]  cln.([bmodel_̂α, bmodel_̂ρ, bmodel_̂μ, bmodel_̂σ])]

#Sadly Optim.jl does not automatically report the hessian, though I am
#sure it is obtainable. So we will use forward-mode automatic
#differentiation to obtain this hessian. However it does not always
#return symmetric matrices, so we will make the matrix symmetric then
#invert it using the cholesky decomposition to be numerically stable.
hess = ForwardDiff.hessian(fun, result.minimizer)

F = cholesky(Hermitian(hess))
#F.U' * F.U = H
hessInv = cln.(F.U \ (F.L \ I))
# F = chol(Hermitian(hess))
# hessInv = cln.(F \ (F' \ I))
result = hessInv

prob = 1 - cdf( Normal(), -(1.0 / sqrt(model_̂σ))*( log(model_̂α) + model_̂ρ*10 + (1-model_̂ρ)*model_̂μ + (model_̂α-1)*log( 7500000)))
result = ["\\Pr( r_t > 1) = " cln(prob)]
