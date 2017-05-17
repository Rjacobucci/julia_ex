#using MultivariateStats
using Mamba
Pkg.build("GraphViz")
using DataFrames
dat = readtable("/Users/Ross/Documents/Github/julia_ex/HS6.dat",header=false,separator=' ')

print(head(dat))

model = Model(

xmat = Stochastic(2,
 (lambda,FS,resid,N,T) ->
   UnivariateDistribution[
     begin
      Normal(lambda[j]*FS[i],sqrt(resid[j]))
     end
     for i in 1:N, j in 1:T
   ],
 false
),

  lambda = Stochastic(1,
    () -> Normal(0, sqrt(1000))
  ),

  FS = Stochastic(1,
    () -> Normal(0,1)
  ),

  resid = Stochastic(1,
    () -> InverseGamma(0.001, 0.001)
  )

)

line = Dict{Symbol, Any}(
  :x2 => Vector(dat[:,:x2]),
  :x3 => Vector(dat[:,:x3]),
  :x4 => Vector(dat[:,:x4]),
  :x5 => Vector(dat[:,:x5]),
  :x6 => Vector(dat[:,:x6]),
  :x1 => Vector(dat[:,:x1]),
  :T => 6,
  :N => 301
)
line[:xmat] = [line[:x1] line[:x2] line[:x3] line[:x4] line[:x5] line[:x6]]


## Initial Values
inits = [
  Dict{Symbol, Any}(
    :xmat => line[:xmat],
    :FS => zeros(line[:N]),
    :lambda => rand(Normal(0, 1), 6),
    :resid => rand(Gamma(1, 1),6)
  )
  for i in 1:1
]

#using GraphViz

#display(Graph(graph2dot(model)))

scheme = [AMWG(:lambda, 0.1),
          Slice(:resid, 1.0),
          Slice(:FS, 0.5)]
setsamplers!(model, scheme)
sim1 = mcmc(model, line, inits, 10000, burnin=250, thin=2, chains=1)
describe(sim1)
