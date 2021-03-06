#using MultivariateStats
using Mamba
Pkg.build("GraphViz")
using DataFrames
dat = readtable("/Users/Ross/Documents/Github/julia_ex/HS6.dat",header=false,separator=' ')

print(head(dat))

model = Model(

  y = Stochastic(1,
    (mu, s2) ->  MvNormal(mu, sqrt(s2)),
    false
  ),

  mu = Logical(1,
    (xmat, beta) -> xmat * beta,
    false
  ),

  beta = Stochastic(1,
    () -> MvNormal(2, sqrt(1000))
  ),

  s2 = Stochastic(
    () -> InverseGamma(0.001, 0.001)
  )

)

line = Dict{Symbol, Any}(
  :x => Vector(dat[:,:x2]),
  :y => Vector(dat[:,:x1])
)
line[:xmat] = [ones(size(dat)[:1]) line[:x]]


## Initial Values
inits = [
  Dict{Symbol, Any}(
    :y => line[:y],
    :beta => rand(Normal(0, 1), 2),
    :s2 => rand(Gamma(1, 1))
  )
  for i in 1:1
]

#using GraphViz

#display(Graph(graph2dot(model)))


scheme1 = [NUTS(:beta),
           Slice(:s2, 3.0)]
setsamplers!(model, scheme1)
sim1 = mcmc(model, line, inits, 10000, burnin=250, thin=2, chains=1)
describe(sim1)
