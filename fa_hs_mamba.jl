#using MultivariateStats
using Mamba
#Pkg.build("GraphViz")
using DataFrames
dat = readtable("/Users/Ross/Documents/Github/julia_ex/HS6.dat",header=false,separator=' ')

print(head(dat))

model = Model(

xmat = Stochastic(2,
 (lambda,FS,alpha,resid,N,T) ->
   UnivariateDistribution[
     begin
      Normal(alpha[j] + lambda[j]*FS[i],resid[j])
     end
     for i in 1:N, j in 1:T
   ],
 false
),

  lambda = Stochastic(1,
    () -> Normal(0, sqrt(1000))
  ),

  alpha = Stochastic(1,
    () -> Normal(0, 1)
  ),

  FS = Stochastic(1,
    () -> Normal(0,1)
  ),

  resid = Stochastic(1,
    () -> Gamma(1, 1)
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
    :alpha => rand(Normal(0,1),6),
    :resid => rand(Gamma(1, 1),6)
  )
  for i in 1:2
]

#using GraphViz

#display(Graph(graph2dot(model)))

scheme = [AMWG(:alpha, 0.1),
          Slice(:lambda, 1.0),
          Slice(:resid,1.0),
          Slice(:FS, 0.5)]
setsamplers!(model, scheme)
sim1 = mcmc(model, line, inits, 5000, burnin=1500, thin=1, chains=2)
describe(sim1)
plot(sim1)
print(gelmandiag(sim1, mpsrf=true, transform=true))


sim = sim1[1000:10000, ["lambda[1]", "lambda[2]"], :]
p = plot(sim)
display(p)
draw(p, filename="summaryplot.svg")
