
using MultivariateStats

using DataFrames
dat = readtable("/Users/Ross/Documents/Github/regsem_analyses/arxiv/growth/wisc4vpe.txt",header=false,separator=' ')

names!(dat,[:V1,:V2,:V4,:V6,:P1,:P2,:P4,:P6,:Moeducat])


line = Dict{Symbol, Any}(
  :V1 => Vector(dat[:,:V1]),
  :V2 => Vector(dat[:,:V2]),
  :V4 => Vector(dat[:,:V4]),
  :V6 => Vector(dat[:,:V6]),
  :T => 4,
  :load => [1,2,4,6],
  :N => size(dat[:V1], 1)
)
line[:xmat] = [line[:V1] line[:V2] line[:V4] line[:V6]]


lgm_mod = Model(

xmat = Stochastic(2,
 (FS,resid,N,T) ->
   UnivariateDistribution[
     begin
      Normal(FS[i,1] + load[j]*FS[i,2],resid)
     end
     for i in 1:N, j in 1:T
   ],
 false
),

FS = Stochastic(2,
  (mu, Sigma, N) ->
    MultivariateDistribution[
      MvNormal(mu, Sigma)
      for i in 1:N
      ],
      false
),

mu = Stochastic(1,
  () -> Normal(0, 1)
),

Sigma = Stochastic(2,
  () -> Wishart(2, eye(2)),
  false
),


  resid = Stochastic(1,
    () -> Gamma(1, 1)
  )

)

inits = [
  Dict{Symbol, Any}(
    :xmat => line[:xmat],
    :FS => repmat([10,5], line[:N], 1),
    :mu => [0,0],
    :Sigma => eye(2),
    :resid => rand(Gamma(1, 1),1)
  )
  for i in 1:2
]


scheme3 = [AMWG(:Sigma, 0.1),
          Slice(:mu, 1.0),
          Slice(:resid,1.0),
          Slice(:FS, 0.5)]
setsamplers!(lgm_mod, scheme3)
sim_lgm = mcmc(lgm_mod, line, inits, 5000, burnin=1500, thin=1, chains=2)
