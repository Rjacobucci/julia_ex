Pkg.installed()

Pkg.add("MultivariateStats")
Pkg.add("DataFrames")
Pkg.add("Requests")

using MultivariateStats

using DataFrames

Pkg.add("MixedModels")
using MixedModels



dat = readtable("/Users/RJacobucci/Documents/Github/random_code/wisc4vpe.dat",header=false,separator=' ')
names!(dat,[:V1,:V2,:V4,:V6,:P1,:P2,:P4,:P6,:Moeducat])
print(head(dat))
describe(dat)
dat[:id] = linspace(1,204,204)

# reshape
dat2 = dat[:,[:id,:V1,:V2,:V4,:V6]]
#wisc_stack = stack(dat,[:id])
d = melt(dat2,[:id])
names!(d,[:Grade,:Verbal,:id])

d[:Grade2] = 1
d[205:408,[:Grade2]] = 2
d[205:612,[:Grade2]] = 4
d[613:816,[:Grade2]] = 6


wisc_long = readtable("/Users/RJacobucci/Documents/archive/WISC/wiscRtran.dat",header=false,separator= ' ')
names!(wisc_long,[:id,:occ,:verb,:perf,:momed,:grad])

# run mixed effects model

#fm1 = fit!(lmm(Verbal ~ Grade2 + (Grade2|id),d))
fm1 = fit!(lmm(verb ~ occ + (occ|id),wisc_long))
print(fm1)
print(coef(fm1))
print(fixef(fm1))
print(ranef(fm1))
print(deviance(fm1))
print(objective(fm1))
