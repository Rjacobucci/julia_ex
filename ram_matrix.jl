# http://julia-demo.readthedocs.io/en/latest/stdlib/sparse.html

A = zeros(4000,4000)
ndims(A)
size(A)
Base.summarysize(A)

A_s = sparse(A)
Base.summarysize(A_s)

@time 4*4

I = sparse(eye(size(A)[1]))
Base.summarysize(sparse(I))

Fmat = sparse(eye(size(A)[1]))

S = zeros(4000,4000)
S_s = sparse(S)

Fmat * inv(I - A)

function ram_mult(A,S,F)
  Imat = sparse(eye(size(A)[1]))

  F * inv(Imat - A) * S * transpose(inv(I - A)) * transpose(F)

end

@time impCov = ram_mult(A,S_s,Fmat)
