__precompile__()
module RadauBVP

using NLsolve
# just temporary
import Logging
export radau3

"""
Radau IIA method with 3 collocation points (5th order)

This method is suitable  for both IVP and BVP.

The first argument `c!` must be a function of the form

```
c!(y0, yn, r)
```

where `y0` and `yn` are the boundary values of `y` in equation dy/dt=f(t,y), and
`r` is the residual (penalty) of the same size as `y`. `c!` must fill up `r` according
to initial or boundary problem statement. In IVP case, when `y0 = η`, `c!` will simply
be `r[:] .= y0 .- η`.

The second argument is RHS function of the ODE equation `dy/dt=f(t,y)` of the form:

```
f!(t, y, dydt)
```

The third and the forth arguements are the Jacobians of BC function `c!` and
ODE function `f!` respectively. Jacobian of BC function `c!` must have a form

```
dc!(y0, yn, jac)
```

where `jac` must be a matrix of size `M×2M` where `M` is the length of `y`,
such that `∂(y0[j])[c![i]] = jac[i,j]` and `∂(yn[j])[c![i]] = jac[i, M+j]`

The form of ODE function Jacobian:

```
df!(t, y, jac)
```

with `jac` being a matrix of size `M×M`: `∂(y[j])[f![i]] = jac[i,j]`.

`t0` and `tf` are initial and final time of integration. `n` is the number of time steps.

"""
function radau3(c!, f!, dc!, df!, y0 :: Array{Float64, 2}, t0 :: Float64, tf :: Float64)
	sz = size(y0)                                 # M - spatial size, N - time
	const M :: Int64 = sz[1]
	const N :: Int64 = sz[2]
	const MN :: Int64 = length(y0)                  # total problem size
	local yarg = Array(eltype(y0), M)               # temp location for arg of f!
	local y1 :: Vector{Float64} = Array(eltype(y0), M)
	local yN :: Vector{Float64} = Array(eltype(y0), M)
	local k1 :: Vector{Float64} = Array(eltype(y0), M)
	local k2 :: Vector{Float64} = Array(eltype(y0), M)
	local k3 :: Vector{Float64} = Array(eltype(y0), M)
	# Time
    const t = linspace(t0, tf, N)                    # all time points
    const h :: Float64 = t[2]-t[1]                   # time step
    # Butcher tableau for Radau IIA method (s = 3)
    # all the expressions should be computed at compile time by Julia...
    const c :: Vector{Float64} = [0.4-√6/10, 0.4+√6/10, 1.0]
    const a :: Array{Float64,2} = [
        11/45-7*√6/360     37/225-169*√6/1800 -2/225+√6/75;
        37/225+169*√6/1800 11/45+7*√6/360     -2/225-√6/75;
        4/9-√6/36          4/9+√6/36           1/9]
    const b :: Vector{Float64} = [4/9-√6/36, 4/9+√6/36, 1/9]
    # Resulting non-linear equation
    @inbounds function eq!(x :: Vector{Float64}, r :: Vector{Float64})
		@inline yind(m :: Int64, n :: Int64) = x[M*(n-1) + m]
		@inline kind(m :: Int64, j :: Int64, n :: Int64) =
			x[MN + 3M*(n-1) + M*(j-1) + m]
		@inline yrange(dest :: Vector{Float64}, n :: Int64) =
			dest[:] = @view x[M*(n-1)+1:M*(n-1)+M]
		@inline krange(dest :: Vector{Float64}, i :: Int64, n :: Int64) =
			dest[:] = @view x[MN+3M*(n-1)+M*(i-1)+1:MN+3M*(n-1)+M*(i-1)+M]
        # Boundary conditions
		yrange(y1, 1)
		yrange(yN, N)
		# not sure if there is a better way to do it...
        c!(y1, yN, view(r, 1:M))
        # base index
        ind = M
        # Iterate over each time element
        for n=1:(N-1)
			krange(k1, 1, n)
			krange(k2, 2, n)
			krange(k3, 3, n)
			# y[:,n]
			yrange(yN, n)
			# y[:,n+1]
			yrange(y1, n+1)
            # y[:,n+1] = y[:,n] + h∑b[j]*k[:,j,n]
            # this doesn't fuse in Julia 0.5
            # r[ind+1:ind+ny0] .= yn1 .- yn .- h .* (b[1] .* k1 .+ b[2] .* k2 .+ b[3] .* k3)
            # this form should fuse:
            r[ind+1:ind+M] .= (-).(y1, (+).(yN, (*).(h, (+).((*).(b[1], k1), (*).(b[2], k2), (*).(b[3], k3)))))
            # update base index
            ind += M
            # k[:,j,n] - f(t[i]+c[j]h, y[:,n]+h∑a[j,q]k[:,q, n]) = 0
			# manual loop unrolling: otherwise have to deal with views over
			# k[:,j]
			# j = 1
			yarg .= (+).(yN, (*).(h, (+).((*).(a[1,1], k1), (*).(a[1,2], k2), (*).(a[1,3], k3))))
			rk = @view r[ind+1:ind+M]
			f!(t[n]+c[1]*h, yarg, rk)
			rk .= (-).(k1, rk)
			ind += M
			# j = 2
			yarg .= (+).(yN, (*).(h, (+).((*).(a[2,1], k1), (*).(a[2,2], k2), (*).(a[2,3], k3))))
			rk = @view r[ind+1:ind+M]
			f!(t[n]+c[2]*h, yarg, rk)
			rk .= (-).(k2, rk)
			ind += M
			yarg .= (+).(yN, (*).(h, (+).((*).(a[3,1], k1), (*).(a[3,2], k2), (*).(a[3,3], k3))))
			rk = @view r[ind+1:ind+M]
			f!(t[n]+c[3]*h, yarg, rk)
			rk .= (-).(k3, rk)
			ind += M
            # for j=1:3
            #     # doesn't fuse in Julia 0.5
            #     # yarg .= yn .+ h .* (a[j,1] .* k1 .+ a[j,2] .* k2 .+ a[j,3] .* k3)
            #     yarg .= (+).(yN, (*).(h, (+).((*).(a[j,1], k1), (*).(a[j,2], k2), (*).(a[j,3], k3))))
            #     # storage for equation corresponding to individual k[m,j,n]
			# 	# can't avoid the view here
            #     rk = @view r[ind+1:ind+M]
            #     f!(t[n]+c[j]*h, yarg, rk)
			# 	# not ideal...
            #     rk .= (-).((@view k[:,j,n]), rk)
            #     ind += M
            # end
        end
    end

	# Location for the Jacobian of BC
	dc = Array(eltype(y0), M, 2M)
	# Location for the Jacobian ∂_j[f_i] of f!
	df = Array(eltype(y0), M, M, 3)

	function deq!(x, jac)
		jac[:,:] = 0.0
		y = reshape((@view x[1:MN]), (M,N))
		k = reshape((@view x[MN+1:end]), (M, 3, N-1))
		# Jacobian for BC
		dc!((@view y[:, 1]), (@view y[:, N]), dc)
		for m=1:M
			# for t = t0
			jac[m, 1:M] = @view dc[m, 1:M]
			# for t = tf
			jac[m, MN-M+1:MN] = dc[m, M+1:end]
		end
		# Base index
		ind = M
		for n=1:(N-1)
			k1 = @view k[:,1,n]
			k2 = @view k[:,2,n]
			k3 = @view k[:,3,n]
			yn = @view y[:,n]
			yn1 = @view y[:,n+1]
			# Get the Jacobian ∂_j[f_i]
			for i=1:3
				yarg .= (+).(yn, (*).(h, (+).((*).(a[i,1], k1), (*).(a[i,2], k2), (*).(a[i,3], k3))))
				df!(t[n]+c[i]*h, yarg, (@view df[:, :, i]))
			end
			# Jacobian for y[m, n+1] - y[m, n] - h∑b[i] k[m, i, n-1]
			for m = 1:M
				# wrt y[m, n+1]
				jac[ind + m, M*n + m] = 1.0
				# wrt y[m,n]
				jac[ind + m, M*(n-1) + m] = -1.0
				# wrt k[m, i, n]
				# k[m,i,n] → x[MN + 3M*(n-1) + M*(i-1) + m]
				jac[ind+m, MN + 3M * (n-1) + m]      = - h * b[1]
				jac[ind+m, MN + 3M * (n-1) + M + m]  = - h * b[2]
				jac[ind+m, MN + 3M * (n-1) + 2M + m] = - h * b[3]
			end
			# Jacobian for k[m,i,n] - 𝔉[m,i,n] = 0
			# where 𝔉[m,i,n] = f(t[n]+c[i]h, y[:,n] + h∑a[i,j]k[:,j,n])[m]
			ind += M
			for i=1:3
				# wrt y[μ, n]
				jac[ind + 1 : ind + M, M*(n-1) + 1 : M*(n-1) + M] = - df[1:M, 1:M, i]
				# wrt k[μ, j, n]
				for j=1:3
					jac[ind + 1 : ind + M, MN + 3M*(n-1) + M*(j-1) + 1 : MN + 3M*(n-1) + M*(j-1)+M] = df[1:M, 1:M, i]*(-h)*a[i,j]
				end
				# for k[m,i,n] add 1
				for m=1:M
					jac[ind+m, MN + 3M*(n-1) + M*(i-1) + m] += 1.0
				end
				ind += M
			end

		end

	end

    k0 = Array(eltype(y0), M, 3, N-1)
	for n=1:(N-1)
		for i=1:3
			f!(t[n], (@view y0[:,n]), (@view k0[:,i,n]))
		end
	end
    x0 = Array(eltype(y0), M * N * 4 - M * 3)
    x0[1:MN] = @view y0[:]
	x0[MN+1:end] = k0[:]
	# warn("size(x0) = $(size(x0))")
    result = nlsolve(eq!, deq!, x0)
	t_tot = zeros(Float64, 2*(N-1)+N)
	y_tot = zeros(Float64, (M, length(t_tot)))
	if result.x_converged || result.f_converged
		yres = reshape(result.zero[1:MN], M, N)
		kres = reshape(result.zero[MN+1:end], M, 3, (N-1))
		# Time with all collocation points
		t_tot[1] = t0
		for n=1:(N-1)
			for i=1:2
				t_tot[1+3*(n-1)+i] = t[n] + c[i]*h
			end
			t_tot[1+3*(n-1)+3] = t[n+1]
		end
		# Result in all collocation points
		y_tot[:,1] = yres[:, 1]
		for n=1:(N-1)
			for i=1:2
				y_tot[:,1+3*(n-1)+i] = yres[:,n] + h * (a[i,1]*kres[:,1,n]+a[i,2]*kres[:,2,n]+a[i,3]*kres[:,3,n])
			end
			y_tot[:,1+3*(n-1)+3] = yres[:,n+1]
		end
		return (true, t_tot, y_tot, result)
	end
	return (false, t_tot, y_tot, result)
	# (eq!, deq!, x0)
end


end # module
