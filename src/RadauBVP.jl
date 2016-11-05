__precompile__()
module RadauBVP

using NLsolve
import Optim
export radau3

"""
Radau IIA method with 3 collocation points (5th order)

This method is suitable  for both IVP and BVP.

The first argument `c!` defines boundary conditions and must be a function of
the form

```
c!(y0, yn, r)
```

where `y0` and `yn` are the boundary values of `y` in equation dy/dt=f(t,y), and
`r` is the residual (penalty) of the same size as `y`. `c!` must fill up `r`
according to initial or boundary problem statement. In IVP case, when `y0 = η`,
`c!` will simply be `r[:] .= y0 .- η`.

The second argument is RHS function of the ODE equation `dy/dt=f(t,y)` of the
form

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

`y0` is the initial guess of the solution. It must be a matrix of size
`M×(3(N-1)+1)`, where `N` is the number of time elements (the complicated
	size in time is due to defining initial guess on all collocation points).

`t0` and `tf` are initial and final time of integration. `n` is the number of time steps.

The result is the tuple:

```
(conerged, t, y, raw)
```

of type

```
Tuple{Bool, Vector{Float64}, Array{Float64,2}, Any}
```

where `converged` is the indicator if the solver succeeded, `t` is a vector of
all time points (including collocation points, of length `3(N-1)+1`), `y` is
the solution matrix of size `M×(3(N-1)+1)`, and `raw` is the raw result from
nonlinear solver (see `NLsolve` for more details)
"""
function radau3(c!, f!, dc!, df!, y0 :: Array{Float64, 2}, t0 :: Float64, tf :: Float64)
	sz = size(y0)
	local M :: Int64 = sz[1]                        # size of ODE problem
	local N :: Int64 = div(sz[2] + 2, 3)            # No of time elements
	# y[m,i,n]
	local y1 :: Vector{Float64} = Array(eltype(y0), M)
	local y2 :: Vector{Float64} = Array(eltype(y0), M)
	local y3 :: Vector{Float64} = Array(eltype(y0), M)
	# y[m,3, n-1]
	local yprev :: Vector{Float64} = Array(eltype(y0), M)
	# k[m,i,n]
	local k1 :: Vector{Float64} = Array(eltype(y0), M)
	local k2 :: Vector{Float64} = Array(eltype(y0), M)
	local k3 :: Vector{Float64} = Array(eltype(y0), M)
	# Time
    local t = linspace(t0, tf, N)                   # time elements points
    local h :: Float64 = t[2]-t[1]                  # time step
    # Butcher tableau for Radau IIA method (s = 3)
    # all the expressions should be computed at compile time by Julia...
    const c :: Vector{Float64} = [0.4-√6/10, 0.4+√6/10, 1.0]
    const a :: Array{Float64,2} = [
        11/45-7*√6/360     37/225-169*√6/1800 -2/225+√6/75;
        37/225+169*√6/1800 11/45+7*√6/360     -2/225-√6/75;
        4/9-√6/36          4/9+√6/36           1/9]
    # Resulting non-linear equation
    @inbounds function eq!(x :: Vector{Float64}, r :: Vector{Float64})
		@inline yind(m :: Int64, i :: Int64, n :: Int64) =
			x[3M*(n-1) + M*(i-3) + m]
		@inline yrange(dest :: Vector{Float64}, i :: Int64, n :: Int64) =
			dest[:] = @view x[3M*(n-1) + M*(i-3) + 1:3M*(n-1) + M*(i-3) + M]
        # Boundary conditions
		yrange(y3, 3, 1)
		yrange(y2, 3, N)
		# not sure if there is a better way to do it...
        c!(y3, y2, view(r, 1:M))
        # base index
        ind = M
        # Iterate over each time element
        for n=2:N
			yprev, y3 = y3, yprev
			yrange(y1, 1, n)
			yrange(y2, 2, n)
			yrange(y3, 3, n)
			f!(t[n-1]+c[1]*h, y1, k1)
			f!(t[n-1]+c[2]*h, y2, k2)
			f!(t[n-1]+c[3]*h, y3, k3)
			# y[m,1,n] - y[m,3,n-1] - h∑a[1,j]*k[m,j,n]
			r[ind+1:ind+M] .= (-).(y1, (+).(yprev, (*).(h, (+).((*).(a[1,1], k1), (*).(a[1,2], k2), (*).(a[1,3], k3)))))
			# y[m,2,n] - y[m,3,n-1] - h∑a[2,j]*k[m,j,n]
			r[ind+M+1:ind+2M] .= (-).(y2, (+).(yprev, (*).(h, (+).((*).(a[2,1], k1), (*).(a[2,2], k2), (*).(a[2,3], k3)))))
			# y[m,3,n] - y[m,3,n-1] - h∑a[3,j]*k[m,j,n]
			r[ind+2M+1:ind+3M] .= (-).(y3, (+).(yprev, (*).(h, (+).((*).(a[3,1], k1), (*).(a[3,2], k2), (*).(a[3,3], k3)))))
            # update base index
            ind += 3M
        end
    end

	# Location for the Jacobian of BC
	dc = Array(eltype(y0), M, 2M)
	# Location for the Jacobian ∂_j[f_i] of f!
	df = Array(eltype(y0), M, M, 3)

	@inbounds function deq!(x, jac)
		jac[:,:] = 0.0
		@inline yind(m :: Int64, i :: Int64, n :: Int64) =
			x[3M*(n-1) + M*(i-3) + m]
		@inline yrange(dest :: Vector{Float64}, i :: Int64, n :: Int64) =
			dest[:] = @view x[3M*(n-1) + M*(i-3) + 1:3M*(n-1) + M*(i-3) + M]
		# Jacobian for BC
		yrange(y3, 3, 1)
		yrange(y2, 3, N)
		dc!(y3, y2, dc)
		@inbounds for m=1:M
			# for t = t0
			jac[m, 1:M] = @view dc[m, 1:M]
			# for t = tf
			jac[m, 3M*(N-1)+1:3M*(N-1)+M] = @view dc[m, M+1:end]
		end
		# Base index
		ind = M
		for n=2:N
			yrange(y1, 1, n)
			yrange(y2, 2, n)
			yrange(y3, 3, n)
			# Get the Jacobian ∂_j[f_i]
			df!(t[n-1]+c[1]*h, y1, (@view df[:,:,1]))
			df!(t[n-1]+c[2]*h, y2, (@view df[:,:,2]))
			df!(t[n-1]+c[3]*h, y3, (@view df[:,:,3]))
			# Jacobian for y[m,i,n] - y[m,3,n-1] - h∑a[i,j] k[m, i, n]
			@inbounds for i=1:3
				for m=1:M
					for ι=1:3
						@simd for μ=1:M
							jac[M+3M*(n-2)+M*(i-1)+m, M+3M*(n-2)+M*(ι-1)+μ] = -h*a[i,ι]*df[m,μ,ι]
						end
					end
					# Diagonal and previous element
					jac[M+3M*(n-2)+M*(i-1)+m, M+3M*(n-2)+M*(i-1)+m] += 1.0
					jac[M+3M*(n-2)+M*(i-1)+m, M+3M*(n-3)+M*(3-1)+m] = -1.0
				end
			end
		end
	end
	x0 = Array(eltype(y0), length(y0))
    x0[:] = y0
    result = nlsolve(eq!, deq!, x0, method = :newton, linesearch! = Optim.backtracking_linesearch!)
	# total time points (including all collocation points)
	t_tot = zeros(Float64, 3*(N-1)+1)
	y_tot = zeros(Float64, (M, length(t_tot)))
	y_tot[:] = result.zero[:]
	# Time with all collocation points
	t_tot[1] = t0
	for n=1:(N-1)
		for i=1:2
			t_tot[1+3*(n-1)+i] = t[n] + c[i]*h
		end
		t_tot[1+3*(n-1)+3] = t[n+1]
	end
	return (result.x_converged || result.f_converged, t_tot, y_tot, result)
end


end # module
