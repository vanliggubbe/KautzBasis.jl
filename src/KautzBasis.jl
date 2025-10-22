module KautzBasis

import Polynomials: Polynomial, FactoredPolynomial, RationalFunction, AbstractRationalFunction, numerator, denominator, derivative, degree
import LinearAlgebra: dot

struct Kautz{T <: Number}
    p :: Vector{T}
    Kautz(p) = (
        all(real(p) .> 0.0) ?
        new(p) :
        throw(ArgumentError("Modes of the Kautz basis must have positive decay rate"))
    )
end

struct KautzFunction{S, T} <: AbstractRationalFunction{T, :ω, FactoredPolynomial{T, :ω}}
    basis :: Kautz{S}
    n :: Int
    f :: RationalFunction{T, :ω, FactoredPolynomial{T, :ω}}
end

function Base.getindex(f :: Kautz{T}, n :: Int) where {T}
    if n < 1
        throw(BoundsError(f, n))
    end
    m = (n - 1) ÷ length(f)
    k = (n - 1) % length(f)
    num = Dict{promote_type(T, Complex), Int}()
    den = Dict{promote_type(T, Complex), Int}()
    if m > 0
        for s in f.p
            if !haskey(num, conj(-im * s))
                num[conj(-im * s)] = m
            else
                num[conj(-im * s)] += m
            end
            if !haskey(den, -im * s)
                den[-im * s] = m
            else
                den[-im * s] += m
            end
        end
    end
    for s in f.p[1 : k]
        if !haskey(num, conj(-im * s))
            num[conj(-im * s)] = 1
        else
            num[conj(-im * s)] += 1
        end
        if !haskey(den, -im * s)
            den[-im * s] = 1
        else
            den[-im * s] += 1
        end
    end
    if !haskey(den, -im * f.p[k + 1])
        den[-im * f.p[k + 1]] = 1
    else
        den[-im * f.p[k + 1]] += 1
    end
    return KautzFunction(
        f, n,
        FactoredPolynomial(num, sqrt(2.0 * real(f.p[k + 1])), :ω) // FactoredPolynomial(den, 1.0, :ω)
    )
end

Base.show(io :: IO, f :: KautzFunction) = show(io, f)
(f :: KautzFunction)(x) = f.f(x)

Base.:*(f :: KautzFunction, g :: KautzFunction) = f.f * g.f
Base.:*(f :: KautzFunction, g :: Number) = f.f * g
Base.:*(f :: Number, g :: KautzFunction) = f * g.f

Base.:/(f :: KautzFunction, g :: KautzFunction) = f.f / g.f
Base.:/(f :: KautzFunction, g :: Number) = f.f / g
Base.:/(f :: Number, g :: KautzFunction) = f / g.f

dot(f :: KautzFunction{S, T}, g :: KautzFunction{SS, TT}) where {S, T, SS, TT} = (
    f.basis === g.basis ? (
        f.n == g.n ? one(promote_type(T, TT)) : zero(promote_type(T, TT))
    ) : integral(conj(numerator(f.f)) * numerator(g.f) // conj(denominator(f.f)) * denominator(g.f))
)

function integral(f :: RationalFunction{T, :ω, <: FactoredPolynomial})
    num, den = Polynomial(numerator(f)), denominator(f)
    if degree(num) >= degree(den) - 1
        error(ArgumentError("Integral diverges"))
    end
    den′ = copy(den.coeff)
    ret = zero(promote_type(Complex, T))
    for (pol, deg) in den.coeff
        if iszero(imag(pol))
            error(ArgumentError("Integral diverges"))
        elseif imag(pol) > 0.0
            pop!(den′, pol)
            res += im * derivative(
                num // Polynomial(FactoredPolynomial(den′), den.c),
                deg - 1
            )(pol) / factorial(deg - 1)
            den′[pol] = deg
        end
    end
    return ret
end

Base.length(a :: Kautz) = length(a.p)

function freq(f :: Kautz, n :: Int, ω)
    m = (n - 1) ÷ length(f)
    k = (n - 1) % length(f)
    tmp = [1.0; cumprod((ω .- im * conj(f.p)) ./ (ω .+ im * f.p))]
    return (tmp[end] ^ m) * tmp[k + 1] * sqrt(2.0 * real(f.p[k]))
end


end
