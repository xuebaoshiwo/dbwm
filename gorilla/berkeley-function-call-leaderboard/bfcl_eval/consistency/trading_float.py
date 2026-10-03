"""Binary64 arithmetic and Python round(x, 2/3) for finite numbers.

The middle branch rounds the exact significand * 5**digits / 2**shift
to an integer (ties to even), then converts that exact integer / 10**digits
back to binary64. It deliberately does NOT compute round(x * 100) / 100:
that loses the low bits responsible for Python's round(2.675, 2) == 2.67.
"""

import math
import struct
from decimal import Decimal

import z3


F64 = z3.Float64()
RNE = z3.RNE()


def fp(value):
    return value if isinstance(value, z3.FPRef) else z3.FPVal(float(value), F64)


def finite(value):
    return z3.And(z3.Not(z3.fpIsNaN(value)), z3.Not(z3.fpIsInf(value)))


def rounded(value, digits=2):
    if digits not in (2, 3):
        raise ValueError("Only round(..., 2) and round(..., 3) are modeled")
    value = z3.simplify(fp(value))
    if z3.is_app_of(value, z3.Z3_OP_ITE):
        return z3.If(value.arg(0), rounded(value.arg(1), digits), rounded(value.arg(2), digits))
    if z3.is_app(value) and value.decl().kind() in (z3.Z3_OP_FPA_ADD, z3.Z3_OP_FPA_SUB, z3.Z3_OP_FPA_MUL):
        args = value.children()
        for i, arg in enumerate(args):
            if z3.is_app_of(arg, z3.Z3_OP_ITE):
                left, right = list(args), list(args)
                left[i], right[i] = arg.arg(1), arg.arg(2)
                return z3.If(arg.arg(0), rounded(value.decl()(*left), digits), rounded(value.decl()(*right), digits))
    if isinstance(value, z3.FPNumRef):
        return fp(round(as_float(value), digits))
    bits = z3.fpToIEEEBV(value)
    exponent = z3.ZeroExt(53, z3.Extract(62, 52, bits))
    significand = z3.ZeroExt(12, z3.Extract(51, 0, bits)) | (1 << 52)
    numerator = significand * (5 ** digits)
    shift = z3.BitVecVal(1075 - digits, 64) - exponent
    quotient = z3.LShR(numerator, shift)
    remainder = numerator & ((z3.BitVecVal(1, 64) << shift) - 1)
    half = z3.BitVecVal(1, 64) << (shift - 1)
    up = z3.Or(z3.UGT(remainder, half), z3.And(remainder == half, quotient & 1 == 1))
    nearest = quotient + z3.If(up, z3.BitVecVal(1, 64), z3.BitVecVal(0, 64))
    magnitude = z3.fpDiv(RNE, z3.fpToFPUnsigned(RNE, nearest, F64), fp(10 ** digits))
    signed = z3.If(z3.fpIsNegative(value), z3.fpNeg(magnitude), magnitude)
    zero = z3.If(z3.fpIsNegative(value), fp(-0.0), fp(0.0))
    # Beyond this binade the spacing exceeds a decimal unit. Rounding the
    # exact decimal value back to binary64 necessarily returns the input.
    cutoff = 46 if digits == 2 else 43
    return z3.If(z3.fpAbs(value) >= fp(2 ** cutoff), value,
                 z3.If(z3.fpAbs(value) < fp(0.5 / 10 ** digits), zero, signed))


def three_decimals(value):
    return z3.And(finite(value), z3.fpEQ(value, rounded(value, 3)))


def observed(value, number, tolerance=Decimal("0.000000001")):
    """Match the same absolute decimal-text tolerance used by replay."""
    center = Decimal(str(number))
    low, high = center - tolerance, center + tolerance
    lower, upper = float(low), float(high)
    while Decimal(str(lower)) < low:
        lower = math.nextafter(lower, math.inf)
    while Decimal(str(upper)) > high:
        upper = math.nextafter(upper, -math.inf)
    return z3.And(value >= fp(lower), value <= fp(upper))


def as_float(value, model=None):
    bits = z3.fpToIEEEBV(value)
    bits = model.eval(bits, model_completion=True) if model is not None else z3.simplify(bits)
    return struct.unpack(">d", bits.as_long().to_bytes(8, "big"))[0]
