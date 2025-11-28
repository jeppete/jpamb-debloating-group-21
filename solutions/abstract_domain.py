# solutions/ai_domain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Literal, TypeAlias


# --- Sign abstract domain ----------------------------------------------------


Sign: TypeAlias = Literal["+", "0", "-"]


@dataclass(frozen=True)
class SignSet:
    """
    Sign abstraction for integers:

      "+"   – positive
      "0"   – zero
      "-"   – negative

    Lattice order:   subset on the underlying sign sets
    Bottom element:  ⊥ = ∅
    Top element:     ⊤ = {"+", "0", "-"}
    """

    signs: FrozenSet[Sign]

    # Constructors / special elements -----------------------------------------

    @classmethod
    def bottom(cls) -> "SignSet":
        """The impossible abstract value."""
        return cls(frozenset())

    @classmethod
    def top(cls) -> "SignSet":
        """The completely unknown integer."""
        return cls(frozenset({"+", "0", "-"}))

    @classmethod
    def const(cls, n: int) -> "SignSet":
        """Abstract a single concrete integer."""
        if n == 0:
            s = {"0"}
        elif n > 0:
            s = {"+"}
        else:
            s = {"-"}
        return cls(frozenset(s))

    @classmethod
    def abstract(cls, values: Iterable[int]) -> "SignSet":
        """Best abstraction of a finite set of integers."""
        s: set[Sign] = set()
        for v in values:
            if v == 0:
                s.add("0")
            elif v > 0:
                s.add("+")
            else:
                s.add("-")
        return cls(frozenset(s))

    # Lattice structure -------------------------------------------------------

    def is_bottom(self) -> bool:
        return not self.signs

    def is_top(self) -> bool:
        return self.signs == {"+", "0", "-"}

    def __bool__(self) -> bool:
        # bool(⊥) is False, everything else is True.
        return bool(self.signs)

    def __le__(self, other: "SignSet") -> bool:
        return self.signs.issubset(other.signs)

    def __or__(self, other: "SignSet") -> "SignSet":
        return SignSet(self.signs | other.signs)

    def __and__(self, other: "SignSet") -> "SignSet":
        return SignSet(self.signs & other.signs)

    # Convenience -------------------------------------------------------------

    def __contains__(self, n: int) -> bool:
        """
        n ∈ self  iff  sign(n) ∈ self.signs
        """
        if n == 0:
            return "0" in self.signs
        if n > 0:
            return "+" in self.signs
        return "-" in self.signs

    def __repr__(self) -> str:  # cosmetic
        if not self.signs:
            return "⊥"
        if self.signs == {"+", "0", "-"}:
            return "⊤"
        return "{" + ",".join(sorted(self.signs)) + "}"


class SignArithmetic:
    """
    Operations enumerate all combinations of operand signs.  This keeps
    things simple and guarantees monotonicity for the unbounded analysis
    """

    @staticmethod
    def add(a: SignSet, b: SignSet) -> SignSet:
        if a.is_bottom() or b.is_bottom():
            return SignSet.bottom()

        out: set[Sign] = set()
        for sa in a.signs:
            for sb in b.signs:
                match (sa, sb):
                    case ("0", "0"):
                        out.add("0")
                    case ("0", "+") | ("+", "0") | ("+", "+"):
                        out.add("+")
                    case ("0", "-") | ("-", "0") | ("-", "-"):
                        out.add("-")
                    case ("+", "-") | ("-", "+"):
                        out.update({"+", "0", "-"})
        return SignSet(frozenset(out)) if out else SignSet.bottom()

    @staticmethod
    def sub(a: SignSet, b: SignSet) -> SignSet:
        if a.is_bottom() or b.is_bottom():
            return SignSet.bottom()

        m = {"+": "-", "-": "+", "0": "0"}
        neg_b = SignSet(frozenset(m[s] for s in b.signs))
        return SignArithmetic.add(a, neg_b)

    @staticmethod
    def mul(a: SignSet, b: SignSet) -> SignSet:
        if a.is_bottom() or b.is_bottom():
            return SignSet.bottom()

        out: set[Sign] = set()
        for sa in a.signs:
            for sb in b.signs:
                match (sa, sb):
                    case ("0", _) | (_, "0"):
                        out.add("0")
                    case ("+", "+") | ("-", "-"):
                        out.add("+")
                    case ("+", "-") | ("-", "+"):
                        out.add("-")
        return SignSet(frozenset(out)) if out else SignSet.bottom()

    @staticmethod
    def div(a: SignSet, b: SignSet) -> tuple[SignSet, bool]:
        """
        Abstract division a / b.

        Returns:
          (result_signs, may_divide_by_zero)

        The result only accounts for the non‑zero divisor cases; if the
        divisor is definitely zero, the result is ⊥
        """
        if a.is_bottom() or b.is_bottom():
            return (SignSet.bottom(), False)

        out: set[Sign] = set()
        may_div_zero = "0" in b.signs
        non_zero_b = SignSet(frozenset(s for s in b.signs if s != "0"))

        for sa in a.signs:
            for sb in non_zero_b.signs:
                match (sa, sb):
                    case ("0", _):
                        out.add("0")
                    case ("+", "+") | ("-", "-"):
                        out.add("+")
                    case ("+", "-") | ("-", "+"):
                        out.add("-")
        return (SignSet(frozenset(out)) if out else SignSet.bottom(), may_div_zero)

    @staticmethod
    def rem(a: SignSet, b: SignSet) -> tuple[SignSet, bool]:
        """
        Abstract remainder a % b

        JVM requires b != 0 at runtime; we flag if b might be 0.
        Sign of the remainder follows the dividend (except it can be 0)

        As with div, the result only accounts for the non‑zero divisor
        cases; if the divisor is definitely zero, the result is ⊥
        """
        if a.is_bottom() or b.is_bottom():
            return (SignSet.bottom(), False)

        may_div_zero = "0" in b.signs

        # If we only know that b == 0 then there is no non‑exceptional case.
        non_zero_possible = any(s != "0" for s in b.signs)
        if not non_zero_possible:
            return (SignSet.bottom(), may_div_zero)

        out: set[Sign] = set()
        for sa in a.signs:
            if sa == "0":
                out.add("0")
            elif sa == "+":
                out.update({"+", "0"})
            elif sa == "-":
                out.update({"-", "0"})
        return (SignSet(frozenset(out)) if out else SignSet.bottom(), may_div_zero)

    @staticmethod
    def neg(a: SignSet) -> SignSet:
        """
        Unary negate
        """
        if a.is_bottom():
            return SignSet.bottom()

        m = {"+": "-", "-": "+", "0": "0"}
        return SignSet(frozenset(m[s] for s in a.signs))
