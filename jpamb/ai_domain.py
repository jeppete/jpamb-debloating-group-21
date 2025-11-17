from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Literal, TypeAlias


# --- Sign abstract domain ----------------------------------------------------

Sign: TypeAlias = Literal["+", "0", "-"]


@dataclass(frozen=True)
class SignSet:
    """
    Simple sign abstraction for integers.

    Represents a (possibly empty) set of integer signs:
    - "+" for strictly positive
    - "0" for zero
    - "-" for strictly negative
    """
    signs: FrozenSet[Sign]

    # Constructors / special elements -----------------------------------------

    @classmethod
    def bottom(cls) -> "SignSet":
        """Empty set, represents 'no possible value'."""
        return cls(frozenset())

    @classmethod
    def top(cls) -> "SignSet":
        """All signs possible."""
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
        """
        Abstract a (finite) set of integers into a sign set.

        Helpful for property testing and sanity checks.
        """
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

    def __bool__(self) -> bool:
        # bottom is falsy
        return bool(self.signs)

    def __le__(self, other: "SignSet") -> bool:
        """Partial order: subset of signs."""
        return self.signs.issubset(other.signs)

    def __or__(self, other: "SignSet") -> "SignSet":
        """Join (⊔): union of possible signs."""
        return SignSet(self.signs | other.signs)

    def __and__(self, other: "SignSet") -> "SignSet":
        """Meet (⊓): intersection of possible signs."""
        return SignSet(self.signs & other.signs)

    # Convenience operations --------------------------------------------------

    def __contains__(self, n: int) -> bool:
        """Check whether a concrete integer is compatible with this set."""
        if n == 0:
            return "0" in self.signs
        if n > 0:
            return "+" in self.signs
        return "-" in self.signs

    def __repr__(self) -> str:  # pragma: no cover - purely cosmetic
        if not self.signs:
            return "⊥"
        return "{" + ",".join(sorted(self.signs)) + "}"


class SignArithmetic:
    """
    Abstract arithmetic on SignSet.

    The methods are deliberately simple and slightly over-approximate;
    you can refine them later as needed.
    """

    @staticmethod
    def add(a: SignSet, b: SignSet) -> SignSet:
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
                        # Could be anything
                        out.update({"+", "0", "-"})
        return SignSet(frozenset(out)) if out else SignSet.bottom()

    @staticmethod
    def sub(a: SignSet, b: SignSet) -> SignSet:
        """
        Abstract subtraction a - b.
        Implemented by reusing add with negated second argument.
        """
        neg_b = SignSet(
            frozenset({"+": "-", "-": "+", "0": "0"}[s] for s in b.signs)
        )
        return SignArithmetic.add(a, neg_b)

    @staticmethod
    def mul(a: SignSet, b: SignSet) -> SignSet:
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

        Returns (result_signs, may_divide_by_zero).
        If the divisor *might* be zero, we flag that.
        """
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
        return (SignSet(frozenset(out)) if out else SignSet.bottom(),
                may_div_zero)

    # Comparisons -------------------------------------------------------------

    @staticmethod
    def compare_le(a: SignSet, b: SignSet) -> set[bool]:
        """
        Abstract comparison a <= b.

        We return a set of possible boolean results.
        This is intentionally coarse for now: if both outcomes
        are possible, we return {True, False}.
        """
        # You can refine this heavily if you want.
        if not a.signs or not b.signs:
            # Comparing bottoms is undefined; be conservative.
            return {True, False}
        return {True, False}
