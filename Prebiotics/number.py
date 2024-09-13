class Number:
    def __init__(self, value = None, decimals = 15):
        self.decimals = decimals
        self.factor = 10 ** self.decimals
        self.value = round(value * self.factor)
        self.float_value = self.value / self.factor

    def __repr__(self):
        return f"{self.float_value:.{self.decimals}f}"

    def __float__(self):
        return self.float_value

    def __int__(self):
        return int(self.float_value)

    def decimal(self, N):
        self.value = round(self.value * 10 ** (N - self.decimals))
        self.decimals = N
        self.factor = 10 ** self.decimals
        self.float_value = self.value / self.factor
        return self

    def __add__(self, other):
        if isinstance(other, Number):
            if self.decimals >= other.decimals:
                diff = self.decimals - other.decimals
                return Number(
                    (self.value + other.value * 10 ** diff) / self.factor,
                    other.decimals
                )
            else:
                diff = other.decimals - self.decimals
                return Number(
                    (self.value * 10 ** diff + other.value) / other.factor,
                    self.decimals
                )
        return self + Number(other, self.decimals)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            if self.decimals >= other.decimals:
                diff = self.decimals - other.decimals
                return Number(
                    (self.value - other.value * 10 ** diff) / self.factor,
                    other.decimals
                )
            else:
                diff = other.decimals - self.decimals
                return Number(
                    (self.value * 10 ** diff - other.value) / other.factor,
                    self.decimals
                )
        return self - Number(other, self.decimals)

    def __rsub__(self, other):
        return Number(other, self.decimals) - self

    def __mul__(self, other):
        if isinstance(other, Number):
            return Number(
                self.value * other.value / (self.factor * other.factor),
                min(self.decimals, other.decimals)
            )
        return self * Number(other, self.decimals)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            if self.decimals >= other.decimals:
                diff = self.decimals - other.decimals
                return Number(
                    round(
                        self.value / (other.value * 10 ** diff),
                        other.decimals
                    ),
                    other.decimals
                )
            else:
                diff = other.decimals - self.decimals
                return Number(
                    round(
                        self.value * 10 ** diff / other.value,
                        self.decimals
                    ),
                    self.decimals
                )
        return self / Number(other, self.decimals)

    def __rtruediv__(self, other):
        return Number(other, self.decimals) / self

    def __pow__(self, other):
        if isinstance(other, Number):
            return Number(
                self.float_value ** other.float_value,
                self.decimals
            )
        return self ** Number(other, self.decimals)

    def __rpow__(self, other):
        return Number(other, self.decimals) ** self

    def __floordiv__(self, other):
        if isinstance(other, Number):
            if self.decimals >= other.decimals:
                diff = self.decimals - other.decimals
                return Number(
                    self.value // (other.value * 10 ** diff),
                    other.decimals
                )
            else:
                diff = other.decimals - self.decimals
                return Number(
                    self.value * 10 ** diff // other.value,
                    self.decimals
                )
        return self // Number(other, self.decimals)

    def __rfloordiv__(self, other):
        return Number(other, self.decimals) // self

    def __mod__(self, other):
        if isinstance(other, Number):
            if self.decimals >= other.decimals:
                diff = self.decimals - other.decimals
                return Number(
                    self.value % (other.value * 10 ** diff),
                    other.decimals
                )
            else:
                diff = other.decimals - self.decimals
                return Number(
                    (self.value * 10 ** diff) % other.value,
                    self.decimals
                )
        return self % Number(other, self.decimals)

    def __rmod__(self, other):
        return Number(other, self.decimals) % self

    def __divmod__(self, other):
        if isinstance(other, Number):
            return (self.__floordiv__(other), self.__mod__(other))
        return divmod(self, Number(other, self.decimals))

    def __rdivmod__(self, other):
        return divmod(Number(other, self.decimals), self)

    def __iadd__(self, other):
        result = self + other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __isub__(self, other):
        result = self - other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __imul__(self, other):
        result = self * other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __itruediv__(self, other):
        result = self / other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __ipow__(self, other):
        result = self ** other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __ifloordiv__(self, other):
        result = self / other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __imod__(self, other):
        result = self % other
        self.value = result.value
        self.decimals = result.decimals
        self.float_value = self.value / self.factor
        return self

    def __eq__(self, other):
        if isinstance(other, Number):
            return self.float_value == other.float_value
        return self == Number(other, self.decimals)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Number):
            return self.float_value < other.float_value
        return self < Number(other, self.decimals)

    def __le__(self, other):
        if isinstance(other, Number):
            return self.float_value <= other.float_value
        return self <= Number(other, self.decimals)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __pos__(self):
        return self

    def __neg__(self):
        return Number(-self.float_value, self.decimals)

    def __abs__(self):
        return Number(abs(self.float_value), self.decimals)

    def __bool__(self):
        return bool(self.value)

    def __copy__(self):
        return Number(self.float_value, self.decimals)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __hash__(self):
        return hash((self.value, self.decimals))
