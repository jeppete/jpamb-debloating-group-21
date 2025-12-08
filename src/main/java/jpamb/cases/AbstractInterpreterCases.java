package jpamb.cases;

import java.lang.Math;
import jpamb.utils.Case;

public class AbstractInterpreterCases {


  @Case("(5) -> ok")
  @Case("(15) -> ok")
  public static int oldClassicCheck(int x) {
    int n = x*x;

    if (n < 0) {
        return 5;
    }
    
    return 0;
  }

  @Case("(5) -> ok")
  @Case("(15) -> ok")
  public static int contradictoryConditions(int x) {
    if (x > 10) {
      if (x < 5) {
        assert false;
        return -1;
      }
      return x * 2;
    }
    return 0;
  }

  @Case("(5) -> ok")
  @Case("(-5) -> ok")
  public static int signContradiction(int x) {
    if (x > 0) {
      if (x < 0) {
        assert false;
        return -1;
      }
      return x;
    }
    return 0;
  }

  @Case("(5) -> ok")
  @Case("(-5) -> ok")  
  public static int positiveNotZero(int x) {
    if (x > 0) {
      if (x == 0) {
        assert false;
        return -1;
      }
      return x;
    }
    return 0;
  }

  @Case("(4) -> ok")
  @Case("(9) -> ok")
  @Case("(16) -> ok")
  @Case("(5) -> ok")
  public static int squareCheck(int n) {
    if (n < 0) {
      assert false;
      return -1;
    }
    
    int sqrt = (int) Math.sqrt(n);
    if (sqrt * sqrt == n) {
      if (n < 0) {
        assert false;
        return -2;
      }
      return 1;
    } else {
      if (sqrt * sqrt == n) {
        assert false;
        return -3;
      }
      return 0;
    }
  }

  @Case("(3) -> ok")
  @Case("(7) -> ok")
  public static int rangeAnalysis(int x) {
    if (x >= 0 && x <= 10) {
      if (x > 15) {
        assert false;
        return -1;
      }
      if (x < -5) {
        assert false;
        return -2;
      }
      return x;
    }
    return -1;
  }

  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  public static int valuePropagation(int n) {
    int x = n;
    if (x == 0) {
      assert false;
      return 0;
    }
    int y = x;
    if (y == 0) {
      assert false;
      return -1;
    }
    return 100 / y;
  }

  @Case("(10) -> ok")
  @Case("(20) -> ok")
  public static int arithmeticConstraints(int a) {
    int b = a + 5;
    if (a > 10) {
      if (b <= 15) {
        assert false;
        return -1;
      }
      return b;
    }
    return 0;
  }

  @Case("(8) -> ok")
  @Case("(12) -> ok")
  public static int multipleConstraints(int x) {
    if (x > 5 && x < 10) {
      if (x >= 10) {
        assert false;
        return -1;
      }
      if (x <= 5) {
        assert false;
        return -2;
      }
      return x;
    }
    return 0;
  }

  @Case("(5) -> ok")
  public static int loopInvariant(int n) {
    for (int i = 0; i < n; i++) {
      if (i < 0) {
        assert false;
        return -1;
      }
      if (i >= n) {
        assert false;
        return -2;
      }
    }
    return 0;
  }

  @Case("() -> ok")
  public static int constantPropagation() {
    int x = 10;
    int y = x + 5;
    if (y < 10) {
      assert false;
      return -1;
    }
    if (y > 20) {
      assert false;
      return -2;
    }
    return y;
  }

  @Case("(5) -> ok")
  @Case("(10) -> ok")
  public static int divisionByZeroTracking(int n) {
    if (n > 0) {
      int x = n - n;
      if (x == 0) {
        if (100 / x > 0) {
          assert false;
          return -1;
        }
      }
    }
    return 0;
  }
  @Case("(6) -> ok")
  @Case("(8) -> ok")
  public static int nestedValueDependencies(int x) {
    if (x > 5) {
      int y = x - 1;
      if (y > 4) {
        if (y <= 4) {
          assert false;
          return -1;
        }
        return y;
      }
    }
    return 0;
  }

  @Case("(10) -> ok")
  @Case("(11) -> ok")
  public static int moduloAnalysis(int n) {
    int remainder = n % 2;
    if (remainder == 0) {
      if (remainder == 1) {
        assert false;
        return -1;
      }
      return 0;
    } else {
      if (remainder == 0) {
        assert false;
        return -2;
      }
      return 1;
    }
  }

  @Case("(3) -> ok")
  @Case("(7) -> ok")
  public static int complexArithmetic(int a) {
    int b = a * 2;
    int c = b + 1;
    
    if (a > 0) {
      if (c <= 1) {
        assert false;
        return -1;
      }
      if (b <= 0) {
        assert false;
        return -2;
      }
      return c;
    }
    return 0;
  }

  @Case("(5) -> ok")
  @Case("(10) -> ok")
  public static int arrayBoundsTracking(int size) {
    if (size > 0 && size <= 100) {
      int[] arr = new int[size];
      int index = size - 1;
      
      if (index >= 0 && index < size) {
        if (index < 0) {
          assert false;
          return -1;
        }
        if (index >= size) {
          assert false;
          return -2;
        }
        return arr[index];
      }
    }
    return 0;
  }

  @Case("(true) -> ok")
  @Case("(false) -> ok")
  public static int booleanValueTracking(boolean flag) {
    if (flag) {
      if (!flag) {
        assert false;
        return -1;
      }
      return 1;
    } else {
      if (flag) {
        assert false;
        return -2;
      }
      return 0;
    }
  }

  @Case("(7) -> ok")
  @Case("(8) -> ok")
  public static int chainedComparisons(int x) {
    if (x > 5 && x < 10) {
      if (x == 5) {
        assert false;
        return -1;
      }
      if (x == 10) {
        assert false;
        return -2;
      }
      if (x <= 5) {
        assert false;
        return -3;
      }
      if (x >= 10) {
        assert false;
        return -4;
      }
      return x;
    }
    return 0;
  }

}

