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
  // Case 1: Contradictory conditions - abstract interpreter should detect
  // that the inner branch is always false
  @Case("(5) -> ok")
  @Case("(15) -> ok")
  public static int contradictoryConditions(int x) {
    if (x > 10) {
      // This branch is reachable
      if (x < 5) {
        // Dead code: x > 10 && x < 5 is impossible
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
      // x is positive here
      if (x < 0) {
        // Dead code: x > 0 && x < 0 is impossible
        // Sign abstraction: positive AND negative is empty set
        assert false;
        return -1;
      }
      return x;
    }
    return 0;
  }

  // Another sign-based test: checking zero after positive
  @Case("(5) -> ok")
  @Case("(-5) -> ok")  
  public static int positiveNotZero(int x) {
    if (x > 0) {
      // x is positive here
      if (x == 0) {
        // Dead code: positive cannot be zero
        assert false;
        return -1;
      }
      return x;
    }
    return 0;
  }

  // Case 2: Square check - abstract interpreter should detect dead branches
  // based on value analysis
  @Case("(4) -> ok")
  @Case("(9) -> ok")
  @Case("(16) -> ok")
  @Case("(5) -> ok")
  public static int squareCheck(int n) {
    // Check if n is a perfect square
    if (n < 0) {
      // Dead code if we can prove n >= 0 from context
      assert false;
      return -1;
    }
    
    int sqrt = (int) Math.sqrt(n);
    if (sqrt * sqrt == n) {
      // Perfect square
      if (n < 0) {
        // Dead code: we already know n >= 0
        assert false;
        return -2;
      }
      return 1;
    } else {
      // Not a perfect square
      if (sqrt * sqrt == n) {
        // Dead code: this condition is always false here
        assert false;
        return -3;
      }
      return 0;
    }
  }

  // Case 3: Range analysis - abstract interpreter should track value ranges
  @Case("(3) -> ok")
  @Case("(7) -> ok")
  public static int rangeAnalysis(int x) {
    if (x >= 0 && x <= 10) {
      // x is in [0, 10]
      if (x > 15) {
        // Dead code: x <= 10, so x > 15 is impossible
        assert false;
        return -1;
      }
      if (x < -5) {
        // Dead code: x >= 0, so x < -5 is impossible
        assert false;
        return -2;
      }
      return x;
    }
    return -1;
  }

  // Case 4: Value propagation through assignments
  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  public static int valuePropagation(int n) {
    int x = n;
    if (x == 0) {
      assert false;
      return 0;
    }
    // After the check, x != 0
    int y = x;
    if (y == 0) {
      // Dead code: y == x and x != 0
      assert false;
      return -1;
    }
    return 100 / y;
  }

  // Case 5: Arithmetic constraints
  @Case("(10) -> ok")
  @Case("(20) -> ok")
  public static int arithmeticConstraints(int a) {
    int b = a + 5;
    if (a > 10) {
      // b = a + 5 and a > 10, so b > 15
      if (b <= 15) {
        // Dead code: b > 15, so b <= 15 is false
        assert false;
        return -1;
      }
      return b;
    }
    return 0;
  }

  // Case 6: Multiple constraints creating contradiction
  @Case("(8) -> ok")
  @Case("(12) -> ok")
  public static int multipleConstraints(int x) {
    if (x > 5 && x < 10) {
      // x is in (5, 10)
      if (x >= 10) {
        // Dead code: x < 10, so x >= 10 is false
        assert false;
        return -1;
      }
      if (x <= 5) {
        // Dead code: x > 5, so x <= 5 is false
        assert false;
        return -2;
      }
      return x;
    }
    return 0;
  }

  // Case 7: Loop-invariant dead code
  @Case("(5) -> ok")
  public static int loopInvariant(int n) {
    for (int i = 0; i < n; i++) {
      // i is in [0, n)
      if (i < 0) {
        // Dead code: loop ensures i >= 0
        assert false;
        return -1;
      }
      if (i >= n) {
        // Dead code: loop condition ensures i < n
        assert false;
        return -2;
      }
    }
    return 0;
  }

  // Case 8: Constant propagation with dead branches
  @Case("() -> ok")
  public static int constantPropagation() {
    int x = 10;
    int y = x + 5; // y = 15
    if (y < 10) {
      // Dead code: y = 15, so y < 10 is false
      assert false;
      return -1;
    }
    if (y > 20) {
      // Dead code: y = 15, so y > 20 is false
      assert false;
      return -2;
    }
    return y;
  }

  // Case 9: Division by zero detection through value tracking
  @Case("(5) -> ok")
  @Case("(10) -> ok")
  public static int divisionByZeroTracking(int n) {
    if (n > 0) {
      int x = n - n; // x = 0
      if (x == 0) {
        // x is definitely 0
        if (100 / x > 0) {
          // Dead code: division by zero
          assert false;
          return -1;
        }
      }
    }
    return 0;
  }

  // Case 10: Nested conditions with value dependencies
  @Case("(6) -> ok")
  @Case("(8) -> ok")
  public static int nestedValueDependencies(int x) {
    if (x > 5) {
      int y = x - 1; // y > 4
      if (y > 4) {
        // This branch is always taken when x > 5
        if (y <= 4) {
          // Dead code: y > 4, so y <= 4 is false
          assert false;
          return -1;
        }
        return y;
      }
    }
    return 0;
  }

  // Case 11: Modulo-based dead code detection
  @Case("(10) -> ok")
  @Case("(11) -> ok")
  public static int moduloAnalysis(int n) {
    int remainder = n % 2;
    if (remainder == 0) {
      // n is even
      if (remainder == 1) {
        // Dead code: remainder is 0, so remainder == 1 is false
        assert false;
        return -1;
      }
      return 0;
    } else {
      // n is odd
      if (remainder == 0) {
        // Dead code: remainder is 1, so remainder == 0 is false
        assert false;
        return -2;
      }
      return 1;
    }
  }

  // Case 12: Complex arithmetic with dead branches
  @Case("(3) -> ok")
  @Case("(7) -> ok")
  public static int complexArithmetic(int a) {
    int b = a * 2; // b = 2a
    int c = b + 1; // c = 2a + 1
    
    if (a > 0) {
      // a > 0, so b = 2a > 0, and c = 2a + 1 > 1
      if (c <= 1) {
        // Dead code: c > 1, so c <= 1 is false
        assert false;
        return -1;
      }
      if (b <= 0) {
        // Dead code: b > 0, so b <= 0 is false
        assert false;
        return -2;
      }
      return c;
    }
    return 0;
  }

  // Case 13: Array bounds with value tracking
  @Case("(5) -> ok")
  @Case("(10) -> ok")
  public static int arrayBoundsTracking(int size) {
    if (size > 0 && size <= 100) {
      int[] arr = new int[size];
      int index = size - 1; // index is in [0, size-1]
      
      if (index >= 0 && index < size) {
        // This branch is always taken
        if (index < 0) {
          // Dead code: index >= 0
          assert false;
          return -1;
        }
        if (index >= size) {
          // Dead code: index < size
          assert false;
          return -2;
        }
        return arr[index];
      }
    }
    return 0;
  }

  // Case 14: Boolean logic with value constraints
  @Case("(true) -> ok")
  @Case("(false) -> ok")
  public static int booleanValueTracking(boolean flag) {
    if (flag) {
      // flag is true
      if (!flag) {
        // Dead code: flag is true, so !flag is false
        assert false;
        return -1;
      }
      return 1;
    } else {
      // flag is false
      if (flag) {
        // Dead code: flag is false, so flag is false
        assert false;
        return -2;
      }
      return 0;
    }
  }

  // Case 15: Chained comparisons
  @Case("(7) -> ok")
  @Case("(8) -> ok")
  public static int chainedComparisons(int x) {
    if (x > 5 && x < 10) {
      // x is in (5, 10)
      if (x == 5) {
        // Dead code: x > 5, so x == 5 is false
        assert false;
        return -1;
      }
      if (x == 10) {
        // Dead code: x < 10, so x == 10 is false
        assert false;
        return -2;
      }
      if (x <= 5) {
        // Dead code: x > 5, so x <= 5 is false
        assert false;
        return -3;
      }
      if (x >= 10) {
        // Dead code: x < 10, so x >= 10 is false
        assert false;
        return -4;
      }
      return x;
    }
    return 0;
  }

}

