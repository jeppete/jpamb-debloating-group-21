package jpamb.cases;

import java.lang.Math;
import jpamb.utils.*;
import static jpamb.utils.Tag.TagType.*;

public class Debloating {

  // Method 1: Dead code elimination - unreachable code after return
  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int deadCodeAfterReturn(int n) {
    if (n == 0) {
      assert false;
      return 0; // This return is never reached
    }
    // This code is reachable
    int result = n * 2;
    if (false) {
      // Dead code - condition is always false
      assert false;
      return -1;
    }
    return result;
  }

  // Method 2: Unused helper method - helper is defined but never called
  @Case("(3) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CALL })
  public static int unusedHelper(int n) {
    assert n > 0;
    // Helper method computeHelper is defined below but never called
    return n * 2;
  }

  private static int computeHelper(int x) {
    // This method is never actually called - can be removed during debloating
    return x * x * x;
  }

  // Method 3: Redundant checks - multiple checks for same condition
  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int redundantChecks(int n) {
    if (n == 0) {
      assert false;
      return 0;
    }
    // Redundant check - n is already known to be non-zero
    if (n != 0) {
      int result = 100 / n;
      // Another redundant check
      if (n != 0) {
        return result;
      }
    }
    return 0;
  }

  // Method 4: Unused parameter - parameter that doesn't affect outcome
  @Case("(5, 10) -> ok")
  @Case("(0, 20) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int unusedParameter(int n, int unused) {
    // Parameter 'unused' is never actually used
    assert n > 0;
    return 100 / n;
    // 'unused' could be removed without changing behavior
  }

  // Method 5: Unused variables - computed but never used
  @Case("(4) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int unusedVariables(int n) {
    assert n > 0;
    // These variables are computed but never used
    int square = n * n;
    int cube = n * n * n;
    int sum = square + cube;
    // Only 'n' is actually used
    return 100 / n;
  }

  // Method 6: Try-catch bloat - unreachable exception handlers
  @Case("(5) -> ok")
  @Case("(0) -> divide by zero")
  @Tag({ CONDITIONAL })
  public static int unreachableCatch(int n) {
    try {
      return 100 / n;  // Only throws ArithmeticException
    } catch (NullPointerException e) {
      // dead code - no pointers used
      return -1;
    } catch (ArrayIndexOutOfBoundsException e) {
      // dead code - no arrays in this code
      return -2;
    }
  }

  // Method 7: Try-catch bloat - empty finally block
  @Case("(5) -> ok")
  @Case("(-5) -> ok")
  @Tag({ CONDITIONAL })
  public static int emptyFinally(int n) {
    try {
      return n * 2;
    } finally {
      // Empty finally block that does nothing
    }
  }

  // Method 8: Switch statement bloat - dead cases
  @Case("(1) -> ok")
  @Case("(2) -> ok")
  @Tag({ CONDITIONAL })
  public static int switchDeadCases(int n) {
    int x = 1;  // Always 1, making other cases dead
    switch (x) {
      case 1:
        return n * 2;
      case 2:  // x is always 1, this never executes
        return n * 3;
      case 3:  // x is always 1, this never executes
        return n * 4;
      default: // x is always 1, this never executes
        return n * 5;
    }
  }

  // Method 9: Switch statement bloat - empty fall-through cases
  @Case("(1) -> ok")
  @Case("(2) -> ok")
  @Case("(3) -> ok")
  @Tag({ CONDITIONAL })
  public static int switchEmptyFallthrough(int choice) {
    switch (choice) {
      case 1:
        // Empty case that just falls through
      case 2:
        // Empty case that just falls through
      case 3:
        return choice * 2;
      default:
        return 0;
    }
  }

  // Method 10: Complex dead code - logically unreachable after all paths exit
  @Case("(5, true) -> ok")
  @Case("(0, true) -> divide by zero")
  @Case("(5, false) -> ok")
  @Tag({ CONDITIONAL })
  public static int allBranchesExit(int n, boolean flag) {
    if (flag) {
      if (n > 0) {
        return n * 2;
      } else {
        return 100 / n;  // May throw ArithmeticException
      }
    }
    // All paths above return, so this is only reachable if flag is false
    return n + 1;
  }

  // Method 11: Dead code with constant condition
  @Case("(5) -> ok")
  @Case("(10) -> ok")
  @Tag({ LOOP })
  public static int constantConditionDead(int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
      result += i;
      // Condition is always true, so break always executes
      if (1 + 1 == 2) {
        break;
      }
      // This code never executes but is syntactically reachable
      result = result * 2;
    }
    return result;
  }

  // Method 12: Dead code with constant conditions
  @Case("(10) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int constantConditionBlocks(int n) {
    if (n == 0) {
      assert false;
    }
    
    int result = n * 2;
    
    // Condition always false
    if (5 > 10) {
      result = 1 / 0;
      assert false;
    }
    
    // Condition always false
    if (2 + 2 == 5) {
      int x = 1;
      if (x > 0) {
        int y = 2;
        assert false;
      }
    }
    
    return result;
  }

}

