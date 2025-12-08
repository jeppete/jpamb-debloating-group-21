package jpamb.cases;

import java.lang.Math;
import jpamb.utils.*;
import static jpamb.utils.Tag.TagType.*;

public class Debloating {

  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int deadCodeAfterReturn(int n) {
    if (n == 0) {
      assert false;
      return 0;
    }
    int result = n * 2;
    if (false) {
      assert false;
      return -1;
    }
    return result;
  }

  @Case("(3) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CALL })
  public static int unusedHelper(int n) {
    assert n > 0;
    return n * 2;
  }

  private static int computeHelper(int x) {
    return x * x * x;
  }

  @Case("(5) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int redundantChecks(int n) {
    if (n == 0) {
      assert false;
      return 0;
    }
    if (n != 0) {
      int result = 100 / n;
      if (n != 0) {
        return result;
      }
    }
    return 0;
  }

  @Case("(5, 10) -> ok")
  @Case("(0, 20) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int unusedParameter(int n, int unused) {
    assert n > 0;
    return 100 / n;
  }

  @Case("(4) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int unusedVariables(int n) {
    assert n > 0;
    int square = n * n;
    int cube = n * n * n;
    int sum = square + cube;
    return 100 / n;
  }

  @Case("(5) -> ok")
  @Case("(0) -> divide by zero")
  @Tag({ CONDITIONAL })
  public static int unreachableCatch(int n) {
    try {
      return 100 / n;
    } catch (NullPointerException e) {
      return -1;
    } catch (ArrayIndexOutOfBoundsException e) {
      return -2;
    }
  }

  @Case("(5) -> ok")
  @Case("(-5) -> ok")
  @Tag({ CONDITIONAL })
  public static int emptyFinally(int n) {
    try {
      return n * 2;
    } finally {
    }
  }

  @Case("(1) -> ok")
  @Case("(2) -> ok")
  @Tag({ CONDITIONAL })
  public static int switchDeadCases(int n) {
    int x = 1;
    switch (x) {
      case 1:
        return n * 2;
      case 2:
        return n * 3;
      case 3:
        return n * 4;
      default:
        return n * 5;
    }
  }

  @Case("(1) -> ok")
  @Case("(2) -> ok")
  @Case("(3) -> ok")
  @Tag({ CONDITIONAL })
  public static int switchEmptyFallthrough(int choice) {
    switch (choice) {
      case 1:
      case 2:
      case 3:
        return choice * 2;
      default:
        return 0;
    }
  }

  @Case("(5, true) -> ok")
  @Case("(0, true) -> divide by zero")
  @Case("(5, false) -> ok")
  @Tag({ CONDITIONAL })
  public static int allBranchesExit(int n, boolean flag) {
    if (flag) {
      if (n > 0) {
        return n * 2;
      } else {
        return 100 / n;
      }
    }
    return n + 1;
  }

  @Case("(5) -> ok")
  @Case("(10) -> ok")
  @Tag({ LOOP })
  public static int constantConditionDead(int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
      result += i;
      if (1 + 1 == 2) {
        break;
      }
      result = result * 2;
    }
    return result;
  }

  @Case("(10) -> ok")
  @Case("(0) -> assertion error")
  @Tag({ CONDITIONAL })
  public static int constantConditionBlocks(int n) {
    if (n == 0) {
      assert false;
    }
    
    int result = n * 2;
    
    if (5 > 10) {
      result = 1 / 0;
      assert false;
    }
    
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

