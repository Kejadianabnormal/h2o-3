package hex.glm;

import Jama.Matrix;
import hex.DataInfo;
import water.DKV;
import water.Iced;
import water.Key;
import water.Scope;
import water.fvec.Frame;
import water.util.IcedHashMap;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static water.util.ArrayUtils.innerProduct;

public class ConstrainedGLMUtils {
  // constant setting refer to Michel Bierlaire, Optimization: Principles and Algorithms, Chapter 19, EPEL Press,
  // second edition, 2018.
  public static final double C0CS = 10.0; // also value of Tau
  public static final double ETA0CS = 0.1258925;
  public static final double ALPHACS = 0.1;
  public static final double BETACS = 0.9;
  public static final double EPSILON0 = 0.1;
  public static final double EPS = 1e-15;
  
  public static class LinearConstraints extends Iced { // store one linear constraint
    public IcedHashMap<String, Double> _constraints; // column names, coefficient of constraints
    public double _constraintsVal; // contains evaluated constraint values
    public boolean _active = true; // only applied to less than and equal to zero constraints
    
    public LinearConstraints() {
      _constraints = new IcedHashMap<>();
      _constraintsVal = Double.NaN; // represent constraint not evaluated.
    }
  }
  
  public static class ConstraintsDerivatives extends Iced {
    public IcedHashMap<Integer, Double> _constraintsDerivative;
    public boolean _active;
    
    public ConstraintsDerivatives() {
      _constraintsDerivative = new IcedHashMap<>();
    }
  }
  
  public static class ConstraintsGram extends Iced {
    public IcedHashMap<CoefIndices, Double> _coefIndicesValue;
    public boolean _active;
    
    public ConstraintsGram() {
      _coefIndicesValue = new IcedHashMap<>();
    }
  }
  
  public static class CoefIndices implements hex.glm.CoefIndices {
    final int _firstCoefIndex;
    final int _secondCoefIndex;
    
    public CoefIndices(int firstInd, int secondInd) {
      _firstCoefIndex = firstInd;
      _secondCoefIndex = secondInd;
    }
    
    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      else if (o == null)
        return false;
      else if (this._firstCoefIndex == ((CoefIndices) o)._firstCoefIndex && 
              this._secondCoefIndex == ((CoefIndices) o)._secondCoefIndex)
        return true;
      return false;
    }
    
    public String toString() {
      return "first coefficient index: " + _firstCoefIndex + ", second coefficient index " + _secondCoefIndex;
    }
  }
  
  public static class ConstraintGLMStates {
    double _ckCS = C0CS;
    double _epsilonkCS = 1.0/C0CS;
    double _etakCS = ETA0CS*Math.pow(C0CS, ALPHACS);
    String[] _constraintNames;
    double[][] _initCSMatrix;
    double _gradientMag;
    double _constraintMagSquare;
    
    
    public ConstraintGLMStates(String[] constrainNames, double[][] initMatrix) {
      _constraintNames = constrainNames;
      _initCSMatrix = initMatrix;
    }
  }
  
  public static LinearConstraints[] combineConstraints(LinearConstraints[] const1, LinearConstraints[] const2) {
    List<LinearConstraints> allList = new ArrayList<>();
    if (const1 != null)
      allList.addAll(Arrays.stream(const1).collect(Collectors.toList()));
    if (const2 != null)
      allList.addAll(Arrays.stream(const2).collect(Collectors.toList()));
    return allList.size()==0 ? null : allList.stream().toArray(LinearConstraints[]::new);
  }

  /***
   *
   * This method will extract the constraints specified in beta constraint and combine it with the linear constraints
   * later.  Note that the linear constraints are only accepted in standard form, meaning we only accept the following
   * constraint forms:  2*beta_1-3*beta_4-3 == 0 or 2*beta_1-3*beta_4-3 <= 0.
   * 
   * The beta constraints on the other hand is specified in several forms:
   * 1): -Infinity <= beta <= Infinity: ignored, no constrain here;
   * 2): -Infinity <= beta <= high_val: transformed to beta - high_val <= 0, add to lessThanEqualTo constraint;
   * 3): low_val <= beta <= Infinity: transformed to low_val - beta <= 0, add to lessThanEqualTo constraint;
   * 4): low_val <= beta <= high_val: transformed to two constraints, low_val-beta <= 0, beta-high_val <= 0, add to lessThanEqualTo constraint;
   * 5): val <= beta <= val: transformed to beta-val == 0, add to equalTo constraint.
   * 
   * The newly extracted constraints will be added to fields in state.
   * 
   */
  public static void extractBetaConstraints(ComputationState state, String[] coefNames) {
    GLM.BetaConstraint betaC = state.activeBC();
    List<LinearConstraints> equalityC = new ArrayList<>();
    List<LinearConstraints> lessThanEqualToC = new ArrayList<>();
    if (betaC._betaLB != null) {
      int numCons = betaC._betaLB.length-1;
      for (int index=0; index<numCons; index++) {
        if (!Double.isInfinite(betaC._betaUB[index]) && (betaC._betaLB[index] == betaC._betaUB[index])) { // equality constraint
          addBCEqualityConstraint(equalityC, betaC, coefNames, index);
        } else if (!Double.isInfinite(betaC._betaUB[index]) && !Double.isInfinite(betaC._betaLB[index]) && 
                (betaC._betaLB[index] < betaC._betaUB[index])) { // low < beta < high, generate two lessThanEqualTo constraints
          addBCGreaterThanConstraint(lessThanEqualToC, betaC, coefNames, index);
          addBCLessThanConstraint(lessThanEqualToC, betaC, coefNames, index);
        } else if (Double.isInfinite(betaC._betaUB[index]) && !Double.isInfinite(betaC._betaLB[index])) {  // low < beta < infinity
          addBCGreaterThanConstraint(lessThanEqualToC, betaC, coefNames, index);
        } else if (!Double.isInfinite(betaC._betaUB[index]) && Double.isInfinite(betaC._betaLB[index])) { // -infinity < beta < high
          addBCLessThanConstraint(lessThanEqualToC, betaC, coefNames, index);
        }
      }
    }
    state.setLinearConstraints(equalityC.toArray(new LinearConstraints[0]),
            lessThanEqualToC.toArray(new LinearConstraints[0]), true);
  }

  /***
   * This method will extract the equality constraint and add to equalityC from beta constraint by doing the following
   * transformation: val <= beta <= val: transformed to beta-val == 0, add to equalTo constraint.
   */
  public static void addBCEqualityConstraint(List<LinearConstraints> equalityC, GLM.BetaConstraint betaC,
                                           String[] coefNames, int index) {
    LinearConstraints oneEqualityConstraint = new LinearConstraints();
    oneEqualityConstraint._constraints.put(coefNames[index], 1.0);
    oneEqualityConstraint._constraints.put("constant", -betaC._betaLB[index]);
    equalityC.add(oneEqualityConstraint);
  }

  /***
   * This method will extract the greater than constraint and add to lessThanC from beta constraint by doing the following
   * transformation: low_val <= beta <= Infinity: transformed to low_val - beta <= 0.
   */
  public static void addBCGreaterThanConstraint(List<LinearConstraints> lessThanC, GLM.BetaConstraint betaC,
                                             String[] coefNames, int index) {
    LinearConstraints lessThanEqualToConstraint = new LinearConstraints();
    lessThanEqualToConstraint._constraints.put(coefNames[index], -1.0);
    lessThanEqualToConstraint._constraints.put("constant", betaC._betaLB[index]);
    lessThanC.add(lessThanEqualToConstraint);
  }

  /***
   * This method will extract the less than constraint and add to lessThanC from beta constraint by doing the following
   * transformation: -Infinity <= beta <= high_val: transformed to beta - high_val <= 0.
   */
  public static void addBCLessThanConstraint(List<LinearConstraints> lessThanC, GLM.BetaConstraint betaC,
                                             String[] coefNames, int index) {
    LinearConstraints greaterThanConstraint = new LinearConstraints();
    greaterThanConstraint._constraints.put(coefNames[index], 1.0);
    greaterThanConstraint._constraints.put("constant", -betaC._betaUB[index]);
    lessThanC.add(greaterThanConstraint);  
  }

  /***
   * This method will extract the constraints specified in the Frame with key linearConstraintFrameKey.  For example,
   * the following constraints a*beta_1+b*beta_2-c*beta_5 == 0, d*beta_2+e*beta_6-f <= 0 can be specified as the
   * following rows:
   *  names           values            Type            constraint_numbers
   *  beta_1             a              Equal                   0
   *  beta_2             b              Equal                   0
   *  beta_5            -c              Equal                   0
   *  beta_2             d              LessThanEqual           1
   *  beta_6             e              LessThanEqual           1
   *  constant          -f              LessThanEqual           1
   */
  public static void extractLinearConstraints(ComputationState state, Key<Frame> linearConstraintFrameKey, DataInfo dinfo) {
    List<LinearConstraints> equalityC = new ArrayList<>();
    List<LinearConstraints> lessThanEqualToC = new ArrayList<>();
    Frame linearConstraintF = DKV.getGet(linearConstraintFrameKey);
    Scope.track(linearConstraintF);
    List<String> colNamesList = Stream.of(dinfo._adaptedFrame.names()).collect(Collectors.toList());
    List<String> coefNamesList = Stream.of(dinfo.coefNames()).collect(Collectors.toList());
    int numberOfConstraints = linearConstraintF.vec("constraint_numbers").toCategoricalVec().domain().length;
    int numRow = (int) linearConstraintF.numRows();
    List<Integer> rowIndices = IntStream.range(0,numRow).boxed().collect(Collectors.toList());
    String constraintType;
    int rowIndex;
    for (int conInd = 0; conInd < numberOfConstraints; conInd++) {
      if (!rowIndices.isEmpty()) {
        rowIndex = rowIndices.get(0);
        constraintType = linearConstraintF.vec("types").stringAt(rowIndex).toLowerCase();
        if ("equal".equals(constraintType)) {
          extractConstraint(linearConstraintF, rowIndices, equalityC, dinfo, coefNamesList, colNamesList);
        } else if ("lessthanequal".equals(constraintType)) {
          extractConstraint(linearConstraintF, rowIndices, lessThanEqualToC, dinfo, coefNamesList,
                  colNamesList);
        } else {
          throw new IllegalArgumentException("Type of linear constraints can only be Equal to LessThanEqualTo.");
        }
      }
    }
    state.setLinearConstraints(equalityC.toArray(new LinearConstraints[0]), 
            lessThanEqualToC.toArray(new LinearConstraints[0]), false);
  }
  
  public static void extractConstraint(Frame constraintF, List<Integer> rowIndices, List<LinearConstraints> equalC,
                                       DataInfo dinfo, List<String> coefNames, List<String> colNames) {
    List<Integer> processedRowIndices = new ArrayList<>();
    int constraintNumberFrame = (int) constraintF.vec("constraint_numbers").at(rowIndices.get(0));
    LinearConstraints currentConstraint = new LinearConstraints();
    String constraintType = constraintF.vec("types").stringAt(rowIndices.get(0)).toLowerCase();
    boolean standardize = dinfo._normMul != null;
    boolean constantFound = false;
    for (Integer rowIndex : rowIndices) {
      String coefName = constraintF.vec("names").stringAt(rowIndex);
      String currType = constraintF.vec("types").stringAt(rowIndex).toLowerCase();
      if (!coefNames.contains(coefName) && !"constant".equals(coefName))
        throw new IllegalArgumentException("Coefficient name " + coefName + " is not a valid coefficient name.  It " +
                "be a valid coefficient name or it can be constant");
      if ((int) constraintF.vec("constraint_numbers").at(rowIndex) == constraintNumberFrame) {
        if (!constraintType.equals(currType))
          throw new IllegalArgumentException("Constraint type "+" of the same constraint must be the same but is not." +
                  "  Expected type: "+constraintType+".  Actual type: "+currType);
        if ("constant".equals(coefName))
          constantFound = true;
        processedRowIndices.add(rowIndex);
        // coefNames is valid
        if (standardize && colNames.contains(coefName)) {  // numerical column with standardization
          int colInd = colNames.indexOf(coefName);
          currentConstraint._constraints.put(coefName, constraintF.vec("values").at(rowIndex)*dinfo._normMul[colInd-dinfo._cats]);
        } else {  // categorical column, constant or numerical column without standardization
          currentConstraint._constraints.put(coefName, constraintF.vec("values").at(rowIndex));
        }
      }
    }
    if (!constantFound)
      currentConstraint._constraints.put("constant", 0.0);  // put constant of 0.0
    if (currentConstraint._constraints.size() < 3)
      throw new IllegalArgumentException("linear constraint must have at least two coefficients.  For constraints on" +
              " just one coefficient: "+ constraintF.vec("names").stringAt(0)+", use betaConstraints instead.");
    equalC.add(currentConstraint);
    rowIndices.removeAll(processedRowIndices);
  }
  
  public static double[][] formConstraintMatrix(ComputationState state, List<String> constraintNamesList) {
    // extract coefficient names from constraints
    constraintNamesList.addAll(extractConstraintCoeffs(state));
    // form double matrix
    int numRow = (state._equalityConstraintsBeta == null ? 0 : state._equalityConstraintsBeta.length) +
            (state._lessThanEqualToConstraintsBeta == null ? 0 : state._lessThanEqualToConstraintsBeta.length) +
            (state._equalityConstraints == null ? 0 : state._equalityConstraints.length) + 
            (state._lessThanEqualToConstraints == null ? 0 : state._lessThanEqualToConstraints.length);
    double[][] initConstraintMatrix = new double[numRow][constraintNamesList.size()];
    fillConstraintValues(state, constraintNamesList, initConstraintMatrix);
    return initConstraintMatrix;
  }
  
  public static void fillConstraintValues(ComputationState state, List<String> constraintNamesList, double[][] initCMatrix) {
    int rowIndex = 0;
    if (state._equalityConstraintsBeta != null)
      rowIndex = extractConstraintValues(state._equalityConstraintsBeta, constraintNamesList, initCMatrix, rowIndex);
    if (state._lessThanEqualToConstraintsBeta != null)
      rowIndex= extractConstraintValues(state._lessThanEqualToConstraintsBeta, constraintNamesList, initCMatrix, rowIndex);
    if (state._equalityConstraints != null)
      rowIndex = extractConstraintValues(state._equalityConstraints, constraintNamesList, initCMatrix, rowIndex);
    if (state._lessThanEqualToConstraints != null)
      extractConstraintValues(state._lessThanEqualToConstraints, constraintNamesList, initCMatrix, rowIndex);
  }
  
  public static int extractConstraintValues(LinearConstraints[] constraints, List<String> constraintNamesList, double[][] initCMatrix, int rowIndex) {
    int numConstr = constraints.length;
    for (int index=0; index<numConstr; index++) {
      Set<String> coeffKeys = constraints[index]._constraints.keySet();
      for (String oneKey : coeffKeys) {
        if ( constraintNamesList.contains(oneKey))
          initCMatrix[rowIndex][constraintNamesList.indexOf(oneKey)] = constraints[index]._constraints.get(oneKey);
      }
      rowIndex++;
    }
    return rowIndex;
  }
  
  public static List<String> extractConstraintCoeffs(ComputationState state) {
    List<String> tConstraintCoeffName = new ArrayList<>();
    boolean nonZeroConstant = false;
    if (state._equalityConstraintsBeta != null)
      nonZeroConstant = nonZeroConstant | extractCoeffNames(tConstraintCoeffName, state._equalityConstraintsBeta);

    if (state._lessThanEqualToConstraintsBeta != null)
      nonZeroConstant = nonZeroConstant | extractCoeffNames(tConstraintCoeffName, state._lessThanEqualToConstraintsBeta);

    if (state._equalityConstraints != null)
      nonZeroConstant = nonZeroConstant | extractCoeffNames(tConstraintCoeffName, state._equalityConstraints);

    if (state._lessThanEqualToConstraints != null)
      nonZeroConstant = nonZeroConstant | extractCoeffNames(tConstraintCoeffName, state._lessThanEqualToConstraints);
    
    // remove duplicates in the constraints names
    Set<String> noDuplicateNames = new HashSet<>(tConstraintCoeffName);
    if (!nonZeroConstant) // no non-Zero constant present
      noDuplicateNames.remove("constant");
    return new ArrayList<>(noDuplicateNames);
  }
  
  public static boolean extractCoeffNames(List<String> coeffList, LinearConstraints[] constraints) {
    int numConst = constraints.length;
    boolean nonZeroConstant = false;
    for (int index=0; index<numConst; index++) {
      Set<String> keys = constraints[index]._constraints.keySet();
      coeffList.addAll(keys);
      if (keys.contains("constant"))
        nonZeroConstant = constraints[index]._constraints.get("constant") != 0.0;
    }
    return nonZeroConstant;
  }
  
  public static List<String> foundRedundantConstraints(ComputationState state, final double[][] initConstraintMatrix) {
    Matrix constMatrix = new Matrix(initConstraintMatrix);
    Matrix constMatrixLessConstant = constMatrix.getMatrix(0, constMatrix.getRowDimension() -1, 1, constMatrix.getColumnDimension()-1);
    Matrix constMatrixTConstMatrix = constMatrixLessConstant.times(constMatrixLessConstant.transpose());
    int rank = constMatrixLessConstant.rank();
    if (rank < constMatrix.getRowDimension()) { // redundant constraints are specified
      double[][] rMatVal = constMatrixTConstMatrix.qr().getR().getArray();
      List<Double> diag = IntStream.range(0, rMatVal.length).mapToDouble(x->Math.abs(rMatVal[x][x])).boxed().collect(Collectors.toList());
      int[] sortedIndices = IntStream.range(0, diag.size()).boxed().sorted((i, j) -> diag.get(i).compareTo(diag.get(j))).mapToInt(ele->ele).toArray();
      List<Integer> duplicatedEleIndice = IntStream.range(0, diag.size()-rank).map(x -> sortedIndices[x]).boxed().collect(Collectors.toList());
      return genRedundantConstraint(state, duplicatedEleIndice);
    }
    return null;
  }
  
  public static List<String> genRedundantConstraint(ComputationState state, List<Integer> duplicatedEleIndics) {
    List<String> redundantConstraint = new ArrayList<>();
    int numRedEle = duplicatedEleIndics.size();
    for (Integer redIndex : duplicatedEleIndics)
      redundantConstraint.add(grabRedundantConstraintMessage(state, redIndex));

    return redundantConstraint;
  }

  public static String grabRedundantConstraintMessage(ComputationState state, Integer constraintIndex) {
    // figure out which constraint among state._fromBetaConstraints, state._equalityConstraints, 
    // state._lessThanEqualToConstraints is actually redundant
    LinearConstraints redundantConst = getConstraintFromIndex(state, constraintIndex);
    if (redundantConst != null) {
      boolean isBetaConstraint = redundantConst._constraints.size() < 3 ? true : false;
      boolean standardize = state.activeData()._normMul != null ? true : false;
      StringBuilder sb = new StringBuilder();
      DataInfo dinfo = state.activeData();
      List<String> trainNames = Arrays.stream(dinfo._adaptedFrame.names()).collect(Collectors.toList());
      sb.append("This constraint is redundant ");
      
      if (isBetaConstraint && standardize) {  // beta constraint and standardization is needed
        double constantVal = 0;
        int colIndex = 0;
        for (String coefName : redundantConst._constraints.keySet()) {
          double constrVal = redundantConst._constraints.get(coefName);
          if ("constant".equals(coefName)) {
            constantVal = constrVal;
          } else {
            colIndex = trainNames.indexOf(coefName)-dinfo._cats;
            sb.append(coefName);  // coefficient value is always 1.
          }
        }
        sb.append(constantVal*dinfo._normMul[colIndex]);
      } else {  // beta constraint without standardization or linear constraint with or without standardization
        for (String coefName : redundantConst._constraints.keySet()) {
          double constrVal = redundantConst._constraints.get(coefName);
          if (constrVal != 0) {
            if (constrVal > 0)
              sb.append('+');

            // linearconstraint for numerical columns only that needs standardization
            if (standardize && !"constant".equals(coefName) && trainNames.contains(coefName)) {
              int colInd = trainNames.indexOf(coefName) - dinfo._cats;
              sb.append(constrVal/dinfo._normMul[colInd]);
            } else {  // no standardization or standardization but coefName is constant
              sb.append(constrVal);
            }
            if (!"constant".equals(coefName)) {
              sb.append('*');
              sb.append(coefName);
            }
          }
        }
      }
      sb.append(" <= or == 0.");
      sb.append(" Please remove it from your beta/linear constraints.");
      return sb.toString();
    } else {
      return null;
    }
  }
  
  public static LinearConstraints getConstraintFromIndex(ComputationState state, Integer constraintIndex) {
    int constIndexWOffset = constraintIndex;
    if (state._equalityConstraintsBeta != null) {
      if (constIndexWOffset < state._equalityConstraintsBeta.length) {
        return state._equalityConstraintsBeta[constIndexWOffset];
      } else {
        constIndexWOffset -= state._equalityConstraintsBeta.length;
      }
    }
    
    if (state._lessThanEqualToConstraintsBeta != null) {
      if (constIndexWOffset < state._lessThanEqualToConstraintsBeta.length) {
        return state._lessThanEqualToConstraintsBeta[constIndexWOffset];
      } else {
        constIndexWOffset -= state._lessThanEqualToConstraintsBeta.length;
      }
    }
    
    if (state._equalityConstraints != null) {
      if (constIndexWOffset < state._equalityConstraints.length) {
        return state._equalityConstraints[constIndexWOffset];
      } else {
        constIndexWOffset -= state._equalityConstraints.length;
      }
    }
    
    if (state._lessThanEqualToConstraints != null && constIndexWOffset < state._lessThanEqualToConstraints.length) {
      return state._lessThanEqualToConstraints[constIndexWOffset];
    }
    return null;
  }
  
  public static void setActiveConstraints(LinearConstraints[] lessThanEqualToConstraints,
                                          double[] beta, List<String> coeffNames) {
    // evaluate all constraint values
    Arrays.stream(lessThanEqualToConstraints).collect(Collectors.toList()).parallelStream().forEach(constraint -> evalOneConstraint(constraint, beta, coeffNames));
    // only include constraint with val >= 0 into activeConstraints
    Arrays.stream(lessThanEqualToConstraints).forEach(constraint -> {
      if (constraint._constraintsVal >= 0)
        constraint._active = true;
      else
        constraint._active = false;
    });
  }
  
  public static void evalOneConstraint(LinearConstraints constraint, double[] beta, List<String> coeffNames) {
    double sumV = 0.0;
    Map<String, Double> constraints = constraint._constraints;
    for (String coeff : constraints.keySet()) {
      if ("constant".equals(coeff))
        sumV += constraints.get(coeff);
      else
        sumV += constraints.get(coeff)*beta[coeffNames.indexOf(coeff)];
    }
    constraint._constraintsVal = sumV;
  }

  public static void genInitialLambda(Random randObj, LinearConstraints[] constraints, double[] lambda) {
    int numC = constraints.length;
    LinearConstraints oneC;
    for (int index=0; index<numC; index++) {
      oneC = constraints[index];
      lambda[index] = Math.abs(randObj.nextGaussian());
      if (oneC._active & oneC._constraintsVal < 0) {
          lambda[index] *= -1;  // switch sign of lambda_index such that constraint_index*lambda_index is positive
      }
    }
  }
  
  
  public static double[][] sumGramConstribution(ConstraintsGram[] gramEqual, int numCoefs) {
    if (gramEqual == null)
      return null;
    double[][] gramContr = new double[numCoefs][numCoefs];  // includes intercept terms
    int cGramSize = gramEqual.length;
    ConstraintsGram oneGram;
    int numMap;
    int coef1, coef2;
    for (int index=0; index < cGramSize; index++) {
      oneGram = gramEqual[index];
      if (oneGram._active) {  // only process the contribution if the constraint is active
        for (CoefIndices key : oneGram._coefIndicesValue.keySet()) {
          coef1 = key._firstCoefIndex;
          coef2 = key._secondCoefIndex;
          gramContr[coef1][coef2] += oneGram._coefIndicesValue.get(key);
          if (coef1 != coef2)
            gramContr[coef2][coef1] = gramContr[coef1][coef2];
        }
      }
    }
    return gramContr;
  }
  
  public static void addConstraintGradient(double[] lambda, ConstraintsDerivatives[] constraintD, 
                                           GLM.GLMGradientInfo gradientInfo) {
    int numConstraints = lambda.length;
    ConstraintsDerivatives oneC;
    for (int index=0; index<numConstraints; index++) {
      oneC = constraintD[index];
      if (oneC._active) {
        for (Integer key: oneC._constraintsDerivative.keySet()) {
          gradientInfo._gradient[key] += lambda[index]*oneC._constraintsDerivative.get(key);
        }
      }
    }
  }

  public static void addPenaltyGradient(ConstraintsDerivatives[] constraintDeriv, LinearConstraints[] constraintD, 
                                        GLM.GLMGradientInfo gradientInfo, double ck) {
    int numConstraints = constraintDeriv.length;
    ConstraintsDerivatives oneD;
    LinearConstraints oneConts;
    int[] coeffKeys;
    for (int index=0; index<numConstraints; index++) {
      oneD = constraintDeriv[index];
      if (oneD._active) {
        oneConts = constraintD[index];
        for (Integer coefK : oneD._constraintsDerivative.keySet()) {
          gradientInfo._gradient[coefK] += ck*oneConts._constraintsVal*oneD._constraintsDerivative.get(coefK);
        }
      }
    }
  }
  
  public static void updateConstraintParameters(ComputationState state, double[] lambdaEqual, double[]lambdaLessThan, 
                                                LinearConstraints[] equalConst, LinearConstraints[] lessThanConst) {
    // calculate ||h(beta)|| and check what updates to perform
    double hBetaMag = state._csGLMState._constraintMagSquare;
    if (hBetaMag <= state._csGLMState._etakCS*state._csGLMState._etakCS) {
      if (equalConst != null)
        updateLambda(lambdaEqual, state._csGLMState._ckCS, equalConst);
      updateLambda(lambdaLessThan, state._csGLMState._ckCS, lessThanConst);
      state._csGLMState._epsilonkCS = state._csGLMState._epsilonkCS/state._csGLMState._ckCS;
      state ._csGLMState._etakCS = state._csGLMState._etakCS/Math.pow(state._csGLMState._ckCS, BETACS);
    } else {
      state._csGLMState._ckCS = state._csGLMState._ckCS*C0CS;
      state._csGLMState._epsilonkCS = EPSILON0/state._csGLMState._ckCS;
      state ._csGLMState._etakCS = ETA0CS/Math.pow(state._csGLMState._ckCS, ALPHACS);
    }
  }
  
  public static void updateLambda(double[] lambda, double ckCS, LinearConstraints[] constraints) {
    int numC = constraints.length;
    LinearConstraints oneC;
    for (int index=0; index<numC; index++) {
      oneC = constraints[index];
      if (oneC._active)
        lambda[index] += ckCS*oneC._constraintsVal;
    }
  }

  /***
   * This method will calculate the ||h(beta)|| square.  It will also update the constraint values as well and the 
   * active statues for less than and equal to zero constraints are re-evaluated as well with the beta values.
   * 
   */
  public static double calHBetaMagSquare(double[] beta, LinearConstraints[] constraints, List<String> coefNames, 
                                   boolean setActive2True) {
    if (constraints == null) return 0;
    double sumV = 0;
    int numConst = constraints.length;
    LinearConstraints oneConst;
    double val;
    for (int index=0; index<numConst; index++) {
      oneConst = constraints[index];
      evalOneConstraint(oneConst, beta, coefNames);
      val = oneConst._constraintsVal;
      if (setActive2True) {  // processing equal constraints
        oneConst._active = true;
        sumV += val*val;
      } else {  // processing less than equal constraints
        if (val >= 0) {
          oneConst._active = true;
          sumV += val*val;
        } else {   // constraint ignored since it is < 0
          oneConst._active = false;
        }
      }
    }
    return sumV;
  }

  /***
   * This method will check if the stopping 
   * conditions for constraint GLM are met and they are namely:
   * 1. ||gradient of L with respect to beta and withy respect to lambda|| <= epsilon
   * 2. ||h(beta)|| square <= epsilon
   * 
   * If the stopping conditions are met, it will return true, else it will return false.
   */
  public static boolean constraintsStop(GLM.GLMGradientInfo gradientInfo, ComputationState state, 
                                        LinearConstraints[] equalConst, LinearConstraints[] lessThanConst, 
                                        List<String> coefNames) {
    double[] beta = state.beta();
    state._csGLMState._constraintMagSquare = calHBetaMagSquare(beta, lessThanConst, coefNames, false) + 
            calHBetaMagSquare(beta, equalConst, coefNames, true);
    state._csGLMState._gradientMag = Math.sqrt(innerProduct(gradientInfo._gradient, gradientInfo._gradient));
    if (state._csGLMState._constraintMagSquare <= ComputationState.EPS_CS && state._csGLMState._gradientMag <= ComputationState.EPS_CS)
      return true;
    return false;
  }
  
  public static GLM.GLMGradientInfo calGradient(double[] betaCnd, ComputationState state, GLM.GLMGradientSolver ginfo,
                                                double[] lambdaE, double[] lambdaL, LinearConstraints[] constraintE,
                                                LinearConstraints[] constrainL, ConstraintsDerivatives[] equalD, 
                                                ConstraintsDerivatives[] lessD) {
    // todo: need to add support for predictors removed for whatever reason
    // calculate gradients
    GLM.GLMGradientInfo gradientInfo = ginfo.getGradient(betaCnd); // gradient without constraints
    // add gradient contribution from constraints
    if (equalD != null)
      addConstraintGradient(lambdaE, equalD, gradientInfo);
    addConstraintGradient(lambdaL, lessD, gradientInfo);
    if (equalD != null)
      addPenaltyGradient(equalD, constraintE, gradientInfo, state._csGLMState._ckCS);
    addPenaltyGradient(lessD, constrainL, gradientInfo, state._csGLMState._ckCS);
    return gradientInfo;
  }
  
  public static void updateConstraintValues(double[] betaCnd, List<String> coeffNames, 
                                            LinearConstraints[] equalityConstraints, LinearConstraints[] lessThanEqualToConstraints) {
    if (equalityConstraints != null)  // initialize lambda for equality constraints
      Arrays.stream(equalityConstraints).collect(Collectors.toList()).parallelStream().forEach(constraint -> evalOneConstraint(constraint, betaCnd, coeffNames));
    setActiveConstraints(lessThanEqualToConstraints, betaCnd, coeffNames);
  }
  
  public static String[] collinearInConstraints(String[] collinear_cols, String[] constraintNames) {
    List<String> cNames = Arrays.stream(constraintNames).collect(Collectors.toList());
    return Arrays.stream(collinear_cols).filter(x -> (cNames.contains(x))).toArray(String[]::new);
  }
}
