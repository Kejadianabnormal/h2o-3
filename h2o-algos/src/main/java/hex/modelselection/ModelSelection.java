package hex.modelselection;

import hex.*;
import hex.glm.GLM;
import hex.glm.GLMModel;
import water.DKV;
import water.H2O;
import water.HeartBeat;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.util.PrettyPrint;

import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static hex.gam.MatrixFrameUtils.GamUtils.copy2DArray;
import static hex.genmodel.utils.MathUtils.combinatorial;
import static hex.glm.GLMModel.GLMParameters.Family.*;
import static hex.modelselection.ModelSelectionModel.ModelSelectionParameters.Mode.*;
import static hex.modelselection.ModelSelectionUtils.*;

public class ModelSelection extends ModelBuilder<hex.modelselection.ModelSelectionModel, hex.modelselection.ModelSelectionModel.ModelSelectionParameters, hex.modelselection.ModelSelectionModel.ModelSelectionModelOutput> {
    public String[][] _bestModelPredictors; // store for each predictor number, the best model predictors
    public double[] _bestR2Values;  // store the best R2 values of the best models with fix number of predictors
    public String[][] _predictorsAdd;
    public String[][] _predictorsRemoved;
    DataInfo _dinfo;
    String[] _coefNames;
    int[][] _predictorIndex2CPMIndices;    // map predictor indices to the corresponding Gram matrix indices
    double[][] _crossProdcutMatrix;
    public int _numPredictors;
    public String[] _predictorNames;
    public int _glmNFolds = 0;
    Model.Parameters.FoldAssignmentScheme _foldAssignment = null;
    String _foldColumn = null;

    public ModelSelection(boolean startup_once) {
        super(new hex.modelselection.ModelSelectionModel.ModelSelectionParameters(), startup_once);
    }
    
    public ModelSelection(hex.modelselection.ModelSelectionModel.ModelSelectionParameters parms) {
        super(parms);
        init(false);
    }
    
    public ModelSelection(hex.modelselection.ModelSelectionModel.ModelSelectionParameters parms, Key<hex.modelselection.ModelSelectionModel> key) {
        super(parms, key);
        init(false);
    }
    
    @Override
    protected int nModelsInParallel(int folds) {
        return nModelsInParallel(1,2);  // disallow nfold cross-validation
    }

    @Override
    protected ModelSelectionDriver trainModelImpl() {
        return new ModelSelectionDriver();
    }

    @Override
    public ModelCategory[] can_build() {
        return new ModelCategory[]{ModelCategory.Regression};
    } // because of r2 usage

    @Override
    public boolean isSupervised() {
        return true;
    }

    @Override
    public boolean haveMojo() {
        return false;
    }

    @Override
    public boolean havePojo() {
        return false;
    }
    
    public void init(boolean expensive) {
        if (_parms._nfolds > 0 || _parms._fold_column != null) {    // cv enabled
            if (backward.equals(_parms._mode)) {
                error("nfolds/fold_column", "cross-validation is not supported for backward " +
                        "selection.");
            } else if (maxrsweepsmall.equals(_parms._mode) || maxrsweepfull.equals(_parms._mode) || 
                    maxrsweephybrid.equals(_parms._mode) || maxrsweep.equals(_parms._mode)) {
                error("nfolds/fold_column", "cross-validation is not supported for maxrsweep, " +
                        " maxrsweepsmall, maxrsweep and maxrsweepfull.");
            } else {
                _glmNFolds = _parms._nfolds;
                if (_parms._fold_assignment != null) {
                    _foldAssignment = _parms._fold_assignment;
                    _parms._fold_assignment = null;
                }
                if (_parms._fold_column != null) {
                    _foldColumn = _parms._fold_column;
                    _parms._fold_column = null;
                }
                _parms._nfolds = 0;
            }
        }
        super.init(expensive);
        if (error_count() > 0)
            return;
        if (expensive) {
            initModelSelectionParameters();
            if (error_count() > 0)
                return;
            initModelParameters();
        }
    }
    
    private void initModelParameters() {
        if (!backward.equals(_parms._mode)) {
            _bestR2Values = new double[_parms._max_predictor_number];
            _bestModelPredictors = new String[_parms._max_predictor_number][];
            if (!backward.equals(_parms._mode))
                _predictorsAdd = new String[_parms._max_predictor_number][];
            _predictorsRemoved = new String[_parms._max_predictor_number][];
        }
    }

    private void initModelSelectionParameters() {
        _predictorNames = extractPredictorNames(_parms, _dinfo, _foldColumn);
        _numPredictors = _predictorNames.length;

        if (maxr.equals(_parms._mode) || allsubsets.equals(_parms._mode) || maxrsweepsmall.equals(_parms._mode) 
                || maxrsweepfull.equals(_parms._mode) || maxrsweephybrid.equals(_parms._mode) || maxrsweep.equals(_parms._mode)) { // check for maxr and allsubsets
            if (_parms._lambda == null && !_parms._lambda_search && _parms._alpha == null && 
                    !maxrsweepsmall.equals(_parms._mode) && !maxrsweepfull.equals(_parms._mode) && 
                    !maxrsweephybrid.equals(_parms._mode) && !maxrsweep.equals(_parms._mode))
                _parms._lambda = new double[]{0.0}; // disable regularization if not specified
            if (nclasses() > 1)
                error("response", "'allsubsets', 'maxr', 'maxrsweep', maxrsweepfull', " +
                        "'maxrsweepsmall', 'maxrsweep' only works with regression.");
            
            if (!(AUTO.equals(_parms._family) || gaussian.equals(_parms._family)))
                error("_family", "ModelSelection only supports Gaussian family for 'allsubset' and 'maxr' mode.");
            
            if (AUTO.equals(_parms._family))
                _parms._family = gaussian;

            if (_parms._max_predictor_number < 1 || _parms._max_predictor_number > _numPredictors)
                error("max_predictor_number", "max_predictor_number must exceed 0 and be no greater" +
                        " than the number of predictors of the training frame.");
        } else {    // checks for backward selection only
            _parms._compute_p_values = true;
            if (_parms._valid != null)
                error("validation_frame", " is not supported for ModelSelection mode='backward'");
            if (_parms._lambda_search)
                error("lambda_search", "backward selection does not support lambda_search.");
            if (_parms._lambda != null) {
                if (_parms._lambda.length > 1)
                    error("lambda", "if set must be set to 0 and cannot be an array or more than" +
                            " length one for backward selection.");
                if (_parms._lambda[0] != 0)
                    error("lambda", "must be set to 0 for backward selection");
            } else {
                _parms._lambda = new double[]{0.0};
            }
            if (multinomial.equals(_parms._family) || ordinal.equals(_parms._family))
                error("family", "backward selection does not support multinomial or ordinal");
            if (_parms._min_predictor_number <= 0) 
                error("min_predictor_number", "must be >= 1.");
            if (_parms._min_predictor_number > _numPredictors)
                error("min_predictor_number", "cannot exceed the total number of predictors (" + 
                        _numPredictors + ")in the dataset.");
        }
        
        if (_parms._nparallelism < 0) 
            error("nparallelism", "must be >= 0.");
        
        if (_parms._nparallelism == 0)
            _parms._nparallelism = H2O.NUMCPUS;
        
        if (maxrsweepsmall.equals(_parms._mode) || maxrsweepfull.equals(_parms._mode) ||
                maxrsweephybrid.equals(_parms._mode) || maxrsweep.equals(_parms._mode))
            warn("validation_frame", " is not used in choosing the best k subset for ModelSelection" +
                    " modelss with maxrsweep, maxrsweepsmall, maxrsweepfull.");
    }

    protected void checkMemoryFootPrint(int p) {
        if (maxrsweepsmall.equals(_parms._mode))
            p = (int) Math.ceil(p*(_parms._max_predictor_number+2)/_dinfo.fullN());
        HeartBeat hb = H2O.SELF._heartbeat;
        long mem_usage = (long) (hb._cpus_allowed * (p * p * p));
        long max_mem = hb.get_free_mem();

        if (mem_usage > max_mem) {
            String msg = "Gram matrices (one per thread) won't fit in the driver node's memory ("
                    + PrettyPrint.bytes(mem_usage) + " > " + PrettyPrint.bytes(max_mem)
                    + ") - try reducing the number of columns and/or the number of categorical factors (or switch to the L-BFGS solver).";
            error("_train", msg);
        }
    }
    
    public class ModelSelectionDriver extends Driver {
        public final void buildModel() {
            hex.modelselection.ModelSelectionModel model = null;
            try {
                int numModelBuilt = 0;
                model = new hex.modelselection.ModelSelectionModel(dest(), _parms, new hex.modelselection.ModelSelectionModel.ModelSelectionModelOutput(ModelSelection.this, _dinfo));
                model.write_lock(_job);
                model._output._mode = _parms._mode;
                if (backward.equals(_parms._mode)) {
                    model._output._best_model_ids = new Key[_numPredictors];
                    model._output._coef_p_values = new double[_numPredictors][];
                    model._output._z_values = new double[_numPredictors][];
                    model._output._best_model_coef_names = new String[_numPredictors][];
                    model._output._best_predictors_subset = new String[_numPredictors][];
                    model._output._coefficient_names = new String[_numPredictors][];
                } else {    // maxr, maxrsweep, marxsweepfull, maxrsweepsmall
                    model._output._best_r2_values = new double[_parms._max_predictor_number];
                    model._output._best_model_coef_names = new String[_parms._max_predictor_number][];
                    model._output._best_predictors_subset = new String[_numPredictors][];
                    model._output._coefficient_names = new String[_parms._max_predictor_number][];
                    if ((maxrsweep.equals(_parms._mode) || maxrsweepsmall.equals(_parms._mode) || 
                            maxrsweepfull.equals(_parms._mode)) && !_parms._build_glm_model) {
                        model._output._best_model_coef_values = new double[_parms._max_predictor_number][];
                        model._output._best_model_ids = null;
                    } else {
                        model._output._best_model_ids = new Key[_parms._max_predictor_number];
                    }
                }
                model._output._predictors_removed_per_step = new String[_numPredictors][];
                model._output._predictors_added_per_step = new String[_numPredictors][];
                // build glm model with num_predictors and find one with best R2
                if (allsubsets.equals(_parms._mode))
                    buildAllSubsetsModels(model);
                else if (maxr.equals(_parms._mode))
                    buildMaxRModels(model);
                else if (backward.equals(_parms._mode))
                    numModelBuilt = buildBackwardModels(model);
                else if (maxrsweepfull.equals(_parms._mode))
                    buildMaxRSweepFullModels(model, 1, _parms._max_predictor_number, null);
                else if (maxrsweepsmall.equals(_parms._mode))
                    buildMaxRSweepSmallModels(model, _parms._max_predictor_number);
                else if (maxrsweephybrid.equals(_parms._mode))
                    buildMaxRSweepHybridModels(model);
                else if (maxrsweep.equals(_parms._mode)) 
                    buildMaxRSweepModels(model);
                
                _job.update(0, "Completed GLM model building.  Extracting results now.");
                model.update(_job);
                // copy best R2 and best predictors to model._output
                if (backward.equals(_parms._mode)) {
                    model._output.shrinkArrays(numModelBuilt);
                    model._output.generateSummary(numModelBuilt);
                } else {
                    model._output.generateSummary();
                }
            } finally {
                model.update(_job);
                model.unlock(_job);
            }
        }

        /***
         * The maxrsweep mode will do the following:
         * 1. if the _max_predictor_number is small, maxrSweepSmall will be called to build the models;
         * 2. if the _max_predictor_number > _parms._max_predict_subset, maxrSweepSmall will be called to build the 
         * models with small predictor subsets for higher speed than using maxrsweepfull. For predictor subset 
         * size > _parms._max_predict_subset, the rest of the bigger predictor subsets models will be built by
         * maxrsweepfull as it runs faster with large predictor subsets.
         */
        void buildMaxRSweepHybridModels(ModelSelectionModel model) {
            if (_parms._max_predictor_number > _parms._max_predictor_subset) {
                List<Integer> currSubsetIndices = buildMaxRSweepSmallModels(model, _parms._max_predictor_subset);
                buildMaxRSweepFullModels(model, _parms._max_predictor_subset+1, 
                        _parms._max_predictor_number, currSubsetIndices);
            } else {
                buildMaxRSweepSmallModels(model, _parms._max_predictor_number);
            }
        }
        /***
         * 
         * The maxrsweep mode implementation can be explained in the maxrsweep.pdf doc stored here:
         * https://h2oai.atlassian.net/browse/PUBDEV-8703 .  Apart from actions specific to sweep implementation, the
         * logic in this function is very similar to that of maxr mode.
         * 
         */
        void buildMaxRSweepFullModels(ModelSelectionModel model, int modelIndexStart, int numModels2Build, List<Integer> currSubsetIndices) {
            _coefNames = _dinfo.coefNames();
            List<String> coefNames = new ArrayList<>(Arrays.asList(_predictorNames));
            // store predictor indices that are still available to be added to the bigger subset
            List<Integer> validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
            // generate mapping of predictor index to CPM indices due to enum columns add multiple rows/columns to CPM
            _predictorIndex2CPMIndices = mapPredIndex2CPMIndices(_dinfo, _predictorNames.length);
            // generate cross-product matrix (CPM) as in section III of doc
            double[][] crossProdcutMatrix = createCrossProductMatrix(_job._key, _dinfo);
            checkMemoryFootPrint(crossProdcutMatrix.length);
            if (_parms._intercept)  // sweep the intercept now
                sweepCPM(crossProdcutMatrix, new int[]{crossProdcutMatrix.length-2}, false);
            if (modelIndexStart > 1) {    // perform model sweep before continue
                for (int x : currSubsetIndices)
                    sweepCPM(crossProdcutMatrix, _predictorIndex2CPMIndices[currSubsetIndices.get(x)], false);
            } else {
                currSubsetIndices = new ArrayList<>();    // store best k predictor subsets for 1 to k predictors
            }
            
            for (int predNum = modelIndexStart; predNum <= numModels2Build; predNum++) { // find best predictor subset for each subset size
                Set<BitSet> usedCombos = new HashSet<>();
                currSubsetIndices = forwardStep(currSubsetIndices, coefNames, validSubset, usedCombos, crossProdcutMatrix,
                        _predictorIndex2CPMIndices, currSubsetIndices.size());
                sweepCPM(crossProdcutMatrix, _predictorIndex2CPMIndices[currSubsetIndices.get(currSubsetIndices.size()-1)],
                        false);
                validSubset.removeAll(currSubsetIndices);
                _job.update(predNum, "Finished forward step with "+predNum+" predictors.");

                if (predNum < _numPredictors && predNum > 1) {  // implement the replacement part
                    currSubsetIndices = replacement(currSubsetIndices, coefNames, validSubset, usedCombos,
                            crossProdcutMatrix, _predictorIndex2CPMIndices);
                    validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                    validSubset.removeAll(currSubsetIndices);
                }
                // build glm model with best subcarrier subsets for size and record the update
                if (_parms._build_glm_model) {
                    GLMModel bestR2Model = buildGLMModel(currSubsetIndices);
                    DKV.put(bestR2Model);
                    model._output.updateBestModels(bestR2Model, predNum - 1);
                } else {
                    model._output.updateBestModels(_predictorNames, _coefNames, predNum-1, _parms._intercept,
                            crossProdcutMatrix.length, currSubsetIndices.stream().mapToInt(x->x).toArray(),
                            crossProdcutMatrix);
                }
            }
        }

        /***
         *
         * The maxrsweep mode implementation can be explained in the maxrsweep.pdf doc stored here:
         * https://h2oai.atlassian.net/browse/PUBDEV-8703 .  Apart from actions specific to sweep implementation, the
         * logic in this function is very similar to that of maxr mode.
         *
         */
        List<Integer> buildMaxRSweepSmallModels(ModelSelectionModel model, int numModels2Build) {
            _coefNames = _dinfo.coefNames();
            // generate cross-product matrix (CPM) as in section III of doc
            _crossProdcutMatrix = createCrossProductMatrix(_job._key, _dinfo);
            checkMemoryFootPrint(_crossProdcutMatrix.length);
            // generate mapping of predictor index to CPM indices due to enum columns add multiple rows/columns to CPM
            _predictorIndex2CPMIndices = mapPredIndex2CPMIndices(_dinfo, _predictorNames.length);
            List<Integer> currSubsetIndices = new ArrayList<>();    // store best k predictor subsets for 1 to k predictors
            List<String> coefNames = new ArrayList<>(Arrays.asList(_predictorNames));
            // store predictor indices that are still available to be added to the bigger subset
            List<Integer> validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
            SweepModel bestModel = null;

            for (int predNum = 1; predNum <= numModels2Build; predNum++) { // find best predictor subset for each subset size
                Set<BitSet> usedCombos = new HashSet<>();
                if (bestModel == null) {
                    bestModel = forwardStep(currSubsetIndices, coefNames, validSubset,
                            usedCombos, _crossProdcutMatrix, _predictorIndex2CPMIndices, null, _parms._intercept);
                } else {
                    genBestSweepVector(bestModel, _crossProdcutMatrix, _predictorIndex2CPMIndices, _parms._intercept);
                    bestModel = forwardStep(currSubsetIndices, coefNames, validSubset,
                            usedCombos, _crossProdcutMatrix, _predictorIndex2CPMIndices, bestModel, _parms._intercept);
                }
                validSubset.removeAll(currSubsetIndices);
                _job.update(predNum, "Finished forward step with "+predNum+" predictors.");

                if (predNum < _numPredictors && predNum > 1) {  // implement the replacement part
                    bestModel = replacement(currSubsetIndices, coefNames, validSubset, usedCombos, bestModel);
                    currSubsetIndices = IntStream.of(bestModel._predSubset).boxed().collect(Collectors.toList());
                    validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                    validSubset.removeAll(currSubsetIndices);
                }
                // build glm model with best subcarrier subsets for size and record the update
                if (_parms._build_glm_model) {
                    GLMModel bestR2Model = buildGLMModel(currSubsetIndices);
                    DKV.put(bestR2Model);
                    model._output.updateBestModels(bestR2Model, predNum - 1);
                } else {
                    model._output.updateBestModels(_predictorNames, _coefNames, predNum-1, _parms._intercept, 
                            bestModel._CPM.length, bestModel._predSubset, bestModel._CPM);
                }
            }
            return currSubsetIndices;
        }

        List<Integer> buildMaxRSweepModels(ModelSelectionModel model) {
            _coefNames = _dinfo.coefNames();
            // generate cross-product matrix (CPM) as in section III of doc
            _crossProdcutMatrix = createCrossProductMatrix(_job._key, _dinfo);
            checkMemoryFootPrint(_crossProdcutMatrix.length);
            // generate mapping of predictor index to CPM indices due to enum columns add multiple rows/columns to CPM
            _predictorIndex2CPMIndices = mapPredIndex2CPMIndices(_dinfo, _predictorNames.length);
            List<Integer> currSubsetIndices = new ArrayList<>();    // store best k predictor subsets for 1 to k predictors
            List<String> coefNames = new ArrayList<>(Arrays.asList(_predictorNames));
            // store predictor indices that are still available to be added to the bigger subset
            List<Integer> validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
            SweepModel2 bestModel = null;
            int maxPredNum = _parms._max_predictor_number;
            for (int predNum = 1; predNum <= maxPredNum; predNum++) { // find bes_predictorNamest predictor subset for each subset size
                Set<BitSet> usedCombos = new HashSet<>();
                bestModel = forwardStepO(currSubsetIndices, coefNames, validSubset, usedCombos, _crossProdcutMatrix, 
                        _predictorIndex2CPMIndices, bestModel, _parms._intercept, false, maxPredNum);
                validSubset.removeAll(currSubsetIndices);
                _job.update(predNum, "Finished forward step with "+predNum+" predictors.");

                if (predNum < _numPredictors && predNum > 1) {  // implement the replacement part
                    bestModel = replacementO(currSubsetIndices, coefNames, validSubset, usedCombos, bestModel);
                    currSubsetIndices = IntStream.of(bestModel._predSubset).boxed().collect(Collectors.toList());
                    validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                    validSubset.removeAll(currSubsetIndices);
                }
                
                if (_parms._build_glm_model) {
                    // build glm model with best subcarrier subsets for size and record the update
                    GLMModel bestR2Model = buildGLMModel(currSubsetIndices);
                    DKV.put(bestR2Model);
                    model._output.updateBestModels(bestR2Model, predNum - 1);
                } else {
                    SweepInfo lastInfo = bestModel._sweepInfo.get(bestModel._sweepInfo.size()-1);
                    double[][] lastCPM = lastInfo._cpm[lastInfo._cpm.length-1];
                    model._output.updateBestModels(_predictorNames, _coefNames,  predNum-1, _parms._intercept, bestModel._cpmSize,
                            bestModel._predSubset, lastCPM);
                }
            }
            return currSubsetIndices;
        }
        
        public GLMModel buildGLMModel(List<Integer> bestSubsetIndices) {
            // generate training frame
            int[] subsetIndices = bestSubsetIndices.stream().mapToInt(Integer::intValue).toArray();
            Frame trainFrame = generateOneFrame(subsetIndices, _parms, _predictorNames, null);
            DKV.put(trainFrame);
            // generate training parameters
            final Field[] field1 = ModelSelectionModel.ModelSelectionParameters.class.getDeclaredFields();
            final Field[] field2 = Model.Parameters.class.getDeclaredFields();
            GLMModel.GLMParameters params = new GLMModel.GLMParameters();
            setParamField(_parms, params, false, field1, Collections.emptyList());
            setParamField(_parms, params, true, field2, Collections.emptyList());
            params._train = trainFrame._key;
            if (_parms._valid != null)
                params._valid = _parms._valid;
                
            // build and return model
            GLMModel model = new GLM(params).trainModel().get();
            DKV.remove(trainFrame._key);
            return model;
        }

        /**
         * Perform variable selection using MaxR implementing the sequential replacement method:
         * 1. forward step: find the initial subset of predictors with highest R2 values.  When subset size = 1, this
         *    means basically choosing the one predictor that generates a model with highest R2.  When subset size > 1,
         *    this means we choose one predictor to add to the subset which generates a model with hightest R2.
         * 2. Replacement step: consider the predictors in subset as pred0, pred1, pred2 (using subset size 3 as example):
         *    a. Keep pred1, pred2 and replace pred0 with remaining predictors, find replacement with highest R2
         *    b. keep pred0, pred2 and replace pred1 with remaining predictors, find replacement with highest R2
         *    c. keeep pred0, pred1 and replace pred2 with remaining predictors, find replacement with highest R2
         *    d. from step 2a, 2b, 2c, choose the predictor subset with highest R2, say the subset is pred0, pred4, pred2
         *    e. Take subset pred0, pred4, pred2 and go through steps a,b,c,d again and only stop when the best R2 does
         *       not improve anymore.
         *       
         * see doc at https://h2oai.atlassian.net/browse/PUBDEV-8444 for details.
         * 
         * @param model
         */
        void buildMaxRModels(ModelSelectionModel model) {
            List<Integer> currSubsetIndices = new ArrayList<>();
            List<String> coefNames = new ArrayList<>(Arrays.asList(_predictorNames));
            List<Integer> validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
            for (int predNum = 1; predNum <= _parms._max_predictor_number; predNum++) { // perform for each subset size
                Set<BitSet> usedCombos = new HashSet<>();
                GLMModel bestR2Model = forwardStep(currSubsetIndices, coefNames, predNum - 1, validSubset,
                        _parms, _foldColumn, _glmNFolds, _foldAssignment, usedCombos); // forward step
                validSubset.removeAll(currSubsetIndices);
                _job.update(predNum, "Finished building all models with "+predNum+" predictors.");
                if (predNum < _numPredictors && predNum > 1) {
                    double bestR2ofModel = 0;
                    if (bestR2Model != null)
                        bestR2ofModel = bestR2Model.r2();
                    GLMModel currBestR2Model = replacement(currSubsetIndices, coefNames, bestR2ofModel, _parms,
                            _glmNFolds, _foldColumn, validSubset, _foldAssignment, usedCombos);
                    if (currBestR2Model != null) {
                        bestR2Model.delete();
                        bestR2Model = currBestR2Model;
                    }
                }
                DKV.put(bestR2Model);
                model._output.updateBestModels(bestR2Model, predNum - 1);
            }
        }
        
        /**
         * Implements the backward selection mode.  Refer to III of ModelSelectionTutorial.pdf in 
         * https://h2oai.atlassian.net/browse/PUBDEV-8428
         */
        private int buildBackwardModels(ModelSelectionModel model) {
            List<String> coefNames = new ArrayList<>(Arrays.asList(_predictorNames));
            List<Integer> coefIndice = IntStream.rangeClosed(0, coefNames.size()-1).boxed().collect(Collectors.toList());
            Frame train = DKV.getGet(_parms._train);
            List<String> numPredNames = coefNames.stream().filter(x -> train.vec(x).isNumeric()).collect(Collectors.toList());
            List<String> catPredNames = coefNames.stream().filter(x -> !numPredNames.contains(x)).collect(Collectors.toList());
            int numModelsBuilt = 0;
            String[] coefName = coefNames.toArray(new String[0]);
            for (int predNum = _numPredictors; predNum >= _parms._min_predictor_number; predNum--) {
                int modelIndex = predNum-1;
                int[] coefInd = coefIndice.stream().mapToInt(Integer::intValue).toArray();
                Frame trainingFrame = generateOneFrame(coefInd, _parms, coefName, _foldColumn);
                DKV.put(trainingFrame);
                GLMModel.GLMParameters[] glmParam = generateGLMParameters(new Frame[]{trainingFrame}, _parms, 
                        _glmNFolds, _foldColumn, _foldAssignment);
                GLMModel glmModel = new GLM(glmParam[0]).trainModel().get();
                DKV.put(glmModel);  // track model

                // evaluate which variable to drop for next round of testing and store corresponding values
                // if p_values_threshold is specified, model building may stop
                model._output.extractPredictors4NextModel(glmModel, modelIndex, coefNames, coefIndice, numPredNames, 
                        catPredNames);
                numModelsBuilt++;
                DKV.remove(trainingFrame._key);
                _job.update(predNum, "Finished building all models with "+predNum+" predictors.");
                if (_parms._p_values_threshold > 0) {   // check if p-values are used to stop model building
                    if (DoubleStream.of(model._output._coef_p_values[modelIndex])
                            .limit(model._output._coef_p_values[modelIndex].length-1)
                            .allMatch(x -> x <= _parms._p_values_threshold))
                        break;
                }
            }
            return numModelsBuilt;
        }
        
        /***
         * Find the subset of predictors of sizes 1, 2, ..., _parm._max_predictor_number that generate the
         * highest R2 value using brute-force.  Basically for each subset size, all combinations of predictors
         * are considered.  This method is guaranteed to find the best predictor subset at the cost of computation
         * complexity.
         * 
         * @param model
         */
        void buildAllSubsetsModels(ModelSelectionModel model) {
            for (int predNum=1; predNum <= _parms._max_predictor_number; predNum++) {
                int numModels = combinatorial(_numPredictors, predNum);
                // generate the training frames with num_predictor predictors in the frame
                Frame[] trainingFrames = generateTrainingFrames(_parms, predNum, _predictorNames, numModels,
                        _foldColumn);
                // find best GLM Model with highest R2 value
                GLMModel bestModel = buildExtractBestR2Model(trainingFrames,_parms, _glmNFolds, _foldColumn, _foldAssignment);
                DKV.put(bestModel);
                // extract R2 and collect the best R2 and the predictors set
                int index = predNum-1;
                model._output.updateBestModels(bestModel, index);
                // remove training frames from DKV
                removeTrainingFrames(trainingFrames);
                _job.update(predNum, "Finished building all models with "+predNum+" predictors.");
            }
        }
        
        @Override
        public void computeImpl() {
            if (_parms._lambda_search || !_parms._intercept || _parms._lambda == null || _parms._lambda[0] > 0)
                _parms._use_all_factor_levels = true;
            _dinfo = new DataInfo(_train.clone(), _valid, 1, _parms._use_all_factor_levels,
                    DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                    _parms.missingValuesHandling() == GLMModel.GLMParameters.MissingValuesHandling.Skip,
                    _parms.imputeMissing(), _parms.makeImputer(), false, hasWeightCol(), hasOffsetCol(),
                    hasFoldCol(), null);
            init(true);
            if (error_count() > 0)
                throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(ModelSelection.this);
            _job.update(0, "finished init and ready to build models");
            buildModel();
        }
    }
    
    /**
     * Given the training Frame array, build models for each training frame and return the GLMModel with the best
     * R2 values.
     *
     * @param trainingFrames
     * @return
     */
    public static GLMModel buildExtractBestR2Model(Frame[] trainingFrames,
                                                   ModelSelectionModel.ModelSelectionParameters parms, int glmNFolds,
                                                   String foldColumn, Model.Parameters.FoldAssignmentScheme foldAssignment) {
        GLMModel.GLMParameters[] trainingParams = generateGLMParameters(trainingFrames, parms, glmNFolds,
                foldColumn, foldAssignment);
        // generate the builder;
        GLM[] glmBuilder = buildGLMBuilders(trainingParams);
        // call parallel build
        GLM[] glmResults = ModelBuilderHelper.trainModelsParallel(glmBuilder, parms._nparallelism);
        // find best GLM Model with highest R2 value
        return findBestModel(glmResults);
    }

    /**
     * Given a predictor subset with indices stored in currSubsetIndices, one more predictor from the coefNames
     * that was not found in currSubsetIndices was added to the subset to form a new Training frame. An array of 
     * training frame are built training frames from all elligible predictors from coefNames.  GLMParameters are
     * built for all TrainingFrames, GLM models are built from those parameters.  The GLM model with the highest 
     * R2 value is returned.  The added predictor which resulted in the highest R2 value will be added to 
     * currSubsetIndices.
     * 
     * see doc at https://h2oai.atlassian.net/browse/PUBDEV-8444 for details.
     *
     * @param currSubsetIndices: stored predictors that are chosen in the subset
     * @param coefNames: predictor names of full training frame
     * @param predPos: index/location of predictor to be added into currSubsetIndices
     * @return GLMModel with highest R2 value
     */
    public static GLMModel forwardStep(List<Integer> currSubsetIndices, List<String> coefNames, int predPos, 
                                       List<Integer> validSubsets, ModelSelectionModel.ModelSelectionParameters parms,
                                       String foldColumn, int glmNFolds, 
                                       Model.Parameters.FoldAssignmentScheme foldAssignment, Set<BitSet> usedCombo) {
        String[] predictorNames = coefNames.stream().toArray(String[]::new);
        // generate training frames
        Frame[] trainingFrames = generateMaxRTrainingFrames(parms, predictorNames, foldColumn,
                currSubsetIndices, predPos, validSubsets, usedCombo);
        if (trainingFrames.length > 0) {
            // find GLM model with best R2 value and return it
            GLMModel bestModel = buildExtractBestR2Model(trainingFrames, parms, glmNFolds, foldColumn, foldAssignment);
            List<String> coefUsed = extraModelColumnNames(coefNames, bestModel);
            for (int predIndex = coefUsed.size() - 1; predIndex >= 0; predIndex--) {
                int index = coefNames.indexOf(coefUsed.get(predIndex));
                if (!currSubsetIndices.contains(index)) {
                    currSubsetIndices.add(predPos, index);
                    break;
                }
            }
            removeTrainingFrames(trainingFrames);
            return bestModel;
        } else {
            return null;
        }
    }

    /**
     * Contains information of a predictor subsets like predictor indices of the subset (with the newest predictor as
     * the last element of the array), CPM associated with predictor subset minus the latest element, sweep vector
     * arrays generated in sweeping the CPM and the error variance of the CPM.
     */
    public static class SweepModel {
        int[] _predSubset;
        double[][] _CPM;
        SweepVector[][] _sweepVector;
        double _errorVariance;
        
        public SweepModel(int[] predSubset, double[][] cpm, SweepVector[][] sVector, double mse) {
            _predSubset =  predSubset;
            _CPM = cpm;
            _sweepVector = sVector;
            _errorVariance = mse;
        }
    }

    public static class SweepModel2 {
        int[] _predSubset;
        List<Integer> _replacementPredSubset;
        List<SweepInfo> _sweepInfo;
        double _errorVariance;
        int _cpmSize;   // actual part of cpm that is actually used

        public SweepModel2(int[] predSubset, List<SweepInfo> sInfo, double mse, int cpmSize) {
            _predSubset =  predSubset;
            _sweepInfo = sInfo;
            _errorVariance = mse;
            _cpmSize = cpmSize;
        }
    }
    
    public static class SweepInfo {
        double[][][] _cpm;  // one per sweep
        SweepVector[][] _sweepVector;
        int[] _sweepIndices;
        
        public SweepInfo(double[][][] cpm, SweepVector[][] sVector, int[] sIndices) {
            _cpm = cpm;
            _sweepVector = sVector;
            _sweepIndices = sIndices;
        }
    }

    /***
     *
     * Implement forward model using the incremental sweep method with sweep vector arrays as found in section
     * V.II.IV of doc.
     * 
     * A sweepModel is generated to contain information of the best predictor subset found.
     * 
     */
    public static SweepModel forwardStep(List<Integer> currSubsetIndices, List<String> coefNames,
                                         List<Integer> validSubsets, Set<BitSet> usedCombo, double[][] origCPM,
                                         int[][] predInd2CPMInd, SweepModel bestModel, boolean hasIntercept) {
        String[] predictorNames = coefNames.stream().toArray(String[]::new);
        // generate all models
        List<Integer[]> predSubsetList = new ArrayList<>(); // store all predictor subsets generated by sweep
        
        double[] subsetErrVar = bestModel==null? generateAllErrorVariances(origCPM, null, null, predictorNames, 
                currSubsetIndices, validSubsets, usedCombo, predInd2CPMInd, hasIntercept, predSubsetList, null, null, false):
                generateAllErrorVariances(origCPM, bestModel._sweepVector, bestModel._CPM, predictorNames, currSubsetIndices, 
                        validSubsets, usedCombo, predInd2CPMInd, hasIntercept, predSubsetList, null, null, false);
        
        // find the best subset and the corresponding cpm by checking for lowest error variance
        int bestInd = -1;
        double errorVarianceMin = Double.MAX_VALUE;
        int numModel = subsetErrVar.length;
        for (int index=0; index<numModel; index++) {
            if (subsetErrVar[index] < errorVarianceMin) {
                errorVarianceMin = subsetErrVar[index];
                bestInd = index;
            }
        }
        if (bestInd == -1) { // Predictor sets are duplicates.  Return SweepModel for findBestMSEModel stream operations
            return new SweepModel(null, null, null, errorVarianceMin);
        } else {    // set new predictor subset to curSubsetIndices, 
            currSubsetIndices.clear();
            Arrays.stream(predSubsetList.get(bestInd)).map(x -> currSubsetIndices.add(x)).collect(Collectors.toList());
            int[] subsetPred = Arrays.stream(predSubsetList.get(bestInd)).mapToInt(Integer::intValue).toArray();
            return new SweepModel(subsetPred, null, null, errorVarianceMin);
        }
    }
    
    public static int[] concatSweepIndices(List<SweepInfo> sInfoList) {
        int[] largeSIndices = new int[sumNumSV(sInfoList, sInfoList.size())];
        int numInfo = sInfoList.size();
        int offset = 0;
        for (int index = 1; index < numInfo; index++) {
            SweepInfo oneInfo = sInfoList.get(index);
            int sweepIndLen = oneInfo._sweepIndices.length;
            System.arraycopy(oneInfo._sweepIndices, 0, largeSIndices, offset, sweepIndLen);
            offset += sweepIndLen;
        }
        return largeSIndices;   
    }

    public static SweepModel2 forwardStepO(List<Integer> currSubsetIndices, List<String> coefNames,
                                         List<Integer> validSubsets, Set<BitSet> usedCombo, double[][] origCPM,
                                         int[][] predInd2CPMInd, SweepModel2 bestModel, boolean hasIntercept, 
                                           boolean forReplacement, int maxPredNum) {
        String[] predictorNames = coefNames.stream().toArray(String[]::new);
        // generate all models
        List<Integer[]> predSubsetList = new ArrayList<>(); // store all predictor subsets generated by sweep
        double[] subsetErrVar = null;
        if (bestModel == null) {
            subsetErrVar = generateAllErrorVariances(origCPM, null, null, predictorNames, currSubsetIndices, 
                    validSubsets, usedCombo, predInd2CPMInd, hasIntercept, predSubsetList, null, null, forReplacement);
        } else {
            SweepVector[][] accumSV = concateSweepVectors(bestModel._sweepInfo, (bestModel._cpmSize+1)*2);
            SweepInfo sInfo = bestModel._sweepInfo.get(bestModel._sweepInfo.size()-1);
            double[][] cpm = clone2D(sInfo._cpm[sInfo._cpm.length-1], bestModel._cpmSize);
            int[] sweepIndices = concatSweepIndices(bestModel._sweepInfo);
            subsetErrVar = generateAllErrorVariances(origCPM, accumSV, cpm, predictorNames, currSubsetIndices,
                    validSubsets, usedCombo, predInd2CPMInd, hasIntercept, predSubsetList, bestModel._predSubset, sweepIndices, forReplacement);
        }
        
        // find the best subset and the corresponding cpm by checking for lowest error variance
        int bestInd = -1;
        double errorVarianceMin = Double.MAX_VALUE;
        int numModel = subsetErrVar.length;
        for (int index=0; index<numModel; index++) {
            if (subsetErrVar[index] < errorVarianceMin) {
                errorVarianceMin = subsetErrVar[index];
                bestInd = index;
            }
        }
        if (bestInd == -1) { // Predictor sets are duplicates.  Can happen with replacement step
            return new SweepModel2(null, null, Double.POSITIVE_INFINITY, 0);
        } else {    // set new predictor subset to curSubsetIndices, 
            List<Integer> origCurrSubset = null;
            if (forReplacement)
                origCurrSubset = new ArrayList<>(currSubsetIndices);
            currSubsetIndices.clear();
            Arrays.stream(predSubsetList.get(bestInd)).map(x -> currSubsetIndices.add(x)).collect(Collectors.toList());
            int[] subsetPred = Arrays.stream(predSubsetList.get(bestInd)).mapToInt(Integer::intValue).toArray(); // currest best set with new predictor
            // generate and save cpm, sweep vectors for next round
            if (forReplacement) {
                bestModel._errorVariance = errorVarianceMin;
                bestModel._predSubset = subsetPred;
                return bestModel;
            } else {
                return genUpdateCPMSweepVectors(subsetPred, bestModel, origCPM, predInd2CPMInd, hasIntercept, 
                        origCurrSubset, forReplacement, maxPredNum);
            }
        }
    }
    

    // for maxrsweep
    public static List<Integer> forwardStep(List<Integer> currSubsetIndices, List<String> coefNames,
                                            List<Integer> validSubsets, Set<BitSet> usedCombo, double[][] currCPM,
                                            int[][] predInd2CPMInd, int predPos) {
        String[] predictorNames = coefNames.stream().toArray(String[]::new);
        // generate all models
        List<Integer[]> predSubsetList = new ArrayList<>(); // store all predictor subsets generated by sweep

        double[] subsetErrVar = generateAllErrorVariances(currCPM, predictorNames, currSubsetIndices,
                validSubsets, usedCombo, predSubsetList, predInd2CPMInd, predPos);

        if (predSubsetList.size() > 0) {    // size will be zero if all subcarrier subsets are duplicates
            // find the best subset and the corresponding cpm by checking for lowest error variance
            int bestInd = Arrays.stream(subsetErrVar).boxed().collect(Collectors.toList()).indexOf(
                    Arrays.stream(subsetErrVar).min().getAsDouble());
            currSubsetIndices.clear();
            Arrays.stream(predSubsetList.get(bestInd)).map(x -> currSubsetIndices.add(x)).collect(Collectors.toList());
           // sweepCPM(currCPM, predInd2CPMInd[currSubsetIndices.get(predPos)], false);
            return currSubsetIndices;
        } else {
            return null;
        }
    }

    /**
     * consider the predictors in subset as pred0, pred1, pred2 (using subset size 3 as example):
     *    a. Keep pred1, pred2 and replace pred0 with remaining predictors, find replacement with highest R2
     *    b. keep pred0, pred2 and replace pred1 with remaining predictors, find replacement with highest R2
     *    c. keeep pred0, pred1 and replace pred2 with remaining predictors, find replacement with highest R2
     *    d. from step 2a, 2b, 2c, choose the predictor subset with highest R2, say the subset is pred0, pred4, pred2
     *    e. Take subset pred0, pred4, pred2 and go through steps a,b,c,d again and only stop when the best R2 does
     *       not improve anymore.
     *
     * see doc at https://h2oai.atlassian.net/browse/PUBDEV-8444 for details.
     *
     * The most important thing here is to make sure validSubset contains the true eligible predictors to choose
     * from.  Inside the for loop, I will remove and add predictors that have been chosen in oneLessSubset and add
     * the removed predictor back to validSubset after it is no longer selected in the predictor subset.
     *
     * I also will reset the validSubset from time to time to just start to include all predictors as 
     * valid, then I will remove all predictors that have been chosen in currSubsetIndices.
     *
     */
    public SweepModel replacement(List<Integer> currSubsetIndices, List<String> coefNames, List<Integer> validSubset,
                                  Set<BitSet> usedCombos, SweepModel bestModel) {
        double errorVarianceMin = bestModel._errorVariance;
        int currSubsetSize = currSubsetIndices.size();  // predictor subset size
        int lastBestErrVarPosIndex=-1;
        int[] msePredPosIndex = new int[currSubsetSize];
        SweepModel[] bestModels = new SweepModel[currSubsetSize];
        SweepModel currModel = new SweepModel(bestModel._predSubset, bestModel._CPM, bestModel._sweepVector,
                bestModel._errorVariance);
        SweepModel bestErrVarModel = new SweepModel(bestModel._predSubset, bestModel._CPM, bestModel._sweepVector, 
                bestModel._errorVariance);
        while (true) {  // loop to find better predictor subset via sequential replacement
            for (int index=0; index<currSubsetSize; index++) {
                if (index != lastBestErrVarPosIndex) {
                    ArrayList<Integer> oneLessSubset = new ArrayList<>(findLastModelpredSub(index, bestModels, currSubsetIndices));
                    int removedSubInd = oneLessSubset.remove(index);
                    validSubset.removeAll(oneLessSubset);
                    currModel._predSubset = oneLessSubset.stream().mapToInt(x->x).toArray();
                    genBestSweepVector(currModel, _crossProdcutMatrix, _predictorIndex2CPMIndices, _parms._intercept);
                    bestModels[index] = forwardStep(oneLessSubset, coefNames, validSubset, usedCombos, 
                            _crossProdcutMatrix, _predictorIndex2CPMIndices, currModel, _parms._intercept);
                    validSubset.add(removedSubInd);
                    msePredPosIndex[index] = index;
                }
            }
            int bestErrVarIndex = findBestMSEModel(errorVarianceMin, bestModels);
            if (bestErrVarIndex < 0) {
                validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                validSubset.removeAll(currSubsetIndices);
                break;
            } else {
                bestErrVarModel = bestModels[bestErrVarIndex];
                errorVarianceMin = bestErrVarModel._errorVariance;
                currSubsetIndices = Arrays.stream(bestErrVarModel._predSubset).boxed().collect(Collectors.toList());
                lastBestErrVarPosIndex = msePredPosIndex[bestErrVarIndex];
                validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                validSubset.removeAll(currSubsetIndices);
            }
        }
        return bestErrVarModel;
    }

    public SweepModel2 replacementO(List<Integer> currSubsetIndices, List<String> coefNames, List<Integer> validSubset,
                                  Set<BitSet> usedCombos, SweepModel2 bestModel) {
        double errorVarianceMin = bestModel._errorVariance;
        int currSubsetSize = currSubsetIndices.size();  // predictor subset size
        int lastBestErrVarPosIndex=-1;
        int[] msePredPosIndex = new int[currSubsetSize];
        SweepModel2[] bestModels = new SweepModel2[currSubsetSize];
        SweepModel2 bestErrVarModel = bestModel;
        SweepModel2 currModel = new SweepModel2(bestModel._predSubset.clone(), cloneSweepInfo(bestModel._sweepInfo, bestModel._cpmSize),
                bestModel._errorVariance, bestModel._cpmSize);
        SweepModel2 tempModel;
        int maxPredNum = _parms._max_predictor_number;
        while (true) {  // loop to find better predictor subset via sequential replacement
            for (int index=0; index<currSubsetSize; index++) {  // go through all predictor positions
                if (index != lastBestErrVarPosIndex) {
                    ArrayList<Integer> oneLessSubset = new ArrayList<>(currSubsetIndices);
                    int removedSubInd = oneLessSubset.remove(index);
                    validSubset.removeAll(oneLessSubset);
                    // sweep with removed pred to remove its effect and add the new info back into SweepInfo
                    currModel._sweepInfo.add(genNewSweepInfoR(currModel, removedSubInd, currModel._cpmSize));
                    currModel._predSubset = oneLessSubset.stream().mapToInt(x->x).toArray();
                    tempModel = forwardStepO(new ArrayList<>(currSubsetIndices), coefNames, validSubset, usedCombos,
                            _crossProdcutMatrix, _predictorIndex2CPMIndices, currModel, _parms._intercept, 
                            true, maxPredNum);
                    if (tempModel._predSubset != null) { // if null means duplcated predictor sets
                        currModel = tempModel;
                        List<Integer> currSubsetCopy = new ArrayList<>(currSubsetIndices);
                        currSubsetCopy.add(removedSubInd);
                        currSubsetCopy.add(currModel._predSubset[currModel._predSubset.length - 1]);  // add the newly added best predictor
                        bestModels[index] = new SweepModel2(currModel._predSubset.clone(), null, 
                                currModel._errorVariance, currModel._cpmSize);
                        bestModels[index]._replacementPredSubset = currSubsetCopy;
                    } else {
                        bestModels[index] = new SweepModel2(null, null, Double.POSITIVE_INFINITY, 0);
                    }
                    currModel._predSubset = bestErrVarModel._predSubset.clone();
                    currModel._sweepInfo.remove(currModel._sweepInfo.size()-1);
                    validSubset.add(removedSubInd);
                    msePredPosIndex[index] = index;
                }
            }
            int bestErrVarIndex = findBestMSEModel(errorVarianceMin, bestModels);
            if (bestErrVarIndex < 0) {
                validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                validSubset.removeAll(currSubsetIndices);
                break;
            } else {    // found a better predictor subset
                bestModels[bestErrVarIndex]._sweepInfo = currModel._sweepInfo;
                bestErrVarModel = removeDuplicatePreds(bestModels[bestErrVarIndex], msePredPosIndex[bestErrVarIndex], 
                        _predictorIndex2CPMIndices, _crossProdcutMatrix, _parms._intercept, maxPredNum-bestModel._predSubset.length);    // brand new copy
                errorVarianceMin = bestErrVarModel._errorVariance;
                currSubsetIndices = Arrays.stream(bestErrVarModel._predSubset).boxed().collect(Collectors.toList());
                lastBestErrVarPosIndex = msePredPosIndex[bestErrVarIndex];
                validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                validSubset.removeAll(currSubsetIndices);
                currModel = new SweepModel2(bestErrVarModel._predSubset.clone(), 
                        cloneSweepInfo(bestErrVarModel._sweepInfo), bestErrVarModel._errorVariance, 
                        bestErrVarModel._cpmSize);
            }
        }
        return bestErrVarModel;
    }

    /***
     * In this case, we swept again with the deleted predictor
     */
    public SweepInfo genNewSweepInfo(SweepModel2 sModel, int removedInd) {
        SweepInfo lastInfo = sModel._sweepInfo.get(sModel._sweepInfo.size()-1);
        double[][][] currCPM = clone3D(lastInfo._cpm);
        int[] sweepIndices = locateSInfo(sModel, removedInd);   // locate the sweep indices corresponding to removed predictor
        SweepVector[][] sv = sweepCPM(lastInfo._cpm[lastInfo._cpm.length-1], currCPM, sweepIndices, true);
        SweepInfo newInfo = new SweepInfo(currCPM, sv, sweepIndices);
        return newInfo;
    }

    public SweepInfo genNewSweepInfoR(SweepModel2 sModel, int removedInd) {
        SweepInfo lastInfo = sModel._sweepInfo.get(sModel._sweepInfo.size()-1);
        double[][][] currCPM = clone3D(lastInfo._cpm);
        int[] sweepIndices = locateSInfo(sModel, removedInd);   // locate the sweep indices corresponding to removed predictor
        SweepVector[][] sv = sweepCPM(currCPM[currCPM.length-1], sweepIndices, true);
        SweepInfo newInfo = new SweepInfo(currCPM, sv, sweepIndices);
        return newInfo;
    }

    public SweepInfo genNewSweepInfoR(SweepModel2 sModel, int removedInd, int actualCPMSize) {
        SweepInfo lastInfo = sModel._sweepInfo.get(sModel._sweepInfo.size()-1);
        double[][][] currCPM = clone3D(lastInfo._cpm, actualCPMSize);
        int[] sweepIndices = locateSInfo(sModel, removedInd);   // locate the sweep indices corresponding to removed predictor
        SweepVector[][] sv = sweepCPM(currCPM[currCPM.length-1], sweepIndices, true, actualCPMSize);
        SweepInfo newInfo = new SweepInfo(currCPM, sv, sweepIndices);
        return newInfo;
    }

    // for maxrsweep
    public List<Integer> replacement(List<Integer> currSubsetIndices, List<String> coefNames, List<Integer> validSubset,
                                  Set<BitSet> usedCombos, double[][] currCPM, int[][] predInd2CPMInd) {
        int lastCPMIndex = currCPM.length-1;
        double minMSE = currCPM[lastCPMIndex][lastCPMIndex];  // best mse so far from forward step
        int currSubsetSize = currSubsetIndices.size();  // predictor subset size
        int lastBestErrVarPosIndex=-1;
        int[] msePredPosIndex = new int[currSubsetSize];
        double[] subsetMSEs = new double[currSubsetSize];
        List<List<Integer>> allPredSubsets = new ArrayList<>();
        List<Integer> currPredSubsets;
        
        while (true) {  // loop to find better predictor subset via sequential replacement
            for (int index=0; index<currSubsetSize; index++) {
                if (index != lastBestErrVarPosIndex) {
                    ArrayList<Integer> oneLessSubset = new ArrayList<>(currSubsetIndices);
                    int removedSubInd = oneLessSubset.remove(index);
                    validSubset.removeAll(oneLessSubset);
                    sweepCPM(currCPM, predInd2CPMInd[removedSubInd], false); // undo predictor
                    currPredSubsets = forwardStep(oneLessSubset, coefNames, validSubset, usedCombos, currCPM, predInd2CPMInd, index);
                    if (currPredSubsets == null) {
                        subsetMSEs[index] = Double.MAX_VALUE;
                        allPredSubsets.add(new ArrayList<>());
                        sweepCPM(currCPM, predInd2CPMInd[removedSubInd], false); // redo predictor
                    } else {
                        subsetMSEs[index] = currCPM[lastCPMIndex][lastCPMIndex];
                        validSubset.add(removedSubInd);
                        allPredSubsets.add(currPredSubsets);
                        if (subsetMSEs[index] < minMSE) {   // apply sweep from forward if found lower MSE
                            sweepCPM(currCPM, predInd2CPMInd[currPredSubsets.get(index)], false);
                            currSubsetIndices = new ArrayList<>(currPredSubsets);
                            minMSE = subsetMSEs[index];
                        } else {    // undo the sweep by the chosen new predictor
                            sweepCPM(currCPM, predInd2CPMInd[removedSubInd], false);
                        }
                    }
                    msePredPosIndex[index] = index;
                }
            }
            int bestErrVarIndex = findBestMSEModelIndex(minMSE, subsetMSEs);
            if (bestErrVarIndex < 0) {
                validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                validSubset.removeAll(currSubsetIndices);
                break;
            } else {
                minMSE = subsetMSEs[bestErrVarIndex];
                currSubsetIndices = allPredSubsets.get(bestErrVarIndex);
                lastBestErrVarPosIndex = msePredPosIndex[bestErrVarIndex];
                validSubset = IntStream.rangeClosed(0, coefNames.size() - 1).boxed().collect(Collectors.toList());
                validSubset.removeAll(currSubsetIndices);
            }
        }
        return currSubsetIndices;
    }
    
    public static GLMModel forwardStep(List<Integer> currSubsetIndices, List<String> coefNames, int predPos,
                                       List<Integer> validSubsets, ModelSelectionModel.ModelSelectionParameters parms,
                                       String foldColumn, int glmNFolds,
                                       Model.Parameters.FoldAssignmentScheme foldAssignment) {
        return forwardStep(currSubsetIndices, coefNames, predPos, validSubsets, parms, foldColumn, glmNFolds, 
                foldAssignment, null);
    }

    /**
     * consider the predictors in subset as pred0, pred1, pred2 (using subset size 3 as example):
     *    a. Keep pred1, pred2 and replace pred0 with remaining predictors, find replacement with highest R2
     *    b. keep pred0, pred2 and replace pred1 with remaining predictors, find replacement with highest R2
     *    c. keeep pred0, pred1 and replace pred2 with remaining predictors, find replacement with highest R2
     *    d. from step 2a, 2b, 2c, choose the predictor subset with highest R2, say the subset is pred0, pred4, pred2
     *    e. Take subset pred0, pred4, pred2 and go through steps a,b,c,d again and only stop when the best R2 does
     *       not improve anymore.
     *       
     * see doc at https://h2oai.atlassian.net/browse/PUBDEV-8444 for details.
     *
     */
    public static GLMModel replacement(List<Integer> currSubsetIndices, List<String> coefNames,
                                       double bestR2, ModelSelectionModel.ModelSelectionParameters parms,
                                       int glmNFolds, String foldColumn, List<Integer> validSubset,
                                       Model.Parameters.FoldAssignmentScheme foldAssignment, Set<BitSet> usedCombos) {
        int currSubsetSize = currSubsetIndices.size();
        int lastBestR2PosIndex = -1;
        GLMModel bestR2Model = null;
        GLMModel[] bestR2Models = new GLMModel[currSubsetSize];
        int[] r2PredPosIndex = new int[currSubsetSize];
        int[][] subsetsCombo = new int[currSubsetSize][];
        List<Integer> originalSubset = new ArrayList<>(currSubsetSize);
        while (true) {
            for (int index = 0; index < currSubsetSize; index++) {  // go through all predictor position in subset
                if (index != lastBestR2PosIndex) {
                    ArrayList<Integer> oneLessSubset = new ArrayList<>(currSubsetIndices);
                    int predIndexRemoved = oneLessSubset.remove(index);
                    validSubset.add(predIndexRemoved);
                    bestR2Models[index] = forwardStep(oneLessSubset, coefNames, index, validSubset, parms,
                            foldColumn, glmNFolds, foldAssignment, usedCombos);
                    subsetsCombo[index] = oneLessSubset.stream().mapToInt(i->i).toArray();
                    r2PredPosIndex[index] = index;
                    validSubset.remove(validSubset.indexOf(predIndexRemoved));
                }
            }
            int bestR2ModelIndex = findBestR2Model(bestR2, bestR2Models);
            if (bestR2ModelIndex < 0) {  // done with replacement step
                break;
            } else {
                bestR2Model = bestR2Models[bestR2ModelIndex];
                bestR2 = bestR2Model.r2();
                currSubsetIndices = Arrays.stream(subsetsCombo[bestR2ModelIndex]).boxed().collect(Collectors.toList()); 
                lastBestR2PosIndex = r2PredPosIndex[bestR2ModelIndex];
                updateValidSubset(validSubset, originalSubset, currSubsetIndices);
                originalSubset = currSubsetIndices; // copy over new subset
            }
        }
        return bestR2Model;
    }
}
