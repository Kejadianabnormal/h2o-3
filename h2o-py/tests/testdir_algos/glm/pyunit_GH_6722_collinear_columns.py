import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from tests import pyunit_utils

# The purpose of this test to make sure that constraint GLM works in the presence of collinear columns in the dataset.
# The coefficient names are Intercept, C1.c0.I1, C2.c1.I1, C10, C20, C3, C4, C5, C6, corr1, corr2.  The last two 
# columns are redundant.  The python dict contains: {'Intercept': -0.010590769679846715, 'C1.c0.l1': 0.07588584170223357,
# 'C2.c1.l1': -0.0024769212195866856, 'C10': 0.0002122276247145693, 'C20': 4.180627792435957e-08, 
# 'C3': -0.0002736922405934083, 'C4': 1.5860604693698343e-08, 'C5': 2.0049884308592165e-08, 'C6': 3.660281180014312e-08,
# 'corr1': 0.0, 'corr2': 0.0}
def test_constraints_collinear_columns():
    # first two columns are enums, the last 4 are real columns
    h2o_data = pyunit_utils.genTrainFrame(10000, 6, enumCols=2, enumFactors=2, responseLevel=2, miscfrac=0, randseed=12345)
    # create extra collinear columns
    num1 = h2o_data[2]*0.2-0.5*h2o_data[3]
    num2 = -0.8*h2o_data[4]+0.1*h2o_data[5]
    h2o_collinear = num1.cbind(num2)
    h2o_collinear.set_names(["corr1", "corr2"])
    train_data = h2o_data.cbind(h2o_collinear)
    y = "response"
    x = train_data.names
    x.remove(y)
    bc = []
    lc = []

    name = "C2.c1.l1" # for beta constraints
    lower_bound = -0.02
    upper_bound = 0.5
    bc.append([name, lower_bound, upper_bound])

    name = "C10"
    values = 1
    types = "LessThanEqual"
    contraint_numbers = 0
    lc.append([name, values, types, contraint_numbers])

    name = "C4"
    values = 1
    types = "LessThanEqual"
    contraint_numbers = 0
    lc.append([name, values, types, contraint_numbers])

    name = "constant"
    values = -2
    types = "LessThanEqual"
    contraint_numbers = 0
    lc.append([name, values, types, contraint_numbers])


    beta_constraints = h2o.H2OFrame(bc)
    beta_constraints.set_names(["names", "lower_bounds", "upper_bounds"])
    linear_contraints = h2o.H2OFrame(lc)
    linear_contraints.set_names(["names", "values", "types", "constraint_numbers"])

#    h2o_glm = H2OGeneralizedLinearEstimator(family="binomial", nfolds=10, alpha=0.5, beta_constraints=beta_constraints)
    h2o_glm = H2OGeneralizedLinearEstimator(family="binomial", compute_p_values=True, remove_collinear_columns=True,
                                            lambda_=0.0, solver="irlsm", beta_constraints=beta_constraints, 
                                            linear_constraints=linear_contraints)
    h2o_glm.train(x=x, y=y, training_frame=train_data )

    print("Done")



if __name__ == "__main__":
    pyunit_utils.standalone_test(test_constraints_collinear_columns)
else:
    test_constraints_collinear_columns()
