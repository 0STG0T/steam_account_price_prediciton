import math

class UserDefinedMetric(object):
    def __init__(self, penalty_factor=10.0):
        """
        Parameters
        ----------
        penalty_factor : float, optional (default=10.0)
            Factor by which negative predictions are penalized. Use a large value to strongly discourage negatives.
        """
        self.penalty_factor = penalty_factor

    def is_max_optimal(self):
        """
        Indicates whether higher values of the metric are better.
        Since RMSE measures error, lower is better.
        """
        return False

    def evaluate(self, approxes, target, weight):
        """
        Evaluates the RMSE with a strong penalty for negative predictions.

        Parameters
        ----------
        approxes : list of indexed containers of float
            Vectors of predicted labels.

        target : one-dimensional indexed container of float
            Vectors of true labels.

        weight : one-dimensional indexed container of float, optional (default=None)
            Weight for each instance.

        Returns
        -------
        tuple (metric value, total weight) :
            The penalized RMSE and total weight.
        """
        squared_error_sum = 0.0
        total_weight = 0.0

        for i in range(len(target)):
            prediction = approxes[0][i]  # CatBoost provides approxes as a list of lists
            actual = target[i]
            w = 1.0 if weight is None else weight[i]

            # Compute squared error
            squared_error = (prediction - actual) ** 2

            # Add a very strong penalty if prediction is negative
            if prediction < 0:
                penalty = self.penalty_factor * (-prediction) ** 2
                squared_error += penalty

            squared_error_sum += squared_error * w
            total_weight += w

        return squared_error_sum, total_weight

    def get_final_error(self, error, weight):
        """
        Returns the final RMSE with penalties.

        Parameters
        ----------
        error : float
            Sum of squared errors and penalties.

        weight : float
            Sum of weights for all instances.

        Returns
        -------
        metric value : float
            Weighted RMSE with penalties.
        """
        return math.sqrt(error / weight) if weight > 0 else float('nan')
    
class UserDefinedObjective(object):
    def __init__(self, penalty_factor=10.0):
        """
        Parameters
        ----------
        penalty_factor : float, optional (default=10.0)
            Factor by which negative predictions are penalized.
        """
        self.penalty_factor = penalty_factor

    def calc_ders_range(self, approxes, targets, weights):
        """
        Computes first (gradient) and second (hessian) derivatives of the custom loss function.

        Parameters
        ----------
        approxes : indexed container of floats
            Current predictions for each object.

        targets : indexed container of floats
            Target values provided in the dataset.

        weights : indexed container of floats, optional (default=None)
            Instance weights.

        Returns
        -------
        list of tuples:
            Each tuple contains (gradient, hessian) for the loss function for each data point.
        """
        der1 = []
        der2 = []

        for i in range(len(targets)):
            pred = approxes[i]
            target = targets[i]
            weight = 1.0 if weights is None else weights[i]

            # Standard RMSE derivative
            error = pred - target
            grad = 2 * error * weight
            hess = 2 * weight

            # Add penalty for negative predictions
            if pred < 0:
                grad += 2 * self.penalty_factor * pred * weight
                hess += 2 * self.penalty_factor * weight

            der1.append(grad)
            der2.append(hess)

        return list(zip(der1, der2))