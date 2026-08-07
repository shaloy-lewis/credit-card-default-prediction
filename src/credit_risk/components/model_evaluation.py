import os
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from credit_risk.exception.exception import customexception
from credit_risk.logger.logging import logging
from credit_risk.utils.utils import load_object


class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation initiated")

    def evaluate_model(self, true, pred):
        roc_auc = roc_auc_score(true, pred)
        recall = recall_score(true, np.round_(pred))
        precision = precision_score(true, np.round_(pred))
        f1 = f1_score(true, np.round_(pred))

        return roc_auc, recall, precision, f1

    def initiate_model_evaluation(self, train_array, test_array, train_target, test_target):
        try:
            logging.info("Model loading in progress")
            model_path = os.path.join("artifacts", "model.pkl")
            best_model = load_object(model_path)

            logging.info("Calculating model predictions")
            y_train_pred = best_model.predict_proba(train_array)[:, 1]
            y_pred = best_model.predict_proba(test_array)[:, 1]

            logging.info("Evaluating model")
            roc_auc_train, recall_train, precision_train, f1_train = self.evaluate_model(
                train_target, y_train_pred
            )
            roc_auc_test, recall_test, precision_test, f1_test = self.evaluate_model(
                test_target, y_pred
            )

            logging.info("Model report:")
            logging.info(f"Train ROC-AUC: {roc_auc_train}\t Test ROC-AUC: {roc_auc_test}")
            logging.info(f"Train recall: {recall_train}\t Test recall: {recall_test}")
            logging.info(f"Train precision: {precision_train}\t Test precision: {precision_test}")
            logging.info(f"Train F1-score: {f1_train}\t Test F1-score: {f1_test}")

        except Exception as e:
            logging.info("Exception occured in model evaluation")
            raise customexception(e, sys) from e
